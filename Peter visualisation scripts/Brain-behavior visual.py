import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from src.plot_config import apply_defaults, BRAIN_BEH_COLOR, NOISE_CEIL_COLOR
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT
from src import control_conditions, images, categories

apply_defaults()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Load real data ─────────────────────────────────────────────────────────────

def _norm(d):
    lo, hi = np.nanmin(d), np.nanmax(d)
    return (d - lo) / (hi - lo) if hi > lo else d

# Behavioral 80×80 (group average)
avg_80 = pd.read_table(
    os.path.join(project_root, 'output', 'averaged', 'average_similarity_rdm.tsv'),
    index_col=0)
beh_80_data = _norm(avg_80.values.astype(float))

# Behavioral 8×8 (category average of 80×80)
beh_8 = np.zeros((len(categories), len(categories)))
for i, c1 in enumerate(categories):
    for j, c2 in enumerate(categories):
        imgs1 = [img for img in images if c1 in img]
        imgs2 = [img for img in images if c2 in img]
        beh_8[i, j] = avg_80.loc[imgs1, imgs2].values.mean()
beh_8_data = _norm(beh_8)

# Neural 40×40: per subject (Ventral Stream Visual, first available hemisphere)
roi_name = 'Ventral_Stream_Visual'
neural_40_data = []
for sub_id in range(1, 21):
    sub = f'sub-{sub_id:02d}'
    for hemi in ['R', 'L']:
        path = os.path.join(project_root, 'models', 'subject_rois',
                            f'{sub}_{hemi}_{roi_name}.tsv')
        if os.path.exists(path):
            df = pd.read_table(path, index_col=0)
            neural_40_data.append(_norm(df.values.astype(float)))
            break
    if len(neural_40_data) == 4:
        break

# Neural 8×8: bilateral average per subject (control condition only)
neural_8_data = []
for sub_id in range(1, 21):
    sub = f'sub-{sub_id:02d}'
    l_path = os.path.join(project_root, 'models', 'subject_avg_category_rois',
                          f'{sub}_L_{roi_name}.tsv')
    r_path = os.path.join(project_root, 'models', 'subject_avg_category_rois',
                          f'{sub}_R_{roi_name}.tsv')
    if os.path.exists(l_path) and os.path.exists(r_path):
        l = pd.read_table(l_path, index_col=0).values.astype(float)
        r = pd.read_table(r_path, index_col=0).values.astype(float)
        neural_8_data.append(_norm((l + r) / 2))
    if len(neural_8_data) == 4:
        break

# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_styled_box(ax, x, y, width, height, text,
                    facecolor='#f0f0f0', edgecolor='#333333'):
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1",
                                 linewidth=1.5,
                                 edgecolor=edgecolor,
                                 facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text,
            ha='center', va='center',
            fontsize=APA_TICK, fontweight='bold', color='black')


def draw_matrix(ax, x, y, size, data, cmap='gray', zorder=5):
    ax.imshow(data, extent=(x, x + size, y, y + size),
              cmap=cmap, origin='upper', vmin=0, vmax=1, zorder=zorder)
    ax.add_patch(patches.Rectangle((x, y), size, size,
                                   linewidth=1.5, edgecolor='black', facecolor='none',
                                   zorder=zorder + 1))


def draw_stack(ax, x, y, size, data_list, count=4, cmap='magma'):
    """Stacked RDMs bottom-to-top; top layer populated with real data."""
    n = min(count, len(data_list))
    for i in range(n):
        ox = x + i * STACK_OFF
        oy = y + i * STACK_OFF
        if i < n - 1:
            ax.add_patch(patches.Rectangle(
                (ox, oy), size, size,
                linewidth=1, edgecolor='black', facecolor='#dddddd', zorder=i))
        else:
            draw_matrix(ax, ox, oy, size, data_list[0], cmap=cmap, zorder=10)

# ── Layout constants ───────────────────────────────────────────────────────────

STACK_OFF = 0.10
STACK_N   = 4

y_beh = 6.5   # behavioral stream center
y_neu = 2.2   # neural stream center
y_mid = (y_beh + y_neu) / 2   # ≈ 4.35

raw_size  = 1.4
proc_size = 1.0

x_box  = 0.2
x_raw  = 2.8
x_proc = 5.8
x_corr = 8.2
x_out  = 11.5

box_w, box_h   = 2.0, 1.0
corr_w, corr_h = 2.5, 1.0
out_w,  out_h  = 2.5, 2.5

# Derived right edges / centres for stacks
raw_beh_right  = x_raw + raw_size
raw_neu_right  = x_raw + (STACK_N - 1) * STACK_OFF + raw_size
proc_beh_right = x_proc + proc_size
proc_neu_right = x_proc + (STACK_N - 1) * STACK_OFF + proc_size
proc_neu_cy    = y_neu + (STACK_N - 1) * STACK_OFF   # center y of neural 8×8 top layer

# ── Main figure ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8.5)
ax.axis('off')

# ── Column 1: Input boxes ──────────────────────────────────────────────────────
draw_styled_box(ax, x_box, y_beh - box_h / 2, box_w, box_h,
                "Behavioral Data\n(Group Average)")
draw_styled_box(ax, x_box, y_neu - box_h / 2, box_w, box_h,
                "fMRI Data\n(20 Subjects)")

# Arrows: input box → raw matrix column
ax.annotate("", xy=(x_raw, y_beh), xytext=(x_box + box_w, y_beh),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(x_raw, y_neu), xytext=(x_box + box_w, y_neu),
            arrowprops=dict(arrowstyle="->", lw=1.5))

# ── Column 2: Raw matrices ─────────────────────────────────────────────────────
draw_matrix(ax, x_raw, y_beh - raw_size / 2, raw_size, beh_80_data, cmap='gray')
ax.text(x_raw + raw_size / 2, y_beh + raw_size / 2 + 0.15,
        "80×80 RDM", ha='center', va='bottom', fontsize=APA_ANNOT, color='#555555')

draw_stack(ax, x_raw, y_neu - raw_size / 2, raw_size, neural_40_data,
           count=STACK_N, cmap='gray')
ax.text((x_raw + raw_neu_right) / 2,
        y_neu + raw_size / 2 + (STACK_N - 1) * STACK_OFF + 0.15,
        "L+R Avg\nPer ROI", ha='center', va='bottom', fontsize=APA_ANNOT, color='#555555')
ax.text((x_raw + raw_neu_right) / 2, y_neu - raw_size / 2 - 0.30,
        "40×40 Neural RDMs\n(N=20)", ha='center', va='top', fontsize=APA_TICK)

# ── Arrows: raw → processed (collapse labels) ──────────────────────────────────
ax.annotate("", xy=(x_proc, y_beh), xytext=(raw_beh_right, y_beh),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text((raw_beh_right + x_proc) / 2, y_beh + 0.25,
        "Category Avg\n(→8×8)", ha='center', fontsize=APA_ANNOT, color='#555555')

ax.annotate("", xy=(x_proc, y_neu), xytext=(raw_neu_right, y_neu),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text((raw_neu_right + x_proc) / 2, y_neu + 0.25,
        "Control cond.\n(→8×8)", ha='center', fontsize=APA_ANNOT, color='#555555')

# ── Column 3: Processed 8×8 matrices ──────────────────────────────────────────
draw_matrix(ax, x_proc, y_beh - proc_size / 2, proc_size, beh_8_data, cmap='gray')
ax.text(x_proc + proc_size / 2, y_beh - proc_size / 2 - 0.25,
        "Fixed Behavioral Target\n(8×8 RDM)",
        ha='center', va='top', fontsize=APA_TICK, style='italic')

draw_stack(ax, x_proc, y_neu - proc_size / 2, proc_size, neural_8_data,
           count=STACK_N, cmap='gray')
ax.text((x_proc + proc_neu_right) / 2, y_neu - proc_size / 2 - 0.25,
        "Individual Neural RDMs\n(N=20)", ha='center', va='top', fontsize=APA_TICK)

# ── Arrows: processed matrices → Pearson box ──────────────────────────────────
ax.annotate("", xy=(x_corr, y_mid + 0.2), xytext=(proc_beh_right, y_beh),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(x_corr, y_mid - 0.2), xytext=(proc_neu_right, proc_neu_cy),
            arrowprops=dict(arrowstyle="->", lw=1.5))

# ── Column 4: Pearson correlation box ─────────────────────────────────────────
draw_styled_box(ax, x_corr, y_mid - corr_h / 2, corr_w, corr_h,
                "Pearson Correlation\n(Flattened)")
ax.text(x_corr + corr_w / 2, y_mid - corr_h / 2 - 0.2,
        "Repeated 20 times\n(1 Fixed vs 20 Neural)",
        ha='center', va='top', fontsize=APA_ANNOT, style='italic')

# ── Arrow: Pearson → output ────────────────────────────────────────────────────
ax.annotate("", xy=(x_out, y_mid), xytext=(x_corr + corr_w, y_mid),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text((x_corr + corr_w + x_out) / 2, y_mid + 0.2, "r",
        ha='center', va='bottom', fontsize=APA_TICK, style='italic')

# ── Column 5: Output bar plot ──────────────────────────────────────────────────
out_bottom = y_mid - out_h / 2
rect_plot = patches.Rectangle((x_out, out_bottom), out_w, out_h,
                               linewidth=1.5, edgecolor='#333333', facecolor='white')
ax.add_patch(rect_plot)

# Bar
bar_x      = x_out + out_w / 2
bar_height = 1.5
ax.add_patch(patches.Rectangle(
    (bar_x - 0.3, out_bottom), 0.6, bar_height,
    facecolor=BRAIN_BEH_COLOR, edgecolor='none', alpha=0.85, lw=0))

# Error bar: vertical line only (no caps)
err_top = out_bottom + bar_height + 0.2
err_bot = out_bottom + bar_height - 0.2
ax.plot([bar_x, bar_x], [err_bot, err_top], color='black', lw=1)

# Noise ceiling
ceil_y = out_bottom + out_h - 0.2
ax.add_patch(patches.Rectangle(
    (x_out, ceil_y - 0.1), out_w, 0.2, facecolor=NOISE_CEIL_COLOR, alpha=0.2))
ax.plot([x_out, x_out + out_w], [ceil_y, ceil_y], color=NOISE_CEIL_COLOR, lw=1.5)
ax.text(x_out + out_w + 0.1, ceil_y, "Noise\nCeiling",
        ha='left', va='center', fontsize=APA_ANNOT, color=NOISE_CEIL_COLOR)

ax.text(bar_x, out_bottom - 0.15, "ROI Name",
        ha='center', va='top', fontsize=APA_ANNOT)
ax.text(x_out + out_w / 2, out_bottom + out_h + 0.2, "Result: Mean r ± SD",
        ha='center', fontweight='bold', fontsize=APA_TICK)

plt.tight_layout()
out_fig_dir = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)
plt.savefig(os.path.join(out_fig_dir, 'brain_behavior_pipeline.png'),
            dpi=300, bbox_inches='tight')
plt.show()
