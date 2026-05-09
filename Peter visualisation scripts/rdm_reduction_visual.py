import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import images, categories, control_conditions, conditions
from src.plot_config import apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT

apply_defaults()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_fig_dir  = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Load behavioral RDM (80×80 similarity → dissimilarity) ───────────────────

avg_sim = pd.read_table(
    os.path.join(project_root, 'output', 'averaged', 'average_similarity_rdm.tsv'),
    index_col=0
).reindex(index=images, columns=images)

beh_80_raw = avg_sim.values.astype(float)  # already dissimilarity (0 = self, high = dissimilar)

beh_80 = (beh_80_raw - np.nanmin(beh_80_raw)) / (np.nanmax(beh_80_raw) - np.nanmin(beh_80_raw))

# ── Reduce behavioral to 8×8 (category means) ────────────────────────────────

n_cats = len(categories)
beh_8_raw = np.zeros((n_cats, n_cats))
for i, c1 in enumerate(categories):
    for j, c2 in enumerate(categories):
        imgs1 = [img for img in images if c1 in img]
        imgs2 = [img for img in images if c2 in img]
        beh_8_raw[i, j] = avg_sim.loc[imgs1, imgs2].values.mean()

beh_8 = (beh_8_raw - np.nanmin(beh_8_raw)) / (np.nanmax(beh_8_raw) - np.nanmin(beh_8_raw))

# ── Load neural RDMs and average across subjects ──────────────────────────────

ROI      = 'Ventral_Stream_Visual'
subjects = [f'sub-{str(n+1).zfill(2)}' for n in range(20)]

# 40×40 (all 5 conditions, bilateral average)
neural_40_list = []
for s in subjects:
    l = pd.read_table(
        os.path.join(project_root, 'models', 'subject_rois', f'{s}_L_{ROI}.tsv'),
        index_col=0
    ).reindex(index=conditions, columns=conditions)
    r = pd.read_table(
        os.path.join(project_root, 'models', 'subject_rois', f'{s}_R_{ROI}.tsv'),
        index_col=0
    ).reindex(index=conditions, columns=conditions)
    neural_40_list.append(((l + r) / 2).values)

neural_40_raw = np.nanmean(neural_40_list, axis=0)
np.fill_diagonal(neural_40_raw, 0)
neural_40 = (neural_40_raw - neural_40_raw.min()) / (neural_40_raw.max() - neural_40_raw.min())

# 8×8 (category-averaged control conditions, bilateral average)
neural_8_list = []
for s in subjects:
    l = pd.read_table(
        os.path.join(project_root, 'models', 'subject_avg_category_rois', f'{s}_L_{ROI}.tsv'),
        index_col=0
    ).reindex(index=control_conditions, columns=control_conditions)
    r = pd.read_table(
        os.path.join(project_root, 'models', 'subject_avg_category_rois', f'{s}_R_{ROI}.tsv'),
        index_col=0
    ).reindex(index=control_conditions, columns=control_conditions)
    neural_8_list.append(((l + r) / 2).values)

neural_8_raw = np.nanmean(neural_8_list, axis=0)
np.fill_diagonal(neural_8_raw, 0)
neural_8 = (neural_8_raw - neural_8_raw.min()) / (neural_8_raw.max() - neural_8_raw.min())

# ── Short category labels for 8×8 tick marks ─────────────────────────────────

CAT_LABELS = [c.replace('firehydrant', 'hydrant').replace('building', 'bldg') for c in categories]

# ── Plot ──────────────────────────────────────────────────────────────────────

CMAP = 'gray'

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8),
                         gridspec_kw={'wspace': 0.38})

panels = [
    (beh_80,    'Behavioral RDM\n(80×80, intact images)',     None,       'items'),
    (neural_40, 'Neural RDM\n(40×40, all conditions)',         None,       'conditions'),
    (beh_8,     'Behavioral RDM\n(8×8, category means)',       CAT_LABELS, 'categories'),
    (neural_8,  'Neural RDM\n(8×8, category means)',           CAT_LABELS, 'categories'),
]

for ax, (data, title, tick_labels, unit) in zip(axes, panels):
    im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=1,
                   origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=APA_LABEL, fontweight='bold', pad=6)

    if tick_labels is not None:
        ticks = np.arange(len(tick_labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=APA_ANNOT)
        ax.set_yticklabels(tick_labels, fontsize=APA_ANNOT)
    else:
        n = data.shape[0]
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        ax.set_xticklabels(['1', str(n)], fontsize=APA_ANNOT)
        ax.set_yticklabels(['1', str(n)], fontsize=APA_ANNOT)
        ax.set_xlabel(unit, fontsize=APA_TICK)
        ax.set_ylabel(unit, fontsize=APA_TICK)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    ax.tick_params(length=2, width=0.6, color='#888888')

# Shared colorbar
cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
cbar.set_label('Dissimilarity (normalised)', fontsize=APA_ANNOT)
cbar.ax.tick_params(labelsize=APA_ANNOT)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['low', 'mid', 'high'])


save_path = os.path.join(out_fig_dir, 'rdm_reduction_visual.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()
