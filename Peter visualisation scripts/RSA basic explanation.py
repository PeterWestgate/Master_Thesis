import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from src.plot_config import apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT
from src import control_conditions

apply_defaults()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_fig_dir  = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Load real data for the matrix panels ──────────────────────────────────────

def _load_8x8(path):
    df = pd.read_table(path, index_col=0).reindex(
        index=control_conditions, columns=control_conditions)
    d = df.values.astype(float)
    lo, hi = np.nanmin(d), np.nanmax(d)
    return (d - lo) / (hi - lo) if hi > lo else d

rois_dir = os.path.join(project_root, 'models', 'subject_avg_category_rois')
beh_dir  = os.path.join(project_root, 'output', 'averaged')
models_dir = os.path.join(project_root, 'models', 'theoretical_models')

# Brain RDM: sub-01 bilateral average, Ventral Stream Visual (representative)
l_path = os.path.join(rois_dir, 'sub-01_L_Ventral_Stream_Visual.tsv')
r_path = os.path.join(rois_dir, 'sub-01_R_Ventral_Stream_Visual.tsv')
l_df   = pd.read_table(l_path, index_col=0).reindex(index=control_conditions, columns=control_conditions)
r_df   = pd.read_table(r_path, index_col=0).reindex(index=control_conditions, columns=control_conditions)
brain_data_raw = ((l_df + r_df) / 2).values.astype(float)
lo, hi = brain_data_raw.min(), brain_data_raw.max()
brain_data = (brain_data_raw - lo) / (hi - lo)
np.fill_diagonal(brain_data, 0)

# Behavioral RDM: group-averaged 80×80 collapsed to 8×8
avg_80 = pd.read_table(
    os.path.join(beh_dir, 'average_similarity_rdm.tsv'), index_col=0)
from src import images, categories
beh_8 = np.zeros((len(categories), len(categories)))
for i, c1 in enumerate(categories):
    for j, c2 in enumerate(categories):
        imgs1 = [img for img in images if c1 in img]
        imgs2 = [img for img in images if c2 in img]
        beh_8[i, j] = avg_80.loc[imgs1, imgs2].values.mean()
beh_lo, beh_hi = beh_8.min(), beh_8.max()
beh_data = (beh_8 - beh_lo) / (beh_hi - beh_lo)

# Model RDM: Category model (8×8)
cat_df = pd.read_table(os.path.join(models_dir, 'category8_model.tsv'), index_col=0)
cat_df.index   = control_conditions
cat_df.columns = control_conditions
cat_data = cat_df.values.astype(float)

rdm_data = [brain_data, beh_data, cat_data]

# ── Draw diagram ──────────────────────────────────────────────────────────────

def create_rsa_diagram():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    left_x  = 1.5
    matrix_x = 6.5
    right_x  = 11.5
    y_positions = [5.5, 3.5, 1.5]

    box_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.5)

    labels_left = [
        "Brain Activity\n(fMRI Patterns)",
        "Behavioral Data\n(Similarity Judgments)",
        "Theoretical Models\n(Computer Vision)"
    ]
    for y, label in zip(y_positions, labels_left):
        ax.text(left_x, y, label, ha="center", va="center", bbox=box_props)
        ax.arrow(left_x + 1.6, y, 1.8, 0,
                 head_width=0.1, head_length=0.1, fc='k', ec='k')

    container = patches.Rectangle((matrix_x - 1.5, 0.2), 3, 6.6,
                                   linewidth=2, edgecolor='gray',
                                   facecolor='none', linestyle='--')
    ax.add_patch(container)
    ax.text(matrix_x, 7.3, "Common Format\n(Abstracted Geometry)",
            ha="center", va="top", weight='bold')

    matrix_labels = ["Brain RDM", "Behavioral RDM", "Model RDM"]

    for i, (y, data) in enumerate(zip(y_positions, rdm_data)):
        im_size = 1.2
        extent  = [matrix_x - im_size/2, matrix_x + im_size/2,
                   y - im_size/2,         y + im_size/2]

        ax.imshow(data, cmap='gray', extent=extent, interpolation='nearest',
                  origin='upper', vmin=0, vmax=1)

        rect = patches.Rectangle((extent[0], extent[2]), im_size, im_size,
                                  linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        ax.text(matrix_x, y - 0.8, matrix_labels[i],
                ha="center", va="top", fontsize=APA_ANNOT, style='italic')

        ax.annotate("",
                    xy=(right_x - 0.8, 3.5), xycoords='data',
                    xytext=(matrix_x + 0.8, y), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # Highlight cell in behavioral RDM
    center_y  = y_positions[1]
    cell_size = 1.2 / 8
    im_size   = 1.2
    mat_left  = matrix_x - im_size / 2
    mat_top   = center_y + im_size / 2
    row, col  = 0, 5
    cell_x    = mat_left + col * cell_size
    cell_y    = mat_top  - (row + 1) * cell_size
    ax.add_patch(patches.Rectangle(
        (cell_x, cell_y), cell_size, cell_size,
        linewidth=2, edgecolor='red', facecolor='none', zorder=5))
    ax.annotate("Dissimilarity $(i, j)$",
                xy=(cell_x + cell_size / 2, cell_y + cell_size / 2),
                xytext=(matrix_x + 2.8, center_y + 1.3),
                color='red', weight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.text(right_x, 3.5, "Quantitative\nComparison\n(Correlation)",
            ha="center", va="center", bbox=box_props)

    plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, 'rsa_basic_explanation.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_rsa_diagram()
