import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src import category_palette
from src.plot_config import apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT

apply_defaults()

# ── Paths ──────────────────────────────────────────────────────────────────────

project_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_dir     = os.path.join(project_root, 'models', 'theoretical_models')
out_fig_dir    = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Load real model data ───────────────────────────────────────────────────────

def load_model(fname):
    sep = '\t' if fname.endswith('.tsv') else ','
    return pd.read_csv(os.path.join(models_dir, fname), sep=sep, index_col=0).values.astype(float)

animacy_40  = load_model('animacy40_model.tsv')
category_40 = load_model('category40_model.tsv')
gist_40     = load_model('gist40_model.tsv')

animacy_80  = load_model('animacy80_model.tsv')
category_80 = load_model('category80_model.tsv')
gist_80     = load_model('gist80_model.tsv')

# ── Categories ─────────────────────────────────────────────────────────────────

N_CATS     = 8
CAT_KEYS   = ['person', 'cat', 'bird', 'banana', 'firehydrant', 'tree', 'bus', 'building']
CAT_LABELS = ['Person', 'Cat', 'Bird', 'Banana', 'F. Hydrant', 'Tree', 'Bus', 'Building']
CAT_COLORS = [category_palette[k] for k in CAT_KEYS]

# ── Category colour strips ─────────────────────────────────────────────────────

def add_category_strips(ax, n_per_cat):
    n   = N_CATS * n_per_cat
    rgb = np.array([mcolors.to_rgb(CAT_COLORS[ci])
                    for ci in range(N_CATS) for _ in range(n_per_cat)])

    ax_bottom = ax.inset_axes([0, -0.07, 1, 0.05])
    ax_bottom.imshow(rgb.reshape(1, n, 3), aspect='auto', interpolation='nearest')
    ax_bottom.set_axis_off()

    ax_left = ax.inset_axes([-0.05, 0, 0.03, 1])
    ax_left.imshow(rgb.reshape(n, 1, 3), aspect='auto', interpolation='nearest')
    ax_left.set_axis_off()

    centers = [i * n_per_cat + (n_per_cat - 1) / 2 for i in range(N_CATS)]
    ax.set_xticks(centers)
    ax.set_xticklabels(CAT_LABELS, rotation=45, ha='right', fontsize=APA_ANNOT)
    ax.tick_params(axis='x', length=0, pad=22)
    ax.set_yticks([])

# ── Plot ───────────────────────────────────────────────────────────────────────

MODEL_NAMES = ['Animacy', 'Category', 'Gist']
ROW_LABELS  = ['Conditions', 'Exemplars']
SIZES       = [40, 80]
N_PER_CAT   = [5, 10]

row_data = [
    [animacy_40, category_40, gist_40],
    [animacy_80, category_80, gist_80],
]

fig, axes = plt.subplots(2, 3, figsize=(15, 11))
fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.45)

for row, (models, size, npc) in enumerate(zip(row_data, SIZES, N_PER_CAT)):
    for col, (data, name) in enumerate(zip(models, MODEL_NAMES)):
        ax = axes[row, col]
        im = ax.imshow(data, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'{name} ({size}×{size})', fontsize=APA_LABEL, pad=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1)
        add_category_strips(ax, npc)

for row, label in enumerate(ROW_LABELS):
    axes[row, 0].set_ylabel(label, fontsize=APA_LABEL, labelpad=55)

# Colorbar in dedicated space — will not overlap any subplot
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Dissimilarity', fontsize=APA_TICK)
cbar.ax.tick_params(labelsize=APA_TICK)

plt.savefig(os.path.join(out_fig_dir, 'theoretical_models_rdms.png'), dpi=300, bbox_inches='tight')
plt.show()
