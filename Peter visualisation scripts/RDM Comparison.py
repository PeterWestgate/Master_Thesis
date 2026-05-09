import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from src.plot_config import apply_defaults
from src.figure_style import APA_LABEL, APA_TICK

apply_defaults()

project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
models_dir      = os.path.join(project_root, 'models', 'theoretical_models')
out_fig_dir     = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

n_cats = 8

# ── Analytically-defined models ───────────────────────────────────────────────

def make_category(n_cats, n_per):
    n = n_cats * n_per
    rdm = np.ones((n, n))
    for i in range(n_cats):
        s, e = i * n_per, (i + 1) * n_per
        rdm[s:e, s:e] = 0
    return rdm

def make_animacy(n_cats, n_per):
    n = n_cats * n_per
    rdm = np.ones((n, n))
    animates = [0, 1, 2]
    status = []
    for i in range(n_cats):
        status.extend([0 if i in animates else 1] * n_per)
    for i in range(n):
        for j in range(n):
            if status[i] == status[j]:
                rdm[i, j] = 0
    return rdm

# ── Load real Gist models from file ───────────────────────────────────────────

def load_gist(fname):
    path = os.path.join(models_dir, fname)
    df = pd.read_csv(path, sep='\t', index_col=0)
    data = df.values.astype(float)
    # Normalise to [0, 1] for display
    lo, hi = np.nanmin(data), np.nanmax(data)
    if hi > lo:
        data = (data - lo) / (hi - lo)
    np.fill_diagonal(data, 0)
    return data

rdms_40 = [make_animacy(n_cats, 5),  make_category(n_cats, 5),  load_gist('gist40_model.tsv')]
rdms_80 = [make_animacy(n_cats, 10), make_category(n_cats, 10), load_gist('gist80_model.tsv')]
titles  = ['Animacy', 'Category', 'Gist']

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])

# Row 1: Condition-Level (40×40)
for i, rdm in enumerate(rdms_40):
    ax = plt.subplot(gs[0, i])
    ax.imshow(rdm, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{titles[i]} (40×40)", fontsize=APA_LABEL, fontweight='bold')
    ax.axis('off')
    if i == 0:
        ax.text(-10, 20, "Conditions", rotation=90, va='center', fontsize=APA_LABEL)

# Row 2: Exemplar-Level (80×80)
for i, rdm in enumerate(rdms_80):
    ax = plt.subplot(gs[1, i])
    im = ax.imshow(rdm, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{titles[i]} (80×80)", fontsize=APA_LABEL, fontweight='bold')
    ax.axis('off')
    if i == 0:
        ax.text(-10, 40, "Exemplars", rotation=90, va='center', fontsize=APA_LABEL)

# Colorbar
cax = plt.subplot(gs[:, 3])
plt.colorbar(im, cax=cax, label='Dissimilarity')

plt.tight_layout()
plt.savefig(os.path.join(out_fig_dir, 'rdm_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
