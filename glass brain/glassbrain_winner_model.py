#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Glass brain: dominant theoretical model per ROI.

Each ROI is coloured by whichever theoretical model has the highest full
Pearson r (from rsa_neural_partial_corr.py).
Only models reaching p < .05 are eligible; ties go to the highest r.

  Category → blue    (#0072B2)
  Animacy  → amber   (#E69F00)
  Gist     → green   (#009E73)
  No significant model → grey (#cccccc)

Label encoding (voxel values):
  1 = Category,  2 = Animacy,  3 = Gist,  4 = no sig model
  0 = background (transparent — must NOT be assigned to any ROI)

Output saved to:
  - glass brain/glassbrain_winner_model.png
  - output/figures/glassbrain_winner_model.png

Author: Peter Westgate
'''

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from nilearn import plotting

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.plot_config import apply_defaults, MODEL_COLORS
from src.figure_style import apply_apa_style, fix_figure_fonts, APA_TICK
apply_defaults()

atlas_ref_file = os.path.join(script_dir, 'HCPMMP1cortices_reference.tsv')
atlas_nii_file = os.path.join(script_dir, 'sub-01_atlas-HCPMMP1cortices_space-MNI.nii')
results_file   = os.path.join(project_root, 'output', 'results', 'neural_partial_corr_summary.csv')
out_fig_dir    = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Name fixes ────────────────────────────────────────────────────────────────

NAME_FIXES = {
    'MT__Complex_and_Neighboring_Visual_Areas': 'MT+_Complex_and_Neighboring_Visual_Areas',
    'Temporo_Parieto_Occipital_Junction':       'Temporo-Parieto-Occipital_Junction',
}

def atlas_name(roi):
    return NAME_FIXES.get(roi, roi)

# ── Colour overrides ──────────────────────────────────────────────────────────
# Animacy uses amber here (Wong 2011) instead of vermillion used elsewhere,
# to maximise contrast against blue (Category) and green (Gist).

WINNER_COLORS = {
    'Category': MODEL_COLORS['Category'],   # blue    #0072B2
    'Animacy':  '#E69F00',                  # amber   #E69F00
    'Gist':     MODEL_COLORS['Gist'],       # green   #009E73
}

# ── Model → integer label (1-3; 4 = no sig model; 0 = background only) ───────

MODEL_LABEL = {'Category': 1, 'Animacy': 2, 'Gist': 3}
NO_SIG_LABEL = 4

# ── Load partial correlations & pick winner per ROI ───────────────────────────

df   = pd.read_csv(results_file)
full = df[df['corr_type'] == 'full']

winner_map = {}
for roi in full['roi'].unique():
    roi_rows = full[full['roi'] == roi]
    sig_rows = roi_rows[roi_rows['p'] < .05]
    if sig_rows.empty:
        winner_map[roi] = NO_SIG_LABEL
    else:
        best_model = sig_rows.loc[sig_rows['mean_r'].idxmax(), 'model']
        winner_map[roi] = MODEL_LABEL[best_model]

# ── Build NIfTI ───────────────────────────────────────────────────────────────

reference_df = pd.read_csv(atlas_ref_file, sep='\t')
atlas_img    = nib.load(atlas_nii_file)
atlas_data   = atlas_img.get_fdata()
combined     = np.zeros_like(atlas_data)   # 0 = background (transparent)

rois_missing = []
for roi, label in winner_map.items():
    aname  = atlas_name(roi)
    labels = []
    for hemi in ['L_', 'R_']:
        match = reference_df[reference_df['name'] == hemi + aname]
        if not match.empty:
            labels.append(match.iloc[0]['label'])
    if labels:
        for lbl in labels:
            combined[atlas_data == lbl] = label
    else:
        rois_missing.append(roi)

if rois_missing:
    print(f"Warning: {len(rois_missing)} ROI(s) not mapped: {rois_missing}")

combined_img = nib.Nifti1Image(combined, atlas_img.affine, atlas_img.header)

# ── Discrete colormap: values 1→4 map to 4 colours ───────────────────────────
# vmin=0.5, vmax=4.5 divides the range into 4 equal bins:
#   [0.5–1.5) → Category (blue)
#   [1.5–2.5) → Animacy  (amber)
#   [2.5–3.5) → Gist     (green)
#   [3.5–4.5) → no sig   (grey)

cmap = mcolors.ListedColormap([
    WINNER_COLORS['Category'],   # label 1
    WINNER_COLORS['Animacy'],    # label 2
    WINNER_COLORS['Gist'],       # label 3
    '#cccccc',                   # label 4 = no significant model
])

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5))

plotting.plot_glass_brain(
    combined_img,
    display_mode='lyrz',
    figure=fig,
    cmap=cmap,
    vmin=0.5,
    vmax=4.5,
    colorbar=False,
    plot_abs=False,
    alpha=0.85,
)

# Legend
legend_elements = [
    Patch(facecolor=WINNER_COLORS['Category'], edgecolor='#555555',
          linewidth=0.5, label='Category model'),
    Patch(facecolor=WINNER_COLORS['Animacy'],  edgecolor='#555555',
          linewidth=0.5, label='Animacy model'),
    Patch(facecolor=WINNER_COLORS['Gist'],     edgecolor='#555555',
          linewidth=0.5, label='Gist model'),
    Patch(facecolor='#cccccc',                 edgecolor='#555555',
          linewidth=0.5, label='No significant model (p \u2265 .05)'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=4, fontsize=APA_TICK, bbox_to_anchor=(0.5, -0.05),
           frameon=False)


fix_figure_fonts(fig)
plt.tight_layout()

for save_path in [
    os.path.join(script_dir,  'glassbrain_winner_model.png'),
    os.path.join(out_fig_dir, 'glassbrain_winner_model.png'),
]:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

plt.show()
