#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Glass brain: brain-behaviour correlation strength per ROI.

Each ROI is coloured by its full Pearson r with the average behavioural RDM
(from rsa_brain_beh_partial_corr.py).  Warmer / darker colour = stronger
positive correlation.  ROIs with p < .05 are marked with an asterisk in the
colour-bar annotation.

Output saved to:
  - glass brain/glassbrain_brain_behaviour.png
  - output/figures/glassbrain_brain_behaviour.png

Author: Peter Westgate
'''

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.plot_config import apply_defaults, BRAIN_BEH_COLOR
from src.figure_style import apply_apa_style, fix_figure_fonts, APA_TICK, APA_ANNOT
apply_defaults()

atlas_ref_file = os.path.join(script_dir, 'HCPMMP1cortices_reference.tsv')
atlas_nii_file = os.path.join(script_dir, 'sub-01_atlas-HCPMMP1cortices_space-MNI.nii')
results_file   = os.path.join(project_root, 'output', 'results', 'brain_beh_partial_corr_summary.csv')
out_fig_dir    = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Name fixes: src/__init__.py → atlas ───────────────────────────────────────

NAME_FIXES = {
    'MT__Complex_and_Neighboring_Visual_Areas': 'MT+_Complex_and_Neighboring_Visual_Areas',
    'Temporo_Parieto_Occipital_Junction':       'Temporo-Parieto-Occipital_Junction',
}

def atlas_name(roi):
    return NAME_FIXES.get(roi, roi)

# ── Load brain-behaviour correlations ─────────────────────────────────────────

df = pd.read_csv(results_file)
full_r = df[df['corr_type'] == 'full'].set_index('roi')[['mean_r', 'p']]

# ── Build NIfTI with r-values per ROI ─────────────────────────────────────────

reference_df = pd.read_csv(atlas_ref_file, sep='\t')
atlas_img    = nib.load(atlas_nii_file)
atlas_data   = atlas_img.get_fdata()
combined     = np.zeros_like(atlas_data)

# Use NaN sentinel (0 = background, actual r values fill ROI voxels)
# nilearn treats 0 as transparent so we shift: store r + offset to keep 0 = empty
# Simpler: fill with small sentinel (NaN not supported in NIfTI float32 well)
# We keep 0 as "no ROI" and fill voxels with actual r value directly.

rois_missing = []
for roi, row in full_r.iterrows():
    aname = atlas_name(roi)
    labels = []
    for hemi in ['L_', 'R_']:
        match = reference_df[reference_df['name'] == hemi + aname]
        if not match.empty:
            labels.append(match.iloc[0]['label'])
    if labels:
        for lbl in labels:
            combined[atlas_data == lbl] = row['mean_r']
    else:
        rois_missing.append(roi)

if rois_missing:
    print(f"Warning: {len(rois_missing)} ROI(s) not mapped: {rois_missing}")

combined_img = nib.Nifti1Image(combined, atlas_img.affine, atlas_img.header)

# ── Colormap: white → BRAIN_BEH_COLOR ────────────────────────────────────────

cmap = LinearSegmentedColormap.from_list(
    'brain_beh', ['#ffffff', BRAIN_BEH_COLOR], N=256
)

r_vals  = full_r['mean_r'].values
vmin    = 0.0                     # floor at 0 (negative r would show as white)
vmax    = r_vals.max() * 1.05

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 5))

display = plotting.plot_glass_brain(
    combined_img,
    display_mode='lyrz',
    figure=fig,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    colorbar=True,
    plot_abs=False,
    alpha=0.85,
)

# Annotate colorbar
fig.axes[-1].set_ylabel('Pearson r  (brain–behaviour)', fontsize=APA_TICK, labelpad=6)

# Annotation: list only non-significant ROIs (n.s.) — significant ones are now the majority
ns_rois = full_r[full_r['p'] >= .05].index.tolist()
if ns_rois:
    def _roi_label(roi):
        if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
            return 'MT+ Complex & Neighboring Visual Areas'
        return roi.replace('_', ' ')
    ns_label = ', '.join([_roi_label(r) for r in ns_rois])
    note = f'n.s. (p ≥ .05): {ns_label}'
else:
    note = 'All ROIs significant at p < .05'
fig.text(0.5, -0.04, note,
         ha='center', fontsize=APA_ANNOT, color='#333333', style='italic')


fix_figure_fonts(fig)
plt.tight_layout()

for save_path in [
    os.path.join(script_dir,  'glassbrain_brain_behaviour.png'),
    os.path.join(out_fig_dir, 'glassbrain_brain_behaviour.png'),
]:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

plt.show()
