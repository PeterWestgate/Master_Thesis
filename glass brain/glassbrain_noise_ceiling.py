#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Glass brain: neural noise ceiling per ROI.

Each ROI is coloured by its split-half reliability (mean bilateral Pearson r
across 20 subjects) computed on the 8×8 category-averaged RDMs.  Higher
values indicate more consistent neural representations across participants.

Output saved to:
  - glass brain/glassbrain_noise_ceiling.png
  - output/figures/glassbrain_noise_ceiling.png

Intended for the Appendix (supporting information on data quality).

Author: Peter Westgate
'''

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.plot_config import apply_defaults, NOISE_CEIL_COLOR
from src.figure_style import apply_apa_style, fix_figure_fonts, APA_TICK, APA_ANNOT
apply_defaults()

atlas_ref_file  = os.path.join(script_dir, 'HCPMMP1cortices_reference.tsv')
atlas_nii_file  = os.path.join(script_dir, 'sub-01_atlas-HCPMMP1cortices_space-MNI.nii')
reliability_file = os.path.join(project_root, 'output', 'reliability',
                                'roi_avg-category-8x8_fullmatrix_reliabilities.json')
out_fig_dir     = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Name fixes ────────────────────────────────────────────────────────────────

NAME_FIXES = {
    'MT__Complex_and_Neighboring_Visual_Areas': 'MT+_Complex_and_Neighboring_Visual_Areas',
    'Temporo_Parieto_Occipital_Junction':       'Temporo-Parieto-Occipital_Junction',
}

def atlas_name(roi):
    return NAME_FIXES.get(roi, roi)

# ── Load noise ceilings ───────────────────────────────────────────────────────

nc_data = json.load(open(reliability_file))

# Bilateral mean across subjects per ROI
nc_means = {}
for roi, hemi_vals in nc_data.items():
    l = np.array(hemi_vals['L'])
    r = np.array(hemi_vals['R'])
    nc_means[roi] = float(np.mean((l + r) / 2))

# ── Build NIfTI ───────────────────────────────────────────────────────────────

reference_df = pd.read_csv(atlas_ref_file, sep='\t')
atlas_img    = nib.load(atlas_nii_file)
atlas_data   = atlas_img.get_fdata()
combined     = np.zeros_like(atlas_data)

rois_missing = []
for roi, mean_nc in nc_means.items():
    aname = atlas_name(roi)
    labels = []
    for hemi in ['L_', 'R_']:
        match = reference_df[reference_df['name'] == hemi + aname]
        if not match.empty:
            labels.append(match.iloc[0]['label'])
    if labels:
        for lbl in labels:
            combined[atlas_data == lbl] = mean_nc
    else:
        rois_missing.append(roi)

if rois_missing:
    print(f"Warning: {len(rois_missing)} ROI(s) not mapped: {rois_missing}")

combined_img = nib.Nifti1Image(combined, atlas_img.affine, atlas_img.header)

# ── Colormap: white → NOISE_CEIL_COLOR (orange) ──────────────────────────────

cmap = LinearSegmentedColormap.from_list(
    'noise_ceil', ['#ffffff', NOISE_CEIL_COLOR], N=256
)

all_vals = np.array(list(nc_means.values()))
vmin = max(0.0, all_vals.min() - 0.02)
vmax = all_vals.max() + 0.02

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

fig.axes[-1].set_ylabel('Split-half reliability (Pearson r)', fontsize=APA_TICK, labelpad=6)

# Annotate best and worst ROIs
def _roi_label(roi):
    if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
        return 'MT+ Complex & Neighboring Visual Areas'
    return roi.replace('_', ' ')
best_roi  = _roi_label(max(nc_means, key=nc_means.get))
worst_roi = _roi_label(min(nc_means, key=nc_means.get))
fig.text(0.5, -0.04,
         f'Highest: {best_roi} (r = {max(nc_means.values()):.2f})   '
         f'Lowest: {worst_roi} (r = {min(nc_means.values()):.2f})',
         ha='center', fontsize=APA_ANNOT, color='#333333', style='italic')


fix_figure_fonts(fig)
plt.tight_layout()

for save_path in [
    os.path.join(script_dir,  'glassbrain_noise_ceiling.png'),
    os.path.join(out_fig_dir, 'glassbrain_noise_ceiling.png'),
]:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

plt.show()
