#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Glass brain visualisation: 9 example ROIs from the 22 used in the thesis.

Each ROI has its own colour. ROIs were selected to span the full brain
from occipital to prefrontal cortex.

Output saved to:
  - glass brain/ROI_glass_brain_fixed.png
  - output/figures/ROI_glass_brain.png

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

from src.plot_config import apply_defaults
from src.figure_style import apply_apa_style, fix_figure_fonts, APA_TICK
apply_defaults()

atlas_ref_file = os.path.join(script_dir, 'HCPMMP1cortices_reference.tsv')
atlas_nii_file = os.path.join(script_dir, 'sub-01_atlas-HCPMMP1cortices_space-MNI.nii')
out_fig_dir    = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── 9 example ROIs: (atlas_key, display_label, colour) ───────────────────────
# Covers occipital → temporal → parietal → central → frontal.
# Colours from Wong (2011) + Tol Muted palette — consistent with thesis figures.

ROI_CONFIG = [
    ('Primary_Visual',
     'Primary Visual',
     '#56B4E9'),   # Wong sky blue
    ('Early_Visual',
     'Early Visual',
     '#882255'),   # Tol wine
    ('Ventral_Stream_Visual',
     'Ventral Stream Visual',
     '#0072B2'),   # Wong blue
    ('MT__Complex_and_Neighboring_Visual_Areas',
     'MT+ Complex & Neighboring Visual Areas',
     '#009E73'),   # Wong bluish green
    ('Medial_Temporal',
     'Medial Temporal',
     '#CC79A7'),   # Wong reddish purple
    ('Lateral_Temporal',
     'Lateral Temporal',
     '#E69F00'),   # Wong amber
    ('Superior_Parietal',
     'Superior Parietal',
     '#332288'),   # Tol indigo
    ('Posterior_Cingulate',
     'Posterior Cingulate',
     '#117733'),   # Tol forest green
    ('Somatosensory_and_Motor',
     'Somatosensory & Motor',
     '#D55E00'),   # Wong vermillion
    ('Auditory_Association',
     'Auditory Association',
     '#44AA99'),   # Tol teal
    ('Inferior_Frontal',
     'Inferior Frontal',
     '#AA3377'),   # Tol dark pink
    ('Dorsolateral_Prefrontal',
     'Dorsolateral Prefrontal',
     '#DDAA33'),   # Tol golden
]

# ── Name fixes: src/__init__.py key → atlas file name ─────────────────────────

NAME_FIXES = {
    'MT__Complex_and_Neighboring_Visual_Areas': 'MT+_Complex_and_Neighboring_Visual_Areas',
    'Temporo_Parieto_Occipital_Junction':       'Temporo-Parieto-Occipital_Junction',
}

def atlas_name(roi):
    return NAME_FIXES.get(roi, roi)

# ── Build NIfTI volume and render ─────────────────────────────────────────────

def create_glassbrain():
    for path, label in [(atlas_ref_file, 'reference TSV'), (atlas_nii_file, 'atlas NIfTI')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            return

    reference_df = pd.read_csv(atlas_ref_file, sep='\t')
    atlas_img    = nib.load(atlas_nii_file)
    atlas_data   = atlas_img.get_fdata()
    combined     = np.zeros_like(atlas_data)

    print("Mapping ROIs to atlas...")
    for roi_idx, (roi, display_name, _) in enumerate(ROI_CONFIG, start=1):
        aname  = atlas_name(roi)
        labels = []
        for hemi in ['L_', 'R_']:
            match = reference_df[reference_df['name'] == hemi + aname]
            if not match.empty:
                labels.append(match.iloc[0]['label'])
        if labels:
            for lbl in labels:
                combined[atlas_data == lbl] = roi_idx
            print(f"  [OK] {display_name}")
        else:
            print(f"  [MISS] {display_name}  (looked for '{aname}')")

    print("Generating glass brain...")
    fig = plt.figure(figsize=(14, 5))

    display = plotting.plot_glass_brain(
        None,
        display_mode='lyrz',
        figure=fig,
        colorbar=False,
        plot_abs=False,
    )

    # One binary mask per ROI — individual solid colour
    for roi_idx, (roi, display_name, color) in enumerate(ROI_CONFIG, start=1):
        mask = (combined == roi_idx).astype(np.float32)
        if not mask.any():
            continue
        mask_img   = nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header)
        solid_cmap = mcolors.LinearSegmentedColormap.from_list(roi, [color, color])
        display.add_overlay(mask_img, cmap=solid_cmap,
                            vmin=0.5, vmax=1.5, threshold=0.5, alpha=0.85)

    legend_elements = [
        Patch(facecolor=color, edgecolor='#555555', linewidth=0.5, label=display_name)
        for _, display_name, color in ROI_CONFIG
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=APA_TICK, bbox_to_anchor=(0.5, -0.18),
               frameon=False)

    fix_figure_fonts(fig)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)

    for save_path in [
        os.path.join(script_dir, 'ROI_glass_brain_fixed.png'),
        os.path.join(out_fig_dir, 'ROI_glass_brain.png'),
    ]:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == '__main__':
    create_glassbrain()
