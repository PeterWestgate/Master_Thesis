#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Results Venn diagram: variance partitioning of the behavioural RDM (Y)
by three theoretical models (Category, Animacy, Gist).

Geometry
--------
Circle AREA is proportional to the full Pearson r of each model with Y
(so radius ∝ sqrt(r)).  Radii are computed from the data; see "── Radii"
block below.

Each model circle is positioned so it straddles the Y boundary: the
overlap between the model circle and Y encodes shared variance; the
part of each model circle extending outside Y represents model-specific
variance not captured by the behavioural RDM.

Semi-partial r (sr) labels are placed in each model's exclusive Y∩model
region (inside Y, outside the other two model circles) — verified
analytically.

Data source: output/results/bh_partial_corr_summary.csv

Author: Peter Westgate
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.plot_config import MODEL_COLORS, apply_defaults
from src.figure_style import APA_LABEL, APA_ANNOT
apply_defaults()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
out_fig_dir  = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Results (bh_partial_corr_summary.csv) ─────────────────────────────────────

results = {
    'Category': dict(r=0.59, r_sig='***', sr=0.52, sr_sig='***'),
    'Animacy':  dict(r=0.27, r_sig='***', sr=0.07, sr_sig='*'),
    'Gist':     dict(r=0.10, r_sig='***', sr=0.02, sr_sig='ns'),
}

# ── Colours ───────────────────────────────────────────────────────────────────

col_Y   = '#AED6F1'
col_Y_e = '#5B9BD5'   # Y edge / label colour
col_cat = MODEL_COLORS['Category']   # '#0072B2'
col_ani = MODEL_COLORS['Animacy']    # '#D55E00'
col_gis = MODEL_COLORS.get('Gist', '#009E73')

# ── Radii: area ∝ r  →  radius ∝ sqrt(r) ─────────────────────────────────────
# Scale constant C chosen so Category gets radius R_REF.

R_REF  = 1.82
C_SCALE = R_REF / np.sqrt(results['Category']['r'])   # 2.370

r_cat = C_SCALE * np.sqrt(results['Category']['r'])   # 1.820
r_ani = C_SCALE * np.sqrt(results['Animacy']['r'])    # 1.232
r_gis = C_SCALE * np.sqrt(results['Gist']['r'])       # 0.749

# ── Positions ─────────────────────────────────────────────────────────────────
# Y outer circle.
R_Y  = 2.2
CY_c = np.array([4.5, 4.5])

# Model circle centres — each at distance d from CY_c along the given angle.
# Chosen so that d + r_model > R_Y  (outer edge crosses Y boundary)
#            and d - r_model < R_Y  (inner edge is inside Y — they overlap)
# Verified:
#   Cat  d=1.0  outer 2.82>2.2  inner 3.68<6.70  straddles ✓
#   Ani  d=1.2  outer 2.43>2.2  inner 2.23<6.70  straddles ✓
#   Gist d=1.6  outer 2.35>2.2  inner 2.36<6.70  straddles ✓
# Model–model overlaps (all verified  d < r_i + r_j):
#   Cat–Ani  d=2.13 < 3.05 ✓   Cat–Gist d=2.52 < 2.57 ✓   Ani–Gist d=1.44 < 1.98 ✓

def _pt(centre, d, angle_deg):
    th = np.radians(angle_deg)
    return centre + d * np.array([np.cos(th), np.sin(th)])

CC = _pt(CY_c, 1.0,  90)    # Category  — top        (4.50, 5.50)
CA = _pt(CY_c, 1.2, 240)    # Animacy   — lower-left (3.90, 3.46)
CG = _pt(CY_c, 1.6, 300)    # Gist      — lower-right(5.30, 3.11)

# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
ax.axis('off')
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_aspect('equal', adjustable='datalim')

# ── Y circle (behavioural RDM) ────────────────────────────────────────────────

ax.add_patch(plt.Circle(CY_c, R_Y,
             color=col_Y, alpha=0.18, zorder=1))
ax.add_patch(plt.Circle(CY_c, R_Y,
             fill=False, edgecolor=col_Y_e,
             linewidth=2.2, linestyle='--', zorder=2))

# ── Model circles ─────────────────────────────────────────────────────────────

for ctr, rad, col in [(CC, r_cat, col_cat),
                      (CA, r_ani, col_ani),
                      (CG, r_gis, col_gis)]:
    ax.add_patch(plt.Circle(ctr, rad, color=col, alpha=0.28, zorder=3))
    ax.add_patch(plt.Circle(ctr, rad, fill=False,
                            edgecolor=col, linewidth=2.2, zorder=4))

# ── "shared variance" in Cat–Ani overlap ─────────────────────────────────────
# Position (4.2, 4.5) verified to be inside Cat ∩ Ani ∩ Y and outside Gist.

ax.text(4.2, 4.5, 'shared\nvariance',
        ha='center', va='center',
        fontsize=APA_ANNOT - 1, fontstyle='italic',
        color='#555555', zorder=8)

# ── sr labels in exclusive regions ───────────────────────────────────────────
# Each position verified to be inside Y ∩ model_i and outside the other two.
#   sr_cat (4.5, 6.4)  inside Y (d=1.9<2.2), Cat (d=0.9<1.82), outside Ani+Gist ✓
#   sr_ani (2.8, 3.2)  inside Y (d=2.14<2.2), Ani (d=1.13<1.23), outside Cat+Gist ✓
#   sr_gis (6.0, 3.0)  inside Y (d=2.12<2.2), Gist (d=0.71<0.75), outside Cat+Ani ✓

sr_cfg = [
    ('Category', col_cat, (4.5, 6.4)),
    ('Animacy',  col_ani, (2.8, 3.2)),
    ('Gist',     col_gis, (6.0, 3.0)),
]

for name, col, pos in sr_cfg:
    d = results[name]
    ax.text(*pos,
            f"sr = {d['sr']:.2f}{d['sr_sig']}",
            ha='center', va='center',
            fontsize=APA_ANNOT, fontweight='bold',
            color=col, zorder=9,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white',
                      edgecolor=col,
                      linewidth=1.5, alpha=0.92))

# ── Model name + full r labels (outside Y) ───────────────────────────────────

model_labels = [
    ('Category', col_cat, (4.5,  7.65), (4.5,  7.25)),
    ('Animacy',  col_ani, (2.5,  1.90), (2.5,  1.50)),
    ('Gist',     col_gis, (6.5,  1.90), (6.5,  1.50)),
]

for name, col, name_pos, r_pos in model_labels:
    d = results[name]
    ax.text(*name_pos, name,
            ha='center', va='center',
            fontsize=APA_LABEL, fontweight='bold',
            color=col, zorder=8, clip_on=False)
    ax.text(*r_pos,
            f"r = {d['r']:.2f}{d['r_sig']}",
            ha='center', va='center',
            fontsize=APA_ANNOT, color=col,
            zorder=8, clip_on=False)

# ── Y circle label ────────────────────────────────────────────────────────────

ax.text(CY_c[0] + R_Y + 0.35, CY_c[1],
        'Behavioural\nRDM (Y)',
        ha='left', va='center',
        fontsize=APA_ANNOT, fontweight='bold',
        color=col_Y_e, zorder=8, clip_on=False)

# ── Save (PNG + SVG) ──────────────────────────────────────────────────────────

for ext, kwargs in [('png', dict(dpi=300)), ('svg', {})]:
    save_path = os.path.join(out_fig_dir, f'venn_partial_correlations.{ext}')
    plt.savefig(save_path, bbox_inches='tight', **kwargs)
    print(f"Saved: {save_path}")

plt.show()
