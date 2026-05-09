#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Behavioural RSA: full and semi-partial correlations
against theoretical models (Category, Animacy, GIST).

Computed per participant, then Fisher-z averaged across participants.
Group-level significance: one-sample t-test on Fisher-z values vs. 0.

Full r:        corr(Y, X_model)
Semi-partial:  corr(Y, resid(X_model | other models))
               → unique variance X_model explains in Y

Author: Peter Westgate
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

from src import images
from src.utils import upper_triangle
from src.plot_config import MODEL_COLORS, CORR_COLORS, BAR_DEFAULTS, NOISE_CEIL_COLOR, apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_STAR
apply_defaults()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _residuals(y, covariates):
    """Residuals of y after regressing out covariates (with intercept)."""
    X = np.column_stack([np.ones(len(y))] + list(covariates))
    coef, _, _, _ = lstsq(X, y, rcond=None)
    return y - X @ coef


def semipartial_corr(y, x1, covariates):
    """Semi-partial r: corr(y, resid(x1 | covariates))."""
    return pearsonr(y, _residuals(x1, covariates))


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def group_stats(rs):
    """Fisher-z averaged mean r, SD (in r space), and one-sample t-test p."""
    zs = np.array([fisher_z(r) for r in rs])
    mean_r = np.tanh(np.mean(zs))
    sd_r   = np.std(rs, ddof=1)
    _, p   = ttest_1samp(zs, 0)
    return mean_r, sd_r, p


def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'ns'

# ── Directories ───────────────────────────────────────────────────────────────

project_root           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
theoretical_models_dir = os.path.join(project_root, 'models', 'theoretical_models')
out_rdms_dir           = os.path.join(project_root, 'output', 'rdms')
out_fig_dir            = os.path.join(project_root, 'output', 'figures')
out_results_dir        = os.path.join(project_root, 'output', 'results')
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(out_results_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

rdm_files = sorted(glob.glob(os.path.join(out_rdms_dir, '*.tsv')))
if not rdm_files:
    sys.exit(f"No behavioural RDMs found in {out_rdms_dir}")

similarity_rdms = {
    os.path.basename(f).split('_')[0]: pd.read_table(f, sep='\t', index_col=0)
                                          .reindex(index=images, columns=images)
    for f in rdm_files
}
subjects    = sorted(similarity_rdms.keys())
print(f"Loaded {len(subjects)} behavioural RDMs: {subjects}")

model_files = sorted(glob.glob(os.path.join(theoretical_models_dir, '*80*.tsv')))
if not model_files:
    sys.exit(f"No 80×80 model files found in {theoretical_models_dir}")

theoretical_models = {
    os.path.basename(f).split('_')[0]: pd.read_table(f, sep='\t', index_col=0)
                                          .reindex(index=images, columns=images)
    for f in model_files
}
model_names = sorted(theoretical_models.keys())
print(f"Loaded {len(model_names)} theoretical models: {model_names}")

# ── Per-participant correlations ──────────────────────────────────────────────

corr_types = ['full', 'semipartial']
results    = {m: {ct: [] for ct in corr_types} for m in model_names}

for sub in subjects:
    y     = upper_triangle(similarity_rdms[sub].values)
    mvecs = {m: upper_triangle(theoretical_models[m].values) for m in model_names}

    for m in model_names:
        x1  = mvecs[m]
        cov = [mvecs[other] for other in model_names if other != m]

        results[m]['full'].append(pearsonr(y, x1)[0])
        results[m]['semipartial'].append(semipartial_corr(y, x1, cov)[0])

# ── Group-level summary ───────────────────────────────────────────────────────

print(f"\nBehaviour – Theoretical Model Correlations  (N = {len(subjects)})")
print("=" * 70)
print(f"{'Model':<14} {'Type':<14} {'Mean r':>8} {'SD':>8} {'p':>10}  sig")
print("-" * 70)

summary  = {}
csv_rows = []
for m in model_names:
    label    = m.replace('80', '').capitalize()
    summary[m] = {}
    for ct in corr_types:
        mean_r, sd_r, p = group_stats(results[m][ct])
        summary[m][ct]  = (mean_r, sd_r, p)
        print(f"{label:<14} {ct:<14} {mean_r:>8.3f} {sd_r:>8.3f} {p:>10.4f}  {sig_stars(p)}")
        csv_rows.append({'model': label, 'corr_type': ct,
                         'mean_r': round(mean_r, 4), 'sd_r': round(sd_r, 4),
                         'p': round(p, 4), 'sig': sig_stars(p)})

print("=" * 70)

pd.DataFrame(csv_rows).to_csv(
    os.path.join(out_results_dir, 'bh_partial_corr_summary.csv'), index=False)

sub_rows = []
for sub in subjects:
    for m in model_names:
        label = m.replace('80', '').capitalize()
        for ct in corr_types:
            sub_rows.append({'subject': sub, 'model': label, 'corr_type': ct,
                             'r': results[m][ct][subjects.index(sub)]})
pd.DataFrame(sub_rows).to_csv(
    os.path.join(out_results_dir, 'bh_partial_corr_per_subject.csv'), index=False)

print("Results saved to output/results/")

# ── Plot: Venn diagram ────────────────────────────────────────────────────────
# Behavioural RSA Venn: three model circles arranged inside the behavioural RDM
# circle, overlapping each other to represent shared variance between models.
# Full r annotated outside the behavioural circle; sr annotated in each model's
# unique region (inside behavioural, outside the other two models).

import matplotlib.patches as mpatches

fig_v, ax_v = plt.subplots(figsize=(9, 8))
ax_v.axis('off')
ax_v.set_xlim(0, 10)
ax_v.set_ylim(0, 9.5)
ax_v.set_aspect('equal')

# ── Geometry ──────────────────────────────────────────────────────────────────
cx, cy  = 5.0, 4.6    # behavioural circle centre
r_beh   = 3.0          # behavioural circle radius (large)
d_mod   = 1.1          # offset of each model centre from cx, cy
r_mod   = 1.9          # model circle radius
# → adjacent model centres are ~1.9 apart; with r_mod=1.9 they overlap heavily
# → model circles extend to d_mod+r_mod = 3.0 from cx,cy = just at beh boundary

# model_names sorted alphabetically: animacy80, category80, gist80
model_angles = {
    model_names[0]: 210,   # animacy  → lower-left
    model_names[1]: 90,    # category → top
    model_names[2]: 330,   # gist     → lower-right
}

# ── Behavioural RDM circle ────────────────────────────────────────────────────
ax_v.add_patch(plt.Circle((cx, cy), r_beh,
               color='#AED6F1', alpha=0.14, zorder=1))
ax_v.add_patch(plt.Circle((cx, cy), r_beh,
               fill=False, edgecolor='#2E86AB', lw=2.5, zorder=2))

# Label at bottom of behavioural circle (270° direction is free of model circles)
ax_v.text(cx, cy - 2.15, f'Behavioral\nRDM\n(N\u200a=\u200a{len(subjects)})',
          ha='center', va='center', fontsize=APA_LABEL, fontweight='bold',
          color='#1A5276', zorder=8)

# ── Model circles ─────────────────────────────────────────────────────────────
for m in model_names:
    disp   = m.replace('80', '').capitalize()
    mcolor = MODEL_COLORS.get(disp, '#888888')
    angle  = np.radians(model_angles[m])
    mx     = cx + d_mod * np.cos(angle)
    my     = cy + d_mod * np.sin(angle)

    full_r, full_sd, full_p = summary[m]['full']
    semi_r, semi_sd, semi_p = summary[m]['semipartial']
    stars_f = sig_stars(full_p)
    stars_s = sig_stars(semi_p)

    ax_v.add_patch(plt.Circle((mx, my), r_mod,
                   color=mcolor, alpha=0.22, zorder=3))
    ax_v.add_patch(plt.Circle((mx, my), r_mod,
                   fill=False, edgecolor=mcolor, lw=2, zorder=4))

    # Model name + full r: outside behavioural circle along the same radial direction
    name_x = cx + (d_mod + r_mod + 0.55) * np.cos(angle)
    name_y = cy + (d_mod + r_mod + 0.55) * np.sin(angle)
    ax_v.text(name_x, name_y, disp,
              ha='center', va='center',
              fontsize=APA_LABEL, fontweight='bold', color=mcolor, zorder=9)
    ax_v.text(name_x, name_y - 0.48,
              f'r = {full_r:.2f}{stars_f}',
              ha='center', va='center',
              fontsize=APA_TICK, color=mcolor, zorder=9)

    # Semi-partial r: in the unique region of this model
    # (inside beh circle, inside this model circle, outside the other two)
    # Positioned further along the radial direction than the shared centre
    ann_x = cx + (d_mod + 0.72) * np.cos(angle)
    ann_y = cy + (d_mod + 0.72) * np.sin(angle)
    ax_v.text(ann_x, ann_y,
              f'sr = {semi_r:.2f}{stars_s}',
              ha='center', va='center', fontsize=9, fontweight='bold',
              color=mcolor, zorder=10,
              bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                        alpha=0.92, edgecolor=mcolor, lw=1.0))

# ── Shared-variance label at the triple-overlap centre ────────────────────────
ax_v.text(cx, cy + 0.22,
          'shared\nvariance',
          ha='center', va='center', fontsize=8, color='#555555',
          fontstyle='italic', zorder=11)

# ── Annotation key ────────────────────────────────────────────────────────────
ax_v.text(0.02, 0.02,
          'r = Full Pearson r (total overlap with Behavioral RDM)\n'
          'sr = Semi-partial r (unique overlap, other models partialled out)\n'
          '* p\u200a<\u200a.05,  ** p\u200a<\u200a.01,  *** p\u200a<\u200a.001',
          transform=ax_v.transAxes, fontsize=APA_TICK, color='#555555',
          va='bottom', ha='left',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9',
                    alpha=0.85, edgecolor='#cccccc', lw=0.8))


save_path = os.path.join(out_fig_dir, 'behaviour_partial_correlations.png')
fig_v.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Venn figure saved to {save_path}")
plt.show()
