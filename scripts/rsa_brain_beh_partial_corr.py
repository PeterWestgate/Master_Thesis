#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Brain-behaviour RSA: full and semi-partial correlations per ROI.

For each participant × ROI:
  - Full r:         corr(neural_RDM, avg_behavioral_RDM)
  - Semi-partial r: corr(neural_RDM, resid(avg_behavioral_RDM | Category, Animacy, GIST))
                    → unique variance behaviour explains in neural, above and beyond
                      what the theoretical models already explain

Bilateral ROI: L and R hemispheres averaged per subject.
Behavioural RDM: group average (80×80) collapsed to 8×8 category means.
Neural RDMs: per-subject 8×8 category-averaged, from subject_avg_category_rois/.

Group-level inference: one-sample t-test on Fisher-z values vs. 0.
Bar plot: full r and semi-partial r side by side per ROI.

Author: Peter Westgate
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

from src import images, rois, categories, control_conditions
from src.utils import upper_triangle
from src.plot_config import BRAIN_BEH_COLOR, CORR_COLORS, BAR_DEFAULTS, NOISE_CEIL_COLOR, apply_defaults
from src.figure_style import APA_TICK, APA_TICK_ROI, APA_STAR
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
    rs  = [r for r in rs if not np.isnan(r)]
    zs  = np.array([fisher_z(r) for r in rs])
    return np.tanh(np.mean(zs)), np.std(rs, ddof=1), ttest_1samp(zs, 0)[1]


def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'ns'

# ── Directories ───────────────────────────────────────────────────────────────

project_root           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
theoretical_models_dir = os.path.join(project_root, 'models', 'theoretical_models')
avg_category_brain_dir = os.path.join(project_root, 'models', 'subject_avg_category_rois')
out_avg_dir            = os.path.join(project_root, 'output', 'averaged')
out_fig_dir            = os.path.join(project_root, 'output', 'figures')
out_results_dir        = os.path.join(project_root, 'output', 'results')
reliability_dir        = os.path.join(project_root, 'output', 'reliability')
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(out_results_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

avg_sim = pd.read_table(
    os.path.join(out_avg_dir, 'average_similarity_rdm.tsv'), index_col=0
).reindex(index=images, columns=images)

beh_8x8 = np.zeros((len(categories), len(categories)))
for i, c1 in enumerate(categories):
    for j, c2 in enumerate(categories):
        imgs_i = [img for img in images if c1 in img]
        imgs_j = [img for img in images if c2 in img]
        beh_8x8[i, j] = avg_sim.loc[imgs_i, imgs_j].values.mean()

beh_df  = pd.DataFrame(beh_8x8, index=control_conditions, columns=control_conditions)
vec_beh = beh_df.values.flatten()

def load_model_8x8(fname):
    ext  = os.path.splitext(fname)[1]
    sep  = '\t' if ext == '.tsv' else ','
    path = os.path.join(theoretical_models_dir, fname)
    df   = pd.read_csv(path, sep=sep, index_col=0)
    df.index   = control_conditions
    df.columns = control_conditions
    return df

models_8x8 = {
    'category': load_model_8x8('category8_model.tsv'),
    'animacy':  load_model_8x8('animacy8_model.tsv'),
    'gist':     load_model_8x8('gist8_model.csv'),
}
model_covs = [m.values.flatten() for m in models_8x8.values()]

subjects = [f'sub-{str(n + 1).zfill(2)}' for n in range(20)]
roi_rdms = {roi: [] for roi in rois}

for s in subjects:
    for roi in rois:
        l_file = os.path.join(avg_category_brain_dir, f'{s}_L_{roi}.tsv')
        r_file = os.path.join(avg_category_brain_dir, f'{s}_R_{roi}.tsv')
        try:
            l = pd.read_table(l_file, sep='\t', index_col=0).reindex(
                    index=control_conditions, columns=control_conditions)
            r = pd.read_table(r_file, sep='\t', index_col=0).reindex(
                    index=control_conditions, columns=control_conditions)
            roi_rdms[roi].append((l + r) / 2)
        except FileNotFoundError:
            roi_rdms[roi].append(None)

print(f"Loaded {len(subjects)} subjects × {len(rois)} ROIs")

# ── Per-participant correlations ──────────────────────────────────────────────

full_rs        = {roi: [] for roi in rois}
semipartial_rs = {roi: [] for roi in rois}

for roi in rois:
    for sub_rdm in roi_rdms[roi]:
        if sub_rdm is None:
            full_rs[roi].append(np.nan)
            semipartial_rs[roi].append(np.nan)
            continue

        y = sub_rdm.values.flatten()
        full_rs[roi].append(pearsonr(y, vec_beh)[0])
        semipartial_rs[roi].append(semipartial_corr(y, vec_beh, model_covs)[0])

# ── Group-level summary ───────────────────────────────────────────────────────

print(f"\nBrain-Behaviour RSA  (N = {len(subjects)})")
print("=" * 75)
print(f"{'ROI':<42} {'Type':<14} {'Mean r':>8} {'SD':>8} {'p':>10}  sig")
print("-" * 75)

summary  = {}
csv_rows = []

for roi in rois:
    summary[roi] = {}
    for label, rs in [('full', full_rs[roi]), ('semipartial', semipartial_rs[roi])]:
        mean_r, sd_r, p = group_stats(rs)
        summary[roi][label] = (mean_r, sd_r, p)
        sig = sig_stars(p)
        print(f"{roi:<42} {label:<14} {mean_r:>8.2f} {sd_r:>8.2f} {p:>10.4f}  {sig}")
        csv_rows.append({'roi': roi, 'corr_type': label, 'mean_r': round(mean_r, 4),
                         'sd_r': round(sd_r, 4), 'p': round(p, 4), 'sig': sig})

print("=" * 75)

pd.DataFrame(csv_rows).to_csv(
    os.path.join(out_results_dir, 'brain_beh_partial_corr_summary.csv'), index=False)
print("Results saved to output/results/brain_beh_partial_corr_summary.csv")

# ── Load noise ceiling ────────────────────────────────────────────────────────

import json
nc_file = os.path.join(reliability_dir, 'roi_avg-category-8x8_fullmatrix_reliabilities.json')
noise_ceilings = {}
try:
    nc_data = json.load(open(nc_file))
    for roi in rois:
        if roi in nc_data:
            vals = nc_data[roi].get('L', []) + nc_data[roi].get('R', [])
            noise_ceilings[roi] = (np.mean(vals), np.std(vals))
except FileNotFoundError:
    print(f"Noise ceiling file not found: {nc_file}")

# ── Plot: full vs semi-partial per ROI ───────────────────────────────────────

n_rois    = len(rois)
bar_width = 0.35
x         = np.arange(n_rois)

fig, ax = plt.subplots(figsize=(16, 6))

for j, (label, color, offset) in enumerate([
    ('full',        BRAIN_BEH_COLOR,           -bar_width / 2),
    ('semipartial', CORR_COLORS['semipartial'],  bar_width / 2),
]):
    means      = [summary[roi][label][0] for roi in rois]
    sds        = [summary[roi][label][1] for roi in rois]
    ps         = [summary[roi][label][2] for roi in rois]
    legend_lbl = 'Full r' if label == 'full' else 'Semi-partial r'

    bars = ax.bar(x + offset, means, bar_width, yerr=sds,
                  color=color, label=legend_lbl, **BAR_DEFAULTS)

    for i, (bar, p) in enumerate(zip(bars, ps)):
        sig = sig_stars(p)
        if sig != 'ns':
            ax.text(bar.get_x() + bar.get_width() / 2,
                    means[i] + sds[i] + 0.005,
                    sig, ha='center', va='bottom', fontsize=APA_STAR)

for i, roi in enumerate(rois):
    if roi in noise_ceilings:
        nc_mean, nc_std = noise_ceilings[roi]
        ax.hlines(nc_mean, i - bar_width, i + bar_width,
                  color=NOISE_CEIL_COLOR, linewidth=1, linestyle='--', zorder=1)
        ax.fill_between([i - bar_width, i + bar_width],
                        nc_mean - nc_std / 2, nc_mean + nc_std / 2,
                        color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0)

def _roi_label(roi, sep=' '):
    if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
        return 'MT+ Complex & Neighboring Visual Areas' if sep == ' ' else 'MT+ Complex &\nNeighboring Visual Areas'
    label = roi.replace('_', ' ')
    if sep != '\n':
        return label
    words = label.split()
    if len(words) == 1:
        return label
    mid = len(words) // 2
    return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])

ax.axhline(0, color='k', linewidth=0.5, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels([_roi_label(r) for r in rois],
                   rotation=45, ha='right', fontsize=APA_TICK_ROI)
ax.set_ylabel("Pearson's r")
ax.legend(frameon=False)
plt.tight_layout()

save_path = os.path.join(out_fig_dir, 'brain_beh_partial_correlations.png')
plt.savefig(save_path, dpi=300)
print(f"Figure saved to {save_path}")
plt.show()

# ── Rotated version: ROIs on y-axis ──────────────────────────────────────────

bar_h  = 0.35
GROUP_SPACING = 0.85
y_pos  = np.arange(n_rois) * GROUP_SPACING
roi_labels = [_roi_label(r, sep='\n') for r in rois]

fig2, ax2 = plt.subplots(figsize=(7, 9.5))

bar_specs = [
    ('full',        BRAIN_BEH_COLOR,           -bar_h / 2, 'Full r'),
    ('semipartial', CORR_COLORS['semipartial'],  bar_h / 2, 'Semi-partial r'),
]

for label, color, offset, legend_lbl in bar_specs:
    means2 = [summary[roi][label][0] for roi in rois]
    sds2   = [summary[roi][label][1] for roi in rois]
    ps2    = [summary[roi][label][2] for roi in rois]

    bars2 = ax2.barh(y_pos + offset, means2, height=bar_h, xerr=sds2,
                     color=color, label=legend_lbl, **BAR_DEFAULTS)

    for y, mean, sd, p in zip(y_pos + offset, means2, sds2, ps2):
        sig = sig_stars(p)
        if sig != 'ns':
            x_s = mean + sd + 0.004 if mean >= 0 else mean - sd - 0.004
            ax2.text(x_s, y, sig, ha='left' if mean >= 0 else 'right', va='center', fontsize=APA_STAR)

# Noise ceiling
for roi_idx, roi in enumerate(rois):
    if roi in noise_ceilings:
        nc_mean, nc_std = noise_ceilings[roi]
        y0, y1 = y_pos[roi_idx] - bar_h, y_pos[roi_idx] + bar_h
        ax2.vlines(nc_mean, y0, y1,
                   color=NOISE_CEIL_COLOR, linewidth=1.5, linestyle='--', zorder=3,
                   label='Noise ceiling' if roi_idx == 0 else '')
        ax2.fill_betweenx([y0, y1], nc_mean - nc_std / 2, nc_mean + nc_std / 2,
                          color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0)

ax2.axvline(0, color='k', linewidth=0.5, zorder=0)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(roi_labels, fontsize=APA_TICK_ROI)
ax2.invert_yaxis()
ax2.set_ylim(y_pos[-1] + 0.45, y_pos[0] - 0.45)
ax2.set_xlabel("Pearson's r")
ax2.legend(frameon=False, fontsize=APA_TICK)
plt.tight_layout()

save_h = os.path.join(out_fig_dir, 'brain_beh_partial_correlations_horizontal.png')
plt.savefig(save_h, dpi=300, bbox_inches='tight')
print(f"Rotated figure saved to {save_h}")
plt.show()
