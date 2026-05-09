#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Figure A4 — Neural RSA: Challenge vs Control model

Full Pearson r between per-subject ROI RDMs (40×40) and the binary
Challenge vs Control theoretical model, reported separately for left and
right hemispheres across 22 Glasser HCPMMP1 ROIs.

Challenge vs Control model:
  0 = both conditions share manipulation status
      (both control OR both in {clutter, deletion, occlusion, scrambling})
  1 = one control, one challenged

Correlation convention: upper triangle of the 40×40 RDM (k=1), matching
rsa_brain_theoretical.py and the rest of the neural RSA pipeline.

Noise ceiling: loaded from roi_40x40_uppertriangle_reliabilities.json,
identical to rsa_brain_theoretical.py.

Author: Peter Westgate
'''

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import rois, conditions
from src.utils import upper_triangle
from src.plot_config import NOISE_CEIL_COLOR, BAR_DEFAULTS, apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_TICK_ROI, APA_STAR

apply_defaults()

# Sky Blue from the Wong (2011) colorblind-safe palette — same palette as
# MODEL_COLORS / BRAIN_BEH_COLOR in plot_config.py
CHALLENGE_COLOR = '#56B4E9'

# ── Directories ───────────────────────────────────────────────────────────────

project_root    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
brain_dir       = os.path.join(project_root, 'models', 'subject_rois')
models_dir      = os.path.join(project_root, 'models')
out_fig_dir     = os.path.join(project_root, 'output', 'figures')
out_results_dir = os.path.join(project_root, 'output', 'results')
reliability_dir = os.path.join(project_root, 'output', 'reliability')
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(out_results_dir, exist_ok=True)

# ── Load and validate Challenge vs Control model ──────────────────────────────

model_path = os.path.join(models_dir, 'challenge_v_control_model.tsv')
if os.path.exists(model_path):
    model_df  = pd.read_table(model_path, index_col=0).reindex(
                    index=conditions, columns=conditions)
    model_mat = model_df.values.astype(float)

    assert model_mat.shape == (40, 40), \
        f"Expected 40×40, got {model_mat.shape}"
    assert np.allclose(model_mat, model_mat.T, atol=1e-8), \
        "Model matrix is not symmetric"
    assert np.all(np.diag(model_mat) == 0), \
        "Diagonal is not all zeros"
    unique_vals = set(np.unique(model_mat))
    assert unique_vals <= {0.0, 1.0}, \
        f"Non-binary values: {unique_vals}"
    print(f"Challenge vs Control model loaded and validated "
          f"(40×40, symmetric, binary, zero diagonal). "
          f"n_ones in upper triangle = {int(upper_triangle(model_mat).sum())}")
else:
    # Fallback: build from code
    print("TSV not found — building Challenge vs Control model from code.")
    def _status(cond):
        return 'control' if cond.split('_')[0] == 'control' else 'challenge'
    n = len(conditions)
    model_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and _status(conditions[i]) != _status(conditions[j]):
                model_mat[i, j] = 1.0

model_vec = upper_triangle(model_mat)

# ── Load neural RDMs per subject × ROI × hemisphere ──────────────────────────

subjects = [f'sub-{str(n + 1).zfill(2)}' for n in range(20)]

roi_rdms = {roi: {'L': [], 'R': []} for roi in rois}
n_missing = 0

for s in subjects:
    for roi in rois:
        for hemi in ['L', 'R']:
            fpath = os.path.join(brain_dir, f'{s}_{hemi}_{roi}.tsv')
            try:
                rdm = pd.read_table(fpath, sep='\t', index_col=0).reindex(
                          index=conditions, columns=conditions)
                roi_rdms[roi][hemi].append(rdm)
            except FileNotFoundError:
                roi_rdms[roi][hemi].append(None)
                n_missing += 1

if n_missing:
    print(f"Warning: {n_missing} file(s) not found — those subjects/hemispheres "
          f"will contribute NaN.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def group_stats(rs):
    """Fisher-z mean (back-transformed), r-space SD, t, p."""
    rs_clean = [r for r in rs if not np.isnan(r)]
    zs       = np.array([fisher_z(r) for r in rs_clean])
    mean_r   = np.tanh(np.mean(zs))
    sd_r     = np.std(rs_clean, ddof=1)
    t, p     = ttest_1samp(zs, 0)
    return mean_r, sd_r, t, p

def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return ''

# ── Per-subject correlations ──────────────────────────────────────────────────

correlations = {roi: {'L': [], 'R': []} for roi in rois}

for roi in rois:
    for hemi in ['L', 'R']:
        for sub_rdm in roi_rdms[roi][hemi]:
            if sub_rdm is None:
                correlations[roi][hemi].append(np.nan)
                continue
            sub_vec = upper_triangle(sub_rdm.values)
            correlations[roi][hemi].append(pearsonr(sub_vec, model_vec)[0])

# ── Load noise ceiling ────────────────────────────────────────────────────────

nc_file = os.path.join(reliability_dir, 'roi_40x40_uppertriangle_reliabilities.json')
noise_ceilings = json.load(open(nc_file))

# ── Group-level summary + CSV ─────────────────────────────────────────────────

print(f"\nFigure A4 — Challenge vs Control RSA  (N = {len(subjects)})")
print("=" * 90)
print(f"{'ROI':<44} {'Hemi':>4} {'Mean r':>8} {'SD':>7} {'t':>8} {'p':>10}  {'sig':<4}  {'NC':>6}")
print("-" * 90)

summary  = {roi: {} for roi in rois}
csv_rows = []

for roi in rois:
    for hemi in ['L', 'R']:
        mean_r, sd_r, t, p = group_stats(correlations[roi][hemi])
        nc_vals = noise_ceilings[roi][hemi]
        nc_mean = np.mean(nc_vals)
        nc_sd   = np.std(nc_vals)
        sig     = sig_stars(p)
        summary[roi][hemi] = (mean_r, sd_r, p, nc_mean, nc_sd)
        print(f"{roi:<44} {hemi:>4} {mean_r:>8.3f} {sd_r:>7.3f} "
              f"{t:>8.3f} {p:>10.4f}  {sig:<4}  {nc_mean:>6.3f}")
        csv_rows.append({
            'roi':                 roi,
            'hemisphere':          hemi,
            'mean_r':              round(mean_r, 4),
            'sd_r':                round(sd_r,   4),
            't':                   round(t,       4),
            'p':                   round(p,       4),
            'sig':                 sig,
            'noise_ceiling_mean':  round(nc_mean, 4),
        })

print("=" * 90)

csv_path = os.path.join(out_results_dir, 'challenge_v_control_rsa.csv')
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

n_rois    = len(rois)
bar_width = 0.35
x_pos     = np.arange(n_rois)

fig, ax = plt.subplots(figsize=(14, 6))

# BAR_DEFAULTS has edgecolor='none'; override for R hemisphere so hatch shows
bar_kw_L = {**BAR_DEFAULTS}
bar_kw_R = {**BAR_DEFAULTS, 'edgecolor': CHALLENGE_COLOR, 'linewidth': 0.6}

for hemi, offset, hatch, bar_kw, label in [
    ('L', -bar_width / 2, None,  bar_kw_L, 'Left hemisphere'),
    ('R',  bar_width / 2, '///', bar_kw_R, 'Right hemisphere'),
]:
    means = [summary[roi][hemi][0] for roi in rois]
    sds   = [summary[roi][hemi][1] for roi in rois]
    ps    = [summary[roi][hemi][2] for roi in rois]

    bars = ax.bar(x_pos + offset, means, bar_width, yerr=sds,
                  color=CHALLENGE_COLOR, hatch=hatch, label=label, **bar_kw)

    for bar, mean, sd, p in zip(bars, means, sds, ps):
        sig = sig_stars(p)
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + sd + 0.005,
                    sig, ha='center', va='bottom', fontsize=APA_STAR)

# Noise ceiling per hemisphere bar — positioned above each bar individually
for roi_idx, roi in enumerate(rois):
    for hemi, offset in [('L', -bar_width / 2), ('R', bar_width / 2)]:
        nc_mean = summary[roi][hemi][3]
        nc_sd   = summary[roi][hemi][4]
        x0 = roi_idx + offset - bar_width / 2
        x1 = roi_idx + offset + bar_width / 2
        ax.hlines(nc_mean, x0, x1,
                  color=NOISE_CEIL_COLOR, linestyle='-', linewidth=1.2, zorder=3)
        ax.fill_between([x0, x1],
                        nc_mean - nc_sd / 2, nc_mean + nc_sd / 2,
                        color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0)

def roi_label(roi):
    if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
        return 'MT+ Complex & Neighboring Visual Areas'
    return roi.replace('_', ' ')

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([roi_label(r) for r in rois],
                   rotation=45, ha='right', fontsize=APA_TICK_ROI)
ax.set_ylabel("Pearson's $r$", fontsize=APA_LABEL)

ax.legend(frameon=False, fontsize=APA_TICK,
          loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

save_base = os.path.join(out_fig_dir, 'figure_a4_challenge_v_control')
plt.savefig(save_base + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_base + '.svg', bbox_inches='tight')
print(f"Saved: {save_base}.png")
print(f"Saved: {save_base}.svg")
plt.show()

# ── Vertical figure (ROIs on y-axis) ─────────────────────────────────────────

bar_h  = 0.35
y_pos  = np.arange(n_rois)

fig2, ax2 = plt.subplots(figsize=(7, 8))

bar_kw_L_h = {**BAR_DEFAULTS,
              'error_kw': {'elinewidth': 1, 'capsize': 0, 'ecolor': '#555555'}}
bar_kw_R_h = {**bar_kw_L_h, 'edgecolor': CHALLENGE_COLOR, 'linewidth': 0.6}

for hemi, offset, hatch, bar_kw, label in [
    ('L', -bar_h / 2, None,  bar_kw_L_h, 'Left hemisphere'),
    ('R',  bar_h / 2, '///', bar_kw_R_h, 'Right hemisphere'),
]:
    means = [summary[roi][hemi][0] for roi in rois]
    sds   = [summary[roi][hemi][1] for roi in rois]
    ps    = [summary[roi][hemi][2] for roi in rois]

    bars2 = ax2.barh(y_pos + offset, means, bar_h, xerr=sds,
                     color=CHALLENGE_COLOR, hatch=hatch, label=label, **bar_kw)

    for bar, mean, sd, p in zip(bars2, means, sds, ps):
        sig = sig_stars(p)
        if sig:
            ax2.text(mean + sd + 0.005, bar.get_y() + bar.get_height() / 2,
                     sig, ha='left', va='center', fontsize=APA_STAR)

# Noise ceiling per hemisphere bar
for roi_idx, roi in enumerate(rois):
    for hemi, offset in [('L', -bar_h / 2), ('R', bar_h / 2)]:
        nc_mean = summary[roi][hemi][3]
        nc_sd   = summary[roi][hemi][4]
        y0 = roi_idx + offset - bar_h / 2
        y1 = roi_idx + offset + bar_h / 2
        ax2.vlines(nc_mean, y0, y1,
                   color=NOISE_CEIL_COLOR, linestyle='-', linewidth=1.5, zorder=3)
        ax2.fill_betweenx([y0, y1],
                          nc_mean - nc_sd / 2, nc_mean + nc_sd / 2,
                          color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0)

ax2.axvline(0, color='k', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([roi_label(r) for r in rois], fontsize=APA_TICK_ROI)
ax2.invert_yaxis()
ax2.set_xlabel("Pearson's $r$", fontsize=APA_LABEL)

ax2.legend(frameon=False, fontsize=APA_TICK, loc='lower right')
plt.tight_layout()

save_base_v = os.path.join(out_fig_dir, 'figure_a4_challenge_v_control_vertical')
plt.savefig(save_base_v + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_base_v + '.svg', bbox_inches='tight')
print(f"Saved: {save_base_v}.png")
print(f"Saved: {save_base_v}.svg")
plt.show()
