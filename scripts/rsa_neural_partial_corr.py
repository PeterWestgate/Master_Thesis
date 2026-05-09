#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Neural RSA: full and semi-partial correlations
between per-subject ROI RDMs and theoretical models (Category, Animacy, GIST).

Computed per participant per ROI, then Fisher-z averaged across participants.
Group-level significance: one-sample t-test on Fisher-z values vs. 0.

Full r:        corr(neural_RDM, X_model)
Semi-partial:  corr(neural_RDM, resid(X_model | other models))
               → unique variance X_model explains in neural_RDM

Bilateral ROI: L and R hemispheres averaged per subject.

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

from src import rois, conditions
from src.utils import upper_triangle
from src.plot_config import MODEL_COLORS, CORR_COLORS, BAR_DEFAULTS, NOISE_CEIL_COLOR, apply_defaults
from src.figure_style import APA_TICK, APA_TICK_ROI, APA_STAR, APA_ANNOT
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
    zs     = np.array([fisher_z(r) for r in rs])
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
brain_dir              = os.path.join(project_root, 'models', 'subject_rois')
out_fig_dir            = os.path.join(project_root, 'output', 'figures')
out_results_dir        = os.path.join(project_root, 'output', 'results')
os.makedirs(out_fig_dir, exist_ok=True)
os.makedirs(out_results_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

subjects = [f'sub-{str(n + 1).zfill(2)}' for n in range(20)]

print("Loading neural RDMs...")
roi_rdms = {roi: [] for roi in rois}
missing  = []

for s in subjects:
    for roi in rois:
        l_file = os.path.join(brain_dir, f'{s}_L_{roi}.tsv')
        r_file = os.path.join(brain_dir, f'{s}_R_{roi}.tsv')
        try:
            l_rdm = pd.read_table(l_file, sep='\t', index_col=0).reindex(index=conditions, columns=conditions)
            r_rdm = pd.read_table(r_file, sep='\t', index_col=0).reindex(index=conditions, columns=conditions)
            roi_rdms[roi].append((l_rdm + r_rdm) / 2)
        except FileNotFoundError as e:
            missing.append(str(e))
            roi_rdms[roi].append(None)

if missing:
    print(f"Warning: {len(missing)} file(s) not found. First missing: {missing[0]}")

model_files = sorted(glob.glob(os.path.join(theoretical_models_dir, '*40*.tsv')))
if not model_files:
    sys.exit(f"No 40×40 model files found in {theoretical_models_dir}")

theoretical_models = {
    os.path.basename(f).split('_')[0]: pd.read_table(f, sep='\t', index_col=0)
                                          .reindex(index=conditions, columns=conditions)
    for f in model_files
}
model_names = sorted(theoretical_models.keys())
print(f"Loaded {len(model_names)} theoretical models: {model_names}")
print(f"Loaded {len(rois)} ROIs × {len(subjects)} subjects")

# ── Per-participant correlations per ROI ──────────────────────────────────────

corr_types = ['full', 'semipartial']

results = {
    roi: {m: {ct: [] for ct in corr_types} for m in model_names}
    for roi in rois
}

for roi in rois:
    mvecs = {m: upper_triangle(theoretical_models[m].values) for m in model_names}
    for sub_idx, sub_rdm in enumerate(roi_rdms[roi]):
        if sub_rdm is None:
            for m in model_names:
                for ct in corr_types:
                    results[roi][m][ct].append(np.nan)
            continue

        y = upper_triangle(sub_rdm.values)

        for m in model_names:
            x1  = mvecs[m]
            cov = [mvecs[other] for other in model_names if other != m]

            results[roi][m]['full'].append(pearsonr(y, x1)[0])
            results[roi][m]['semipartial'].append(semipartial_corr(y, x1, cov)[0])

# ── Group-level summary ───────────────────────────────────────────────────────

summary  = {roi: {m: {} for m in model_names} for roi in rois}
csv_rows = []

for roi in rois:
    for m in model_names:
        label = m.replace('40', '').capitalize()
        for ct in corr_types:
            rs = [r for r in results[roi][m][ct] if not np.isnan(r)]
            if len(rs) < 2:
                summary[roi][m][ct] = (np.nan, np.nan, np.nan)
                continue
            mean_r, sd_r, p = group_stats(rs)
            summary[roi][m][ct] = (mean_r, sd_r, p)
            csv_rows.append({'roi': roi, 'model': label, 'corr_type': ct,
                             'n': len(rs), 'mean_r': round(mean_r, 4),
                             'sd_r': round(sd_r, 4), 'p': round(p, 4),
                             'sig': sig_stars(p)})

print(f"\nNeural RSA – Theoretical Model Correlations  (N = {len(subjects)})")
print("=" * 90)
print(f"{'ROI':<42} {'Model':<12} {'Type':<14} {'Mean r':>8} {'SD':>8} {'p':>10}  sig")
print("-" * 90)
for row in csv_rows:
    print(f"{row['roi']:<42} {row['model']:<12} {row['corr_type']:<14} "
          f"{row['mean_r']:>8.3f} {row['sd_r']:>8.3f} {row['p']:>10.4f}  {row['sig']}")
print("=" * 90)

pd.DataFrame(csv_rows).to_csv(
    os.path.join(out_results_dir, 'neural_partial_corr_summary.csv'), index=False)

sub_rows = []
for roi in rois:
    for m in model_names:
        label = m.replace('40', '').capitalize()
        for ct in corr_types:
            for sub_idx, sub in enumerate(subjects):
                sub_rows.append({'roi': roi, 'subject': sub, 'model': label,
                                 'corr_type': ct,
                                 'r': results[roi][m][ct][sub_idx]})
pd.DataFrame(sub_rows).to_csv(
    os.path.join(out_results_dir, 'neural_partial_corr_per_subject.csv'), index=False)

print("Results saved to output/results/")

# ── Plot: grouped bar chart per ROI (all models × 2 corr types) ──────────────

n_rois    = len(rois)
n_models  = len(model_names)
bar_width = 0.15
model_labels = {m: m.replace('40', '').capitalize() for m in model_names}

ct_alpha = {'full': 1.0, 'semipartial': 0.55}

fig, ax = plt.subplots(figsize=(18, 6))
x   = np.arange(n_rois)

slot = 0
n_slots = n_models * len(corr_types)
for j, m in enumerate(model_names):
    base_color = MODEL_COLORS.get(model_labels[m], '#888888')
    for k, ct in enumerate(corr_types):
        offset = (slot - (n_slots - 1) / 2) * bar_width
        means  = [summary[roi][m][ct][0] for roi in rois]
        sds    = [summary[roi][m][ct][1] for roi in rois]
        label  = f"{model_labels[m]} ({'Full' if ct == 'full' else 'Semi-partial'})"

        ax.bar(x + offset, means, bar_width, yerr=sds,
               color=base_color, label=label,
               **{**BAR_DEFAULTS, 'alpha': ct_alpha[ct],
                  'error_kw': {'elinewidth': 1, 'capsize': 0, 'ecolor': '#555555'}})
        slot += 1

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
ax.set_xticklabels([_roi_label(r, sep='\n') for r in rois], fontsize=APA_TICK_ROI)
ax.set_ylabel("Pearson's r")
ax.legend(frameon=False, fontsize=APA_TICK, ncol=3, loc='upper right')
plt.tight_layout()

save_path = os.path.join(out_fig_dir, 'neural_partial_correlations_all_rois.png')
plt.savefig(save_path, dpi=200)
print(f"Full ROI figure saved to {save_path}")
plt.show()

# ── Plot: semi-partial correlations only, all ROIs ────────────────────────────

fig2, ax2 = plt.subplots(figsize=(14, 5))
bar_width2 = 0.25

for j, m in enumerate(model_names):
    offset = (j - 1) * bar_width2
    means  = [summary[roi][m]['semipartial'][0] for roi in rois]
    sds    = [summary[roi][m]['semipartial'][1] for roi in rois]
    ps     = [summary[roi][m]['semipartial'][2] for roi in rois]
    color  = MODEL_COLORS.get(model_labels[m], '#888888')

    bars = ax2.bar(x + offset, means, bar_width2, yerr=sds,
                   color=color, label=model_labels[m], **BAR_DEFAULTS)

    for i, (bar, p) in enumerate(zip(bars, ps)):
        sig = sig_stars(p)
        if sig != 'ns' and not np.isnan(means[i]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     means[i] + sds[i] + 0.005,
                     sig, ha='center', va='bottom', fontsize=APA_STAR)

ax2.axhline(0, color='k', linewidth=0.5, zorder=0)
ax2.set_xticks(x)
ax2.set_xticklabels([_roi_label(r) for r in rois], rotation=45, ha='right', fontsize=APA_TICK_ROI)
ax2.set_ylabel("Semi-partial Pearson's r")
ax2.legend(frameon=False)
plt.tight_layout()

save_path2 = os.path.join(out_fig_dir, 'neural_partial_correlations_partial_only.png')
plt.savefig(save_path2, dpi=300)
print(f"Semi-partial figure saved to {save_path2}")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# ROTATED FIGURES — ROIs on y-axis
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_SPACING = 0.80   # < 1 compresses inter-group gaps without shrinking bars
y_pos      = np.arange(n_rois) * GROUP_SPACING
roi_labels = [_roi_label(r, sep='\n') for r in rois]

# ── Figure A1: Full correlations only (rotated) ───────────────────────────────

bh_full  = 0.22   # bar height per model
offsets3 = np.array([-1, 0, 1]) * bh_full

fig_a1, ax_a1 = plt.subplots(figsize=(7, 9))

for j, m in enumerate(model_names):
    color  = MODEL_COLORS.get(model_labels[m], '#888888')
    means  = [summary[roi][m]['full'][0] for roi in rois]
    sds    = [summary[roi][m]['full'][1] for roi in rois]
    ps     = [summary[roi][m]['full'][2] for roi in rois]

    ax_a1.barh(y_pos + offsets3[j], means, height=bh_full, xerr=sds,
               color=color, label=model_labels[m],
               **{**BAR_DEFAULTS,
                  'error_kw': {'elinewidth': 1, 'capsize': 0, 'ecolor': '#555555'}})

    for y, mean, sd, p in zip(y_pos + offsets3[j], means, sds, ps):
        sig = sig_stars(p)
        if sig != 'ns' and not np.isnan(mean):
            x_s = mean + sd + 0.003 if mean >= 0 else mean - sd - 0.003
            ax_a1.text(x_s, y, sig, ha='left' if mean >= 0 else 'right', va='center', fontsize=APA_STAR)

ax_a1.axvline(0, color='k', linewidth=0.5, zorder=0)
ax_a1.set_yticks(y_pos)
ax_a1.set_yticklabels(roi_labels, fontsize=APA_TICK_ROI)
ax_a1.invert_yaxis()
ax_a1.set_ylim(y_pos[-1] + 0.40, y_pos[0] - 0.40)
ax_a1.set_xlabel("Full Pearson's r")
ax_a1.legend(frameon=False, fontsize=APA_TICK)
plt.tight_layout()
save_a1 = os.path.join(out_fig_dir, 'neural_full_correlations_horizontal.png')
plt.savefig(save_a1, dpi=300, bbox_inches='tight')
print(f"Figure A1 (rotated) saved to {save_a1}")
plt.show()

# ── Figure A2: Semi-partial correlations only (rotated) ───────────────────────

fig_a2, ax_a2 = plt.subplots(figsize=(7, 9))

for j, m in enumerate(model_names):
    color  = MODEL_COLORS.get(model_labels[m], '#888888')
    means  = [summary[roi][m]['semipartial'][0] for roi in rois]
    sds    = [summary[roi][m]['semipartial'][1] for roi in rois]
    ps     = [summary[roi][m]['semipartial'][2] for roi in rois]

    ax_a2.barh(y_pos + offsets3[j], means, height=bh_full, xerr=sds,
               color=color, label=model_labels[m],
               **{**BAR_DEFAULTS,
                  'error_kw': {'elinewidth': 1, 'capsize': 0, 'ecolor': '#555555'}})

    for y, mean, sd, p in zip(y_pos + offsets3[j], means, sds, ps):
        sig = sig_stars(p)
        if sig != 'ns' and not np.isnan(mean):
            x_s = mean + sd + 0.003 if mean >= 0 else mean - sd - 0.003
            ax_a2.text(x_s, y, sig, ha='left' if mean >= 0 else 'right', va='center', fontsize=APA_STAR)

ax_a2.axvline(0, color='k', linewidth=0.5, zorder=0)
ax_a2.set_yticks(y_pos)
ax_a2.set_yticklabels(roi_labels, fontsize=APA_TICK_ROI)
ax_a2.invert_yaxis()
ax_a2.set_ylim(y_pos[-1] + 0.40, y_pos[0] - 0.40)
ax_a2.set_xlabel("Semi-partial Pearson's r")
ax_a2.legend(frameon=False, fontsize=APA_TICK)
plt.tight_layout()
save_a2 = os.path.join(out_fig_dir, 'neural_semipartial_correlations_horizontal.png')
plt.savefig(save_a2, dpi=300, bbox_inches='tight')
print(f"Figure A2 (rotated) saved to {save_a2}")
plt.show()

# ── Figure A3: Combined full + semi-partial (rotated) ─────────────────────────
# 6 bars per ROI: [Category-full, Category-semi, Animacy-full, Animacy-semi, Gist-full, Gist-semi]

bh_comb  = 0.11
n_slots6 = n_models * 2
slot_offsets = (np.arange(n_slots6) - (n_slots6 - 1) / 2) * bh_comb

fig_a3, ax_a3 = plt.subplots(figsize=(8, 10))

ct_alpha = {'full': 1.0, 'semipartial': 0.55}
slot = 0
for j, m in enumerate(model_names):
    base_color = MODEL_COLORS.get(model_labels[m], '#888888')
    for ct in corr_types:
        means  = [summary[roi][m][ct][0] for roi in rois]
        sds    = [summary[roi][m][ct][1] for roi in rois]
        ps     = [summary[roi][m][ct][2] for roi in rois]
        lbl    = f"{model_labels[m]} ({'Full' if ct == 'full' else 'Semi-partial'})"

        ax_a3.barh(y_pos + slot_offsets[slot], means, height=bh_comb, xerr=sds,
                   color=base_color, label=lbl,
                   **{**BAR_DEFAULTS, 'alpha': ct_alpha[ct],
                      'error_kw': {'elinewidth': 0.8, 'capsize': 0, 'ecolor': '#555555'}})

        for y, mean, sd, p in zip(y_pos + slot_offsets[slot], means, sds, ps):
            sig = sig_stars(p)
            if sig != 'ns' and not np.isnan(mean):
                x_s = mean + sd + 0.002 if mean >= 0 else mean - sd - 0.002
                ax_a3.text(x_s, y, sig, ha='left' if mean >= 0 else 'right', va='center', fontsize=APA_STAR)
        slot += 1

ax_a3.axvline(0, color='k', linewidth=0.5, zorder=0)
ax_a3.set_yticks(y_pos)
ax_a3.set_yticklabels(roi_labels, fontsize=APA_TICK_ROI)
ax_a3.invert_yaxis()
ax_a3.set_ylim(y_pos[-1] + 0.40, y_pos[0] - 0.40)
ax_a3.set_xlabel("Pearson's r")
ax_a3.legend(frameon=False, fontsize=APA_TICK, ncol=1,
             loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
save_a3 = os.path.join(out_fig_dir, 'neural_all_correlations_horizontal.png')
plt.savefig(save_a3, dpi=300, bbox_inches='tight')
print(f"Figure A3 (rotated) saved to {save_a3}")
plt.show()
