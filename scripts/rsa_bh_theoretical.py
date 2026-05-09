#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Correlate the average similarity RDM with theoretical RDMs.
Shows full Pearson r and semi-partial r as paired bars per model.
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_1samp
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

from src import images
from src.utils import upper_triangle
from src.plot_config import MODEL_COLORS, NOISE_CEIL_COLOR, BAR_DEFAULTS, apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_STAR

# ── Directories ───────────────────────────────────────────────────────────────

theoretical_models_dir = os.path.join('models', 'theoretical_models')
out_rdms_dir    = os.path.join('output', 'rdms')
out_avg_dir     = os.path.join('output', 'averaged')
out_fig_dir     = os.path.join('output', 'figures')
reliability_dir = os.path.join('output', 'reliability')

# ── Data ──────────────────────────────────────────────────────────────────────

avg_similarity_rdm = pd.read_table(
    os.path.join(out_avg_dir, 'average_similarity_rdm.tsv'), index_col=0
).reindex(index=images, columns=images)

similarity_rdms = {
    os.path.basename(f).split('_')[0]:
    pd.read_table(f, sep='\t', index_col=0).reindex(index=images, columns=images)
    for f in glob.glob(os.path.join(out_rdms_dir, '*.tsv'))
}

theoretical_models = {
    os.path.basename(f).split('_')[0]:
    pd.read_table(f, sep='\t', index_col=0).reindex(index=images, columns=images)
    for f in glob.glob(os.path.join(theoretical_models_dir, '*80*.tsv'))
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return ''

def fisher_mean_r(rs):
    zs = np.arctanh(np.clip(rs, -0.9999, 0.9999))
    return np.tanh(np.mean(zs))

def group_stats(rs):
    zs = np.arctanh(np.clip(rs, -0.9999, 0.9999))
    _, p = ttest_1samp(zs, 0)
    return fisher_mean_r(rs), np.std(rs, ddof=1), p

def _residuals(x, *covariates):
    X = np.column_stack([np.ones(len(x))] + list(covariates))
    coef, _, _, _ = lstsq(X, x, rcond=None)
    return x - X @ coef

def _lighten(hex_color, factor=0.42):
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * factor,
            g + (1 - g) * factor,
            b + (1 - b) * factor)

# ── Full r correlations (per subject) ─────────────────────────────────────────

correlations = {m: [] for m in theoretical_models}
for model_name, model_rdm in theoretical_models.items():
    mv = upper_triangle(model_rdm.values)
    for sub_rdm in similarity_rdms.values():
        correlations[model_name].append(
            pearsonr(upper_triangle(sub_rdm.values), mv)[0]
        )

# ── Semi-partial r (sr) correlations (per subject) ───────────────────────────
# sr(Y, X_i): correlate Y with residuals of X_i after removing all other models.

model_vecs = {m: upper_triangle(rdm.values)
              for m, rdm in theoretical_models.items()}

sr_correlations = {m: [] for m in theoretical_models}
for model_name in theoretical_models:
    covs = [v for m, v in model_vecs.items() if m != model_name]
    resid_m = _residuals(model_vecs[model_name], *covs)
    for sub_rdm in similarity_rdms.values():
        y = upper_triangle(sub_rdm.values)
        sr_correlations[model_name].append(pearsonr(y, resid_m)[0])

# ── Summary print ─────────────────────────────────────────────────────────────

print(f"\n{'Model':<16} {'Full r':>7} {'SD':>7} {'p':>10}  "
      f"{'sr':>7} {'SD':>7} {'p':>10}  sig_sr")
print("-" * 72)
for m in theoretical_models:
    mr, msd, mp   = group_stats(correlations[m])
    sr, ssd, sp   = group_stats(sr_correlations[m])
    print(f"{m:<16} {mr:>7.3f} {msd:>7.3f} {mp:>10.4f}  "
          f"{sr:>7.3f} {ssd:>7.3f} {sp:>10.4f}  {sig_stars(sp)}")
print("-" * 72)

# ── Plot setup ────────────────────────────────────────────────────────────────

apply_defaults()
noise_ceilings    = json.load(open(os.path.join(reliability_dir,
                                                'behaviour_reliabilities.json')))['correlations']
avg_noise_ceiling = np.mean(noise_ceilings)
nc_sd  = np.std(noise_ceilings)
nc_lo  = avg_noise_ceiling - nc_sd / 2
nc_hi  = avg_noise_ceiling + nc_sd / 2

model_keys  = list(correlations.keys())
model_names = [k.split('80')[0].capitalize() for k in model_keys]

means    = [group_stats(correlations[m])[0]    for m in model_keys]
stds     = [group_stats(correlations[m])[1]    for m in model_keys]
pvals    = [group_stats(correlations[m])[2]    for m in model_keys]

sr_means = [group_stats(sr_correlations[m])[0] for m in model_keys]
sr_stds  = [group_stats(sr_correlations[m])[1] for m in model_keys]
sr_pvals = [group_stats(sr_correlations[m])[2] for m in model_keys]

colors    = [MODEL_COLORS.get(n, '#888888') for n in model_names]
sr_colors = [_lighten(c) for c in colors]

# ── Figure builder ────────────────────────────────────────────────────────────

BAR_W = 0.38   # width of each bar
OFFSET = 0.21  # half-gap between the two bars in a group

def _build_figure():
    x = np.arange(len(model_names))
    x_f = x - OFFSET   # full r bar centres
    x_s = x + OFFSET   # sr bar centres

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── Bars ──────────────────────────────────────────────────────────────────
    bars_f = ax.bar(x_f, means,    BAR_W, yerr=stds,
                    color=colors,    **BAR_DEFAULTS)
    bars_s = ax.bar(x_s, sr_means, BAR_W, yerr=sr_stds,
                    color=sr_colors, **BAR_DEFAULTS)

    # ── Individual subject dots ────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for i, m_key in enumerate(model_keys):
        j = rng.uniform(-0.10, 0.10, len(correlations[m_key]))
        ax.scatter(x_f[i] + j, correlations[m_key],
                   color=colors[i], s=18, alpha=0.5, zorder=3, edgecolors='none')
        j2 = rng.uniform(-0.10, 0.10, len(sr_correlations[m_key]))
        ax.scatter(x_s[i] + j2, sr_correlations[m_key],
                   color=sr_colors[i], s=18, alpha=0.5, zorder=3, edgecolors='none')

    # ── Noise ceiling ─────────────────────────────────────────────────────────
    ax.hlines(avg_noise_ceiling, -0.5, len(model_names) - 0.5,
              color=NOISE_CEIL_COLOR, linestyle='--', linewidth=1.5)
    ax.fill_between([-0.5, len(model_names) - 0.5], nc_lo, nc_hi,
                    color=NOISE_CEIL_COLOR, alpha=0.2)

    # ── Significance markers — full r ──────────────────────────────────────────
    for i, (mean, std, p) in enumerate(zip(means, stds, pvals)):
        sig = sig_stars(p)
        if sig:
            top = mean + std
            y   = nc_hi + 0.01 if top > nc_lo else top + 0.01
            ax.text(x_f[i], y, sig,
                    ha='center', va='bottom', fontsize=APA_STAR)

    # ── Significance markers — sr ──────────────────────────────────────────────
    for i, (mean, std, p) in enumerate(zip(sr_means, sr_stds, sr_pvals)):
        sig = sig_stars(p)
        if sig:
            ax.text(x_s[i], mean + std + 0.01, sig,
                    ha='center', va='bottom', fontsize=APA_STAR)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlim(-0.5, len(model_names) - 0.5)
    ax.set_ylim(bottom=None, top=nc_hi + 0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=APA_TICK)
    ax.set_ylabel("Pearson's $r$", fontsize=APA_LABEL)

    # ── Legend: neutral grey to convey dark=full, light=semi-partial ──────────
    lh_full    = mpatches.Patch(facecolor='#4d4d4d', alpha=0.85,
                                label='Full $r$ ± $SD$ (darker)')
    lh_sr      = mpatches.Patch(facecolor='#b3b3b3', alpha=0.85,
                                label='Semi-partial $r$ ± $SD$ (lighter)')
    lh_nc_line = mlines.Line2D([], [], color=NOISE_CEIL_COLOR,
                               linestyle='--', linewidth=1.5)
    lh_nc_band = mpatches.Patch(facecolor=NOISE_CEIL_COLOR, alpha=0.2,
                                edgecolor='none')
    ax.legend(
        handles=[lh_full, lh_sr, (lh_nc_line, lh_nc_band)],
        labels=['Full $r$ ± $SD$ (darker)',
                'Semi-partial $r$ ± $SD$ (lighter)',
                'Noise ceiling ± $SD$'],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc='upper right', frameon=False, fontsize=APA_TICK,
    )
    plt.tight_layout()
    return fig

# ── Save ──────────────────────────────────────────────────────────────────────

fig = _build_figure()
fig.savefig(
    os.path.join(out_fig_dir, 'behaviour_theoretical_correlations.png'), dpi=300)
print(f"Saved: {os.path.join(out_fig_dir, 'behaviour_theoretical_correlations.png')}")
plt.show()
