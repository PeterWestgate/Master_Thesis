#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
RSA between brain data and theoretical models
'''


## Imports
import sys
import os

# Voeg de bovenliggende map toe aan het zoekpad van Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_1samp
import matplotlib.pyplot as plt

from src import images, rois, conditions
from src.utils import upper_triangle
from src.plot_config import MODEL_COLORS, NOISE_CEIL_COLOR, BAR_DEFAULTS, apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_TICK_ROI, APA_STAR

## Directories

model_dir = 'models'
brain_dir = os.path.join(model_dir, 'subject_rois')
theoretical_models_dir = os.path.join(model_dir, 'theoretical_models')
out_dir = 'output'
out_fig_dir = os.path.join(out_dir, 'figures')
reliability_dir = os.path.join(out_dir, 'reliability')

## Data
subjects = [f'sub-{str(n+1).zfill(2)}' for n in range(20)]
# Brain RDMs
roi_rdms = {roi: [] for roi in rois}
for s in subjects:
    for roi in rois:
        l_rdm_file = os.path.join(brain_dir, f'{s}_L_{roi}.tsv')
        r_rdm_file = os.path.join(brain_dir, f'{s}_R_{roi}.tsv')
        l_rdm = pd.read_table(l_rdm_file, sep='\t', index_col=0).reindex(index=conditions, columns=conditions)
        r_rdm = pd.read_table(r_rdm_file, sep='\t', index_col=0).reindex(index=conditions, columns=conditions)
        avg_rdm = (l_rdm + r_rdm) / 2
        roi_rdms[roi].append(avg_rdm)


# Theoretical models
theoretical_model_files = glob.glob(os.path.join(theoretical_models_dir, '*40*.tsv'))
theoretical_models = {
    os.path.basename(file).split('_')[0]: pd.read_table(file, sep='\t', index_col=0).reindex(index=conditions, columns=conditions)
    for file in theoretical_model_files
}

## Correlations

# Calculate correlations with theoretical models
correlations = {roi: {model_name: [] for model_name in theoretical_models.keys()} for roi in rois}
for roi in rois:
    for model_name, model_rdm in theoretical_models.items():
        model_rdm = model_rdm.reindex(index=conditions, columns=conditions)
        model_vector = upper_triangle(model_rdm.values)
        for sub_idx in range(len(roi_rdms[roi])):
            sub_rdm = roi_rdms[roi][sub_idx]
            sub_rdm = sub_rdm.reindex(index=conditions, columns=conditions)
            sub_vector = upper_triangle(sub_rdm.values)
            corr = pearsonr(sub_vector, model_vector)
            correlations[roi][model_name].append(corr[0])
    

print("Brain - theoretical models correlations:")
print("-" * 75)
print(f"{'ROI':<40} {'Model':<14} {'Mean r':>8} {'SD':>8} {'p':>10}  sig")
print("-" * 75)
for roi in rois:
    for model_name, stats in correlations[roi].items():
        zs = np.arctanh(np.clip(stats, -0.9999, 0.9999))
        _, p = ttest_1samp(zs, 0)
        mean_r = np.mean(stats)
        sd_r = np.std(stats)
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        print(f"{roi:<40} {model_name:<14} {mean_r:>8.3f} {sd_r:>8.3f} {p:>10.4f}  {sig}")
print("-" * 75)


## Plots

def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return ''

apply_defaults()
noise_ceilings = json.load(open(os.path.join(reliability_dir, 'roi_40x40_uppertriangle_reliabilities.json')))

model_names = ['category40', 'animacy40', 'gist40']
n_rois   = len(rois)
n_models = len(model_names)

means = np.zeros((n_rois, n_models))
sds   = np.zeros((n_rois, n_models))
pvals = np.zeros((n_rois, n_models))
for i, roi in enumerate(rois):
    for j, model in enumerate(model_names):
        vals = correlations[roi][model]
        means[i, j] = np.mean(vals)
        sds[i, j]   = np.std(vals)
        zs           = np.arctanh(np.clip(vals, -0.9999, 0.9999))
        _, pvals[i, j] = ttest_1samp(zs, 0)

fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.2
x_pos    = np.arange(n_rois)
offsets  = np.linspace(-bar_width, bar_width, n_models)

for j, model in enumerate(model_names):
    label = model.split('40')[0].capitalize()
    color = MODEL_COLORS.get(label, '#888888')
    bars  = ax.bar(x_pos + offsets[j], means[:, j], width=bar_width,
                   yerr=sds[:, j], color=color, label=label, **BAR_DEFAULTS)
    for i, (bar, p) in enumerate(zip(bars, pvals[:, j])):
        sig = sig_stars(p)
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    means[i, j] + sds[i, j] + 0.005,
                    sig, ha='center', va='bottom', fontsize=APA_STAR)

for roi_idx, roi in enumerate(rois):
    avg_nc = (np.mean(noise_ceilings[roi]['L']) + np.mean(noise_ceilings[roi]['R'])) / 2
    std_nc = (np.std(noise_ceilings[roi]['L'])  + np.std(noise_ceilings[roi]['R']))  / 2
    ax.hlines(avg_nc, roi_idx - bar_width, roi_idx + bar_width,
              color=NOISE_CEIL_COLOR, linestyle='-', linewidth=1.2, zorder=3)
    ax.fill_between(
        [roi_idx - bar_width, roi_idx + bar_width],
        avg_nc - std_nc / 2, avg_nc + std_nc / 2,
        color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0
    )

def _roi_label(roi, sep=' '):
    if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
        return 'MT+ Complex & Neighboring Visual Areas' if sep == ' ' else 'MT+\nComplex &\nNeighboring\nVisual Areas'
    return roi.replace('_', sep)

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([_roi_label(r) for r in rois], rotation=45, ha='right')
ax.set_ylabel("Pearson's r")
ax.legend(frameon=False, ncol=3,
          loc='upper center', bbox_to_anchor=(0.5, -0.22))
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(os.path.join(out_fig_dir, 'brain_theoretical_grouped_barh.png'), dpi=300, bbox_inches='tight')
plt.show()

