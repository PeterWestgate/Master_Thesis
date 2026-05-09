#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
RSA between brain and similarity behavioural RDMs

- We take the average behavioural RDM across participants
- We correlate it with the 20 control-only brain RDMs per ROI
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

from src import images, rois, control_conditions, categories
from src.utils import upper_triangle
from src.plot_config import BRAIN_BEH_COLOR, NOISE_CEIL_COLOR, BAR_DEFAULTS, apply_defaults
from src.figure_style import APA_TICK, APA_TICK_ROI, APA_STAR

## Directories

model_dir = 'models'
control_brain_dir = os.path.join(model_dir, 'subject_control_rois')
avg_category_brain_dir = os.path.join(model_dir, 'subject_avg_category_rois')
theoretical_models_dir = os.path.join(model_dir, 'theoretical_models')
out_dir = 'output'
out_avg_dir = os.path.join(out_dir, 'averaged')
out_fig_dir = os.path.join(out_dir, 'figures')
reliability_dir = os.path.join(out_dir, 'reliability')

## Data
# Average similarity data
avg_similarity_rdm = pd.read_table(os.path.join(out_avg_dir, 'average_similarity_rdm.tsv'), index_col=0)
avg_similarity_rdm = avg_similarity_rdm.reindex(index = images, columns = images)
# Create 8x8 control similarity RDM (one entry per category)
control_similarity_rdm = np.zeros((len(categories), len(categories)))
for c_idx1, c1 in enumerate(categories):
    for c_idx2, c2 in enumerate(categories):
        c1_images = [img for img in images if c1 in img]
        c2_images = [img for img in images if c2 in img]
        c_df = avg_similarity_rdm.loc[c1_images, c2_images]
        c_value = c_df.values.mean()
        control_similarity_rdm[c_idx1, c_idx2] = c_value
control_similarity_df = pd.DataFrame(control_similarity_rdm, index=control_conditions, columns=control_conditions)

# Brain RDMs
subjects = [f'sub-{str(n+1).zfill(2)}' for n in range(20)]
roi_rdms = {roi: [] for roi in rois}
for s in subjects:
    for roi in rois:
        l_rdm_file = os.path.join(avg_category_brain_dir, f'{s}_L_{roi}.tsv')
        r_rdm_file = os.path.join(avg_category_brain_dir, f'{s}_R_{roi}.tsv')
        l_rdm = pd.read_table(l_rdm_file, sep='\t', index_col=0).reindex(index=control_conditions, columns=control_conditions)
        r_rdm = pd.read_table(r_rdm_file, sep='\t', index_col=0).reindex(index=control_conditions, columns=control_conditions)
        avg_rdm = (l_rdm + r_rdm) / 2
        roi_rdms[roi].append(avg_rdm)

## Correlations

# Calculate correlations with theoretical models
correlations = {roi: [] for roi in rois}
beh_flat = control_similarity_df.values.flatten()
for roi in rois:
    for sub_idx in range(len(roi_rdms[roi])):
        sub_rdm = roi_rdms[roi][sub_idx]
        sub_rdm = sub_rdm.reindex(index=control_conditions, columns=control_conditions)
        corr = pearsonr(sub_rdm.values.flatten(), beh_flat)
        correlations[roi].append(corr[0])

print("Brain - behaviour correlations:")
print("-" * 60)
print(f"{'ROI':<40} {'Mean r':>8} {'SD':>8} {'p':>10}  sig")
print("-" * 60)
for roi in rois:
    zs = np.arctanh(np.clip(correlations[roi], -0.9999, 0.9999))
    _, p = ttest_1samp(zs, 0)
    mean_r = np.mean(correlations[roi])
    sd_r = np.std(correlations[roi])
    sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
    print(f"{roi:<40} {mean_r:>8.2f} {sd_r:>8.2f} {p:>10.4f}  {sig}")
print("-" * 60)


## Plots

def sig_stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return ''

apply_defaults()
noise_ceilings = json.load(open(os.path.join(reliability_dir, 'roi_avg-category-8x8_fullmatrix_reliabilities.json')))

bar_width = 0.5
x_pos     = np.arange(len(rois))

# Group-level p-values
pvals = []
for roi in rois:
    zs = np.arctanh(np.clip(correlations[roi], -0.9999, 0.9999))
    _, p = ttest_1samp(zs, 0)
    pvals.append(p)

fig, ax = plt.subplots(figsize=(14, 6))

means = [np.mean(correlations[roi]) for roi in rois]
sds   = [np.std(correlations[roi])  for roi in rois]

bars = ax.bar(x_pos, means, width=bar_width, yerr=sds,
              color=BRAIN_BEH_COLOR, label='Mean r ± SD', **BAR_DEFAULTS)

# Significance asterisks
for bar, mean, sd, p in zip(bars, means, sds, pvals):
    sig = sig_stars(p)
    if sig:
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.005,
                sig, ha='center', va='bottom', fontsize=APA_STAR)

for roi_idx, roi in enumerate(rois):
    avg_nc = (np.mean(noise_ceilings[roi]['L']) + np.mean(noise_ceilings[roi]['R'])) / 2
    std_nc = (np.std(noise_ceilings[roi]['L'])  + np.std(noise_ceilings[roi]['R']))  / 2
    ax.hlines(avg_nc, roi_idx - bar_width / 2, roi_idx + bar_width / 2,
              color=NOISE_CEIL_COLOR, linestyle='-', linewidth=1.2, zorder=3)
    ax.fill_between(
        [roi_idx - bar_width / 2, roi_idx + bar_width / 2],
        avg_nc - std_nc / 2, avg_nc + std_nc / 2,
        color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0
    )

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

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([_roi_label(r) for r in rois], rotation=45, ha='right')
ax.set_ylabel("Pearson's r")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(out_fig_dir, 'brain_behaviour.png'), dpi=300)
plt.show()
plt.close()

# ── Rotated version: ROIs on y-axis ──────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(6, 8))

GROUP_SPACING = 0.75
y_pos      = np.arange(len(rois)) * GROUP_SPACING
bar_height = 0.55
roi_labels = [_roi_label(r, sep='\n') for r in rois]

bars2 = ax2.barh(y_pos, means, height=bar_height, xerr=sds,
                 color=BRAIN_BEH_COLOR, **BAR_DEFAULTS)

# Significance asterisks (to the right of bar + error)
for y, mean, sd, p in zip(y_pos, means, sds, pvals):
    sig = sig_stars(p)
    if sig:
        x_s = mean + sd + 0.005 if mean >= 0 else mean - sd - 0.005
        ax2.text(x_s, y, sig, ha='left' if mean >= 0 else 'right', va='center', fontsize=APA_STAR)

# Noise ceiling per ROI (vertical span within each ROI band)
for roi_idx, roi in enumerate(rois):
    avg_nc = (np.mean(noise_ceilings[roi]['L']) + np.mean(noise_ceilings[roi]['R'])) / 2
    std_nc = (np.std(noise_ceilings[roi]['L'])  + np.std(noise_ceilings[roi]['R']))  / 2
    y0, y1 = y_pos[roi_idx] - bar_height / 2, y_pos[roi_idx] + bar_height / 2
    ax2.vlines(avg_nc, y0, y1,
               color=NOISE_CEIL_COLOR, linestyle='-', linewidth=1.5, zorder=3,
               label='Noise ceiling' if roi_idx == 0 else '')
    ax2.fill_betweenx([y0, y1], avg_nc - std_nc / 2, avg_nc + std_nc / 2,
                      color=NOISE_CEIL_COLOR, alpha=0.2, zorder=0)

ax2.axvline(0, color='k', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(roi_labels, fontsize=APA_TICK_ROI)
ax2.invert_yaxis()
ax2.set_ylim(y_pos[-1] + 0.40, y_pos[0] - 0.40)
ax2.set_xlabel("Pearson's r")
ax2.legend(frameon=False, fontsize=APA_TICK)
plt.tight_layout()
save_h = os.path.join(out_fig_dir, 'brain_behaviour_horizontal.png')
plt.savefig(save_h, dpi=300, bbox_inches='tight')
print(f"Rotated figure saved to {save_h}")
plt.show()
plt.close()
