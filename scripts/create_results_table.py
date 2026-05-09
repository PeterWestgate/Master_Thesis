#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Generate APA-style Table 2: Brain-Behaviour RSA
Full and semi-partial Pearson r per ROI.

Correlations computed directly here (not from CSV) using .flatten()
on 8x8 category-level RDMs, consistent with rsa_brain_beh_partial_corr.py.

Author: Peter Westgate
'''

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
from numpy.linalg import lstsq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import images, rois, categories, control_conditions
from src.figure_style import apply_apa_style, APA_LABEL, APA_TICK
apply_apa_style()

# ── Directories ───────────────────────────────────────────────────────────────

project_root           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
avg_category_brain_dir = os.path.join(project_root, 'models', 'subject_avg_category_rois')
theoretical_models_dir = os.path.join(project_root, 'models', 'theoretical_models')
out_avg_dir            = os.path.join(project_root, 'output', 'averaged')
out_fig_dir            = os.path.join(project_root, 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _roi_label(roi):
    if roi == 'MT__Complex_and_Neighboring_Visual_Areas':
        return 'MT+ Complex & Neighboring Visual Areas'
    return roi.replace('_', ' ')

def _residuals(y, covariates):
    X = np.column_stack([np.ones(len(y))] + list(covariates))
    coef, _, _, _ = lstsq(X, y, rcond=None)
    return y - X @ coef

def semipartial_corr(y, x1, covariates):
    return pearsonr(y, _residuals(x1, covariates))[0]

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def group_stats(rs):
    rs = [r for r in rs if not np.isnan(r)]
    zs = np.array([fisher_z(r) for r in rs])
    return np.tanh(np.mean(zs)), np.std(rs, ddof=1), ttest_1samp(zs, 0)[1]

def sig_stars(p):
    if pd.isna(p): return ''
    if p < .001:   return '***'
    if p < .01:    return '**'
    if p < .05:    return '*'
    return ''

def fmt_cell(mean_r, sd_r, p):
    if pd.isna(mean_r): return '—'
    return f'{mean_r:.2f}{sig_stars(p)}\n({sd_r:.2f})'

# ── Load behavioural RDM → 8×8 flatten ───────────────────────────────────────

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

# ── Load theoretical model covariates → flatten ───────────────────────────────

def load_model_8x8(fname):
    ext  = os.path.splitext(fname)[1]
    sep  = '\t' if ext == '.tsv' else ','
    df   = pd.read_csv(os.path.join(theoretical_models_dir, fname), sep=sep, index_col=0)
    df.index   = control_conditions
    df.columns = control_conditions
    return df

models_8x8 = {
    'category': load_model_8x8('category8_model.tsv'),
    'animacy':  load_model_8x8('animacy8_model.tsv'),
    'gist':     load_model_8x8('gist8_model.csv'),
}
model_covs = [m.values.flatten() for m in models_8x8.values()]

# ── Load neural RDMs ──────────────────────────────────────────────────────────

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

# ── Compute correlations ──────────────────────────────────────────────────────

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
        semipartial_rs[roi].append(semipartial_corr(y, vec_beh, model_covs))

# ── Build table data ──────────────────────────────────────────────────────────

print(f"\nTable 2 — Brain-Behaviour RSA  (N = {len(subjects)})")
print("=" * 70)
print(f"{'ROI':<42} {'Full r (SD)':>14} {'Semi-partial r (SD)':>20}  sig")
print("-" * 70)

rows = []
for roi in rois:
    full_mean,  full_sd,  full_p  = group_stats(full_rs[roi])
    semi_mean,  semi_sd,  semi_p  = group_stats(semipartial_rs[roi])
    print(f"{roi:<42} {full_mean:>6.2f} ({full_sd:.2f}){sig_stars(full_p):3}  "
          f"{semi_mean:>6.2f} ({semi_sd:.2f}){sig_stars(semi_p):3}")
    rows.append({
        'ROI':                 _roi_label(roi),
        'Full r\n(SD)':        fmt_cell(full_mean,  full_sd,  full_p),
        'Semi-partial r\n(SD)': fmt_cell(semi_mean, semi_sd,  semi_p),
    })

print("=" * 70)

# ── Render table ──────────────────────────────────────────────────────────────

df_t2 = pd.DataFrame(rows)
n_rows = len(df_t2)

fig, ax = plt.subplots(figsize=(7, 2.5 + 0.45 * n_rows))
ax.axis('off')

tbl = ax.table(
    cellText=df_t2.values,
    colLabels=df_t2.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.55, 0.225, 0.225],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(APA_TICK)
tbl.scale(1, 2.2)

for col in range(len(df_t2.columns)):
    tbl[0, col].set_facecolor('#e8e8e8')
    tbl[0, col].set_text_props(weight='bold')

for row in range(1, n_rows + 1):
    for col in range(len(df_t2.columns)):
        tbl[row, col].set_facecolor('#f7f7f7' if row % 2 == 0 else 'white')

ax.set_title(
    'Table 2. Brain–Behaviour RSA by ROI\n'
    'Semi-partial r controls for Category, Animacy, and GIST models\n'
    r'$^*p < .05$,  $^{**}p < .01$,  $^{***}p < .001$  |  SD in parentheses',
    fontsize=APA_LABEL, fontweight='bold', pad=12
)

save_path = os.path.join(out_fig_dir, 'table2_brain_behaviour_rsa.png')
fig.savefig(save_path, bbox_inches='tight', dpi=200)
print(f"\nSaved: {save_path}")
plt.close(fig)

# ── Table A: Neural noise ceilings per ROI ────────────────────────────────────

reliability_dir = os.path.join(project_root, 'output', 'reliability')
nc_file = os.path.join(reliability_dir, 'roi_avg-category-8x8_fullmatrix_reliabilities.json')

try:
    nc_data = json.load(open(nc_file))
except FileNotFoundError:
    print(f"Noise ceiling file not found: {nc_file}")
    nc_data = {}

nc_rows = []
for roi in rois:
    if roi not in nc_data:
        nc_rows.append({'ROI': _roi_label(roi), 'Mean r\n(SD)': '—'})
        continue
    vals = nc_data[roi].get('L', []) + nc_data[roi].get('R', [])
    cell = f'{np.mean(vals):.2f}\n({np.std(vals):.2f})' if vals else '—'
    nc_rows.append({'ROI': _roi_label(roi), 'Mean r\n(SD)': cell})

df_nc = pd.DataFrame(nc_rows)
n_nc  = len(df_nc)

fig_nc, ax_nc = plt.subplots(figsize=(7, 2.5 + 0.45 * n_nc))
ax_nc.axis('off')

tbl_nc = ax_nc.table(
    cellText=df_nc.values,
    colLabels=df_nc.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.75, 0.25],
)
tbl_nc.auto_set_font_size(False)
tbl_nc.set_fontsize(APA_TICK)
tbl_nc.scale(1, 2.2)

for col in range(len(df_nc.columns)):
    tbl_nc[0, col].set_facecolor('#e8e8e8')
    tbl_nc[0, col].set_text_props(weight='bold')

for row in range(1, n_nc + 1):
    for col in range(len(df_nc.columns)):
        tbl_nc[row, col].set_facecolor('#f7f7f7' if row % 2 == 0 else 'white')

ax_nc.set_title(
    'Table A. Neural Noise Ceiling per ROI\n'
    'Split-half reliability (Pearson r), 8×8 category-averaged RDMs  |  SD in parentheses',
    fontsize=APA_LABEL, fontweight='bold', pad=12
)

save_nc = os.path.join(out_fig_dir, 'table_a_noise_ceiling.png')
fig_nc.savefig(save_nc, bbox_inches='tight', dpi=200)
print(f"Saved: {save_nc}")
plt.close(fig_nc)
