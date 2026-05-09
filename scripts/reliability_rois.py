#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Calculate inter-participant correlation reliability of ROI RDMs
'''

import os
import json
import numpy as np
from src import categories, rois, conditions, control_conditions
from src.utils import read_json, reconstruct_json, upper_triangle, progress_bar
from scipy.stats import pearsonr
import pandas as pd

## Directories

decoding_dir = '/Users/tim/fmri/categorisationtask/BIDS/derivatives/cosmomvpa/pairwise_decoding'
out_dir = 'output'
reliability_dir = os.path.join(out_dir, 'reliability')
subject_avg_category_rois_dir = os.path.join('models', 'subject_avg_category_rois')

## Parameters

control_conditions = [f'control_{c}' for c in categories]
sample_size = 20
subjects = [f'sub-{str(n+1).zfill(2)}' for n in range(sample_size)]
atlas = 'HCPMMP1cortices'
file_extension = '.json'

decoding_files = {
    sub: os.path.join(decoding_dir, sub, f'svm_pairwise_crossval_task-categorisationtask_{sub}_atlas-{atlas}_input-MNI_smoothing_contrasts{file_extension}')
    for sub in subjects
}

## Collect RDMs

roi_matrices = {
    'L': {roi: [] for roi in rois},
    'R': {roi: [] for roi in rois}
}
roi_control_matrices = {
    'L': {roi: [] for roi in rois},
    'R': {roi: [] for roi in rois}
}

for s in subjects:

    file = decoding_files[s]
    data = reconstruct_json(read_json(file))

    for roi in rois:
        for hemi in ['L', 'R']:

            roi_data = data[f'{hemi}_{roi}']['stimuli']
            roi_matrices[hemi][roi].append(roi_data)

            # Keep only the 8x8 control part of the matrix
            control_data = roi_data.loc[control_conditions, control_conditions]
            roi_control_matrices[hemi][roi].append(control_data)

            # Temporary line to save all the ROIs to tsv files
            # roi_data.reindex(index=conditions, columns=conditions).to_csv(f'models/subject_rois/{s}_{hemi}_{roi}.tsv', sep='\t')

            # Temporary line to average 40x40 into 8x8 and save all the ROIs to tsv files
            avg_roi_rdm = pd.DataFrame(np.zeros((len(control_conditions), len(control_conditions))), index=control_conditions, columns=control_conditions)
            for c_idx1, c1 in enumerate(categories):
                for c_idx2, c2 in enumerate(categories):
                    c1_images = [img for img in conditions if c1 in img]
                    c2_images = [img for img in conditions if c2 in img]
                    c_df = roi_data.loc[c1_images, c2_images]
                    c_value = np.nanmean(c_df.values)
                    avg_roi_rdm.iloc[c_idx1, c_idx2] = c_value
            avg_roi_rdm.reindex(index=control_conditions, columns=control_conditions).to_csv(f'models/subject_avg_category_rois/{s}_{hemi}_{roi}.tsv', sep='\t')



## Calculate reliability

control_reliabilities = {
    roi: {'L': [], 'R': []} for roi in rois
}
full_reliabilities = {
    roi: {'L': [], 'R': []} for roi in rois
}

for sub_idx, _ in enumerate(subjects):

    for roi in rois:

        for hemi in ['L', 'R']:

            # Full RDM reliability

            sub_rdm = roi_matrices[hemi][roi][sub_idx].values

            other_subs_avg_rdm = np.mean(
                [roi_matrices[hemi][roi][i].values for i in range(sample_size) if i != sub_idx], axis=0
            )

            # corr = pearsonr(
            #     np.nan_to_num(sub_rdm.flatten(), nan=0),
            #     np.nan_to_num(other_subs_avg_rdm.flatten(), nan=0)
            # )[0]
            corr = np.corrcoef(
                upper_triangle(sub_rdm),
                upper_triangle(other_subs_avg_rdm)
            )[0, 1]

            full_reliabilities[roi][hemi].append(corr)

            # Control RDM reliability

            control_sub_rdm = roi_control_matrices[hemi][roi][sub_idx].values

            other_subs_avg_rdm = np.mean(
                [roi_control_matrices[hemi][roi][i].values for i in range(sample_size) if i != sub_idx], axis=0
            )

            # corr = pearsonr(
            #     np.nan_to_num(control_sub_rdm.flatten(), nan=0),
            #     np.nan_to_num(other_subs_avg_rdm.flatten(), nan=0)
            # )[0]
            corr = np.corrcoef(
                upper_triangle(control_sub_rdm),
                upper_triangle(other_subs_avg_rdm)
            )[0, 1]

            control_reliabilities[roi][hemi].append(corr)

    progress_bar(len(subjects), sub_idx)

## Save the results
control_output_file = os.path.join(reliability_dir, 'roi_8x8_uppertriangle_reliabilities.json')
full_output_file = os.path.join(reliability_dir, 'roi_40x40_uppertriangle_reliabilities.json')
# control_output_file = os.path.join(reliability_dir, 'roi_8x8_fullmatrix_reliabilities.json')
# full_output_file = os.path.join(reliability_dir, 'roi_40x40_fullmatrix_reliabilities.json')

with open(control_output_file, 'w') as f:
    json.dump(control_reliabilities, f, indent=4)

with open(full_output_file, 'w') as f:
    json.dump(full_reliabilities, f, indent=4)


## Reliability of the average category 8x8 RDMs from the collapsed 40x40 RDMs

roi_matrices = {
    'L': {roi: [] for roi in rois},
    'R': {roi: [] for roi in rois}
}
for roi in rois:
    for s in subjects:
        l_rdm_file = os.path.join(subject_avg_category_rois_dir, f'{s}_L_{roi}.tsv')
        r_rdm_file = os.path.join(subject_avg_category_rois_dir, f'{s}_R_{roi}.tsv')
        l_rdm = pd.read_table(l_rdm_file, sep='\t', index_col=0).reindex(index=control_conditions, columns=control_conditions)
        r_rdm = pd.read_table(r_rdm_file, sep='\t', index_col=0).reindex(index=control_conditions, columns=control_conditions)
        roi_matrices['L'][roi].append(l_rdm)
        roi_matrices['R'][roi].append(r_rdm)

reliabilities = {
    roi: {'L': [], 'R': []} for roi in rois
}

for sub_idx, _ in enumerate(subjects):

    for roi in rois:

        for hemi in ['L', 'R']:

            sub_rdm = roi_matrices[hemi][roi][sub_idx].values

            other_subs_avg_rdm = np.mean(
                [roi_matrices[hemi][roi][i].values for i in range(sample_size) if i != sub_idx], axis=0
            )

            corr = np.corrcoef(
                upper_triangle(sub_rdm),
                upper_triangle(other_subs_avg_rdm)
            )[0, 1]

            reliabilities[roi][hemi].append(corr)

    progress_bar(len(subjects), sub_idx)

avg_category_output_file = os.path.join(reliability_dir, 'roi_avg-category-8x8_fullmatrix_reliabilities.json')

with open(avg_category_output_file, 'w') as f:
    json.dump(reliabilities, f, indent=4)
