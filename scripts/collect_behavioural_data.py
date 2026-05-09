#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Collect behavioural data

1. Read aggregated inverse MDS data from Meadows
2. Compute similarity RDM per participant
3. Save it as .tsv
4. Plot individual RDMs and save as .png
'''
import sys
import os

# Voeg de bovenliggende map toe aan het zoekpad van Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Imports
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from src import images, categories, category_palette
from src.utils import progress_bar
from src.plot_config import apply_defaults

from src.figure_style import APA_LABEL, APA_TICK, APA_TICK_ROI, APA_STAR, APA_ANNOT
apply_defaults()

# Parameters
data_dir = 'data'
out_dir = 'output'
figures_dir = os.path.join(out_dir, 'figures')
rdms_dir = os.path.join(out_dir, 'rdms')

# Read the data — supports multiple CSV files in data/ (concatenated)
data_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
assert len(data_files) >= 1, f"No CSV files found in '{data_dir}'"
csv_data = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
print(f"Loaded {len(data_files)} data file(s): {[os.path.basename(f) for f in data_files]}")

# Find the participants and loop over them
participants = csv_data['participation'].unique()

print(f"Collecting data from {len(participants)} participants...")

for ppt_id, ppt in enumerate(participants):
    # Initialise a similarity matrix
    similarity_matrix = np.zeros((len(images), len(images)))
    trial_nb_matrix = np.zeros((len(images), len(images)))
    # Extract the data corresponding to this participant
    ppt_data = csv_data[csv_data['participation'] == ppt]
    # Find the unique trials for this participant
    ppt_trials = ppt_data['trial'].unique()

    for trial in ppt_trials:
        # Extract trial data
        trial_data = ppt_data[ppt_data['trial'] == trial]
        trial_images = trial_data['stim1_name'].values
        trial_coordinates = trial_data[['x', 'y']].values
        # Loop over pairs of images and find their distance
        for i, trial_image in enumerate(trial_images):
            for j, other_image in enumerate(trial_images):
                dist = np.linalg.norm(trial_coordinates[i] - trial_coordinates[j])
                idx_i = images.index(trial_image)
                idx_j = images.index(other_image)
                similarity_matrix[idx_i, idx_j] += dist
                trial_nb_matrix[idx_i, idx_j] += 1

    # Divide the similarity matrix by the number of trials
    avg_similarity_matrix = similarity_matrix / trial_nb_matrix
    # Save the results
    output_rdm = pd.DataFrame(avg_similarity_matrix, index=images, columns=images)
    output_rdm.to_csv(os.path.join(rdms_dir, f'sub-{str(ppt_id+1).zfill(2)}_similarity_rdm.tsv'), sep='\t')

    # Plot the number of trials per pair of images, raw matrix and averaged matrix
    matrix_palette = 'gray'
    n_images = len(images)
    n_categories = len(categories)
    category_width = n_images // n_categories

    f, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(trial_nb_matrix, aspect='equal', cmap=matrix_palette)
    ax[0].set_title('# trials')
    ax[0].axis('off')

    ax[1].imshow(similarity_matrix, aspect='equal', cmap=matrix_palette)
    ax[1].set_title('summed distances')
    ax[1].axis('off')

    ax[2].imshow(avg_similarity_matrix, aspect='equal', cmap=matrix_palette)
    ax[2].set_title('averaged distances')
    ax[2].axis('off')

    # Draw category color bar along the bottom (x-axis)
    for axis in ax:
        for i in range(n_categories):
            category = categories[i]
            start = i * category_width
            end = (i + 1) * category_width if i < n_categories - 1 else n_images
            rect = Rectangle(
                (start - 0.5, n_images - 0.5),  # x, y: move bar further down to be outside the matrix
                end - start,                    # width
                2.0,                            # height: thin bar
                color=category_palette[category],
                transform=axis.transData,
                clip_on=False,
                linewidth=0
            )
            axis.add_patch(rect)

        # Draw category color bar along the left (y-axis), outside the matrix
        for i in range(n_categories):
            category = categories[i]
            start = i * category_width
            end = (i + 1) * category_width if i < n_categories - 1 else n_images
            rect = Rectangle(
                (-2.5, start - 0.5),  # x, y: move bar further left to be outside the matrix
                2.0,                  # width: thin bar
                end - start,          # height
                color=category_palette[category],
                transform=axis.transData,
                clip_on=False,
                linewidth=0
            )
            axis.add_patch(rect)

    # Create a legend for the category palette
    legend_patches = [
        Rectangle((0, 0), 1, 1, color=category_palette[cat], label=cat)
        for cat in categories
    ]
    f.legend(
        handles=legend_patches,
        labels=categories,
        title="Categories",
        loc='center left',
        bbox_to_anchor=(0.95, 0.5),
        borderaxespad=0.,
        frameon=False
    )

    plt.colorbar(ax[0].images[0], ax=ax[0], fraction=0.03, pad=0.04)
    plt.colorbar(ax[1].images[0], ax=ax[1], fraction=0.03, pad=0.04)
    plt.colorbar(ax[2].images[0], ax=ax[2], fraction=0.03, pad=0.04)

    plt.suptitle(f'sub-{str(ppt_id+1).zfill(2)} similarity RDM', fontweight='bold', y=0.95, fontstyle='italic')
    plt.savefig(os.path.join(figures_dir, f'sub-{str(ppt_id+1).zfill(2)}_similarity_rdm.png'), dpi=300)
    # plt.show()

    progress_bar(len(participants), ppt_id)