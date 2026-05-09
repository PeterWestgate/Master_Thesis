#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
SCRIPT TO AVERAGE DISSIMILARITY MATRICES

This script reads all TSV files from a specified input directory,
assumes they are square dissimilarity matrices, and calculates an
average dissimilarity matrix. The resulting matrix is then saved
to a new TSV file.

1. Read all .tsv files from the input directory
2. Validate and store them as pandas DataFrames
3. Calculate the average dissimilarity matrix
4. Save the average matrix to a TSV file
5. Perform PCA on the average matrix and plot results
6. Save the average matrix plot and PCA plot as PNG files

Author Peter Westgate
'''
import sys
import os

# Voeg de bovenliggende map toe aan het zoekpad van Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import os
import numpy as np

from src import images, categories, category_palette
from src.plot_config import apply_defaults
from sklearn.decomposition import PCA
from scipy.spatial import procrustes


# ------------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
#    Define the input and output directories and filenames.
# ------------------------------------------------------------------------------

# The directory where the individual dissimilarity matrices are stored.
# This assumes the script is run from the project's root directory.
input_dir = 'output/rdms'

# The directory where the final average matrix will be saved.
output_dir = 'output/averaged'
fig_dir = 'output/figures'

# The filename for the output average matrix.
output_filename = 'average_similarity_rdm.tsv'


# Create the output directory if it doesn't already exist.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ------------------------------------------------------------------------------
# 2. READ AND VALIDATE MATRICES
#    Iterate through the input directory and load all matrices.
# ------------------------------------------------------------------------------

# Initialize an empty list to store the DataFrames of the matrices.
matrix_list = []

# Get a list of all files in the input directory.
try:
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.tsv')]
    if not file_list:
        print(f"No TSV files found in the directory: {input_dir}")
except FileNotFoundError:
    print(f"Error: The directory '{input_dir}' was not found.")
    print("Please make sure the directory and its contents exist.")
    exit()

print(f"Found {len(file_list)} TSV files to process.")

# Loop through each file, read it into a pandas DataFrame, and add it to the list.
for filename in file_list:
    filepath = os.path.join(input_dir, filename)
    try:
        df = pd.read_table(filepath, index_col=0)
        if not list(df.index) == images:
            print(f"Warning: '{filename}' row labels do not match expected image names. Re-ordering.")
            df = df.reindex(index=images, columns=images)

        # Optional: Add a check to ensure the file is a square matrix
        if df.shape[0] != df.shape[1]:
            print(f"Warning: Skipping file '{filename}'. It is not a square matrix.")
            continue
        matrix_list.append(df)
        print(f"Successfully loaded {filename}.")
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")


# ------------------------------------------------------------------------------
# 3. CALCULATE THE AVERAGE MATRIX
#    Sum all matrices and then divide by the number of matrices.
# ------------------------------------------------------------------------------

if not matrix_list:
    print("No valid matrices were loaded. Exiting.")
    exit()

# Start with a zero matrix of the correct size and with the correct labels.
# Use the first matrix in the list as a template for structure.
first_matrix = matrix_list[0]
average_matrix = pd.DataFrame(0, index=first_matrix.index, columns=first_matrix.columns)

# Sum all matrices in the list.
for matrix in matrix_list:
    average_matrix = average_matrix.add(matrix, fill_value=0)

# Calculate the mean by dividing the sum by the total number of matrices.
num_matrices = len(matrix_list)
average_matrix = average_matrix / num_matrices

print("\nSuccessfully calculated the average matrix.")

# ------------------------------------------------------------------------------
# 4. SAVE THE RESULT
#    Save the final averaged matrix to a new TSV file in the output directory.
# ------------------------------------------------------------------------------

output_filepath = os.path.join(output_dir, output_filename)
average_matrix.to_csv(output_filepath, sep='\t')

print(f"Average dissimilarity matrix saved to: {output_filepath}")


#  3b. PCA on the average matrix
# X = average_matrix.values
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

#  3b. PCA on each matrix + average

Xs = [matrix.values for matrix in matrix_list]

# ── Viewing angle: adjust these two values to rotate the 3D perspective ────────
AZIMUTH_DEG   = 190   # horizontal spin (left/right)
ELEVATION_DEG = 130   # vertical tilt  (up/down)
# ──────────────────────────────────────────────────────────────────────────────

def _view_project(coords_3d, az_deg, el_deg):
    """Rotate a (N,3) array and return its first 2 columns as the 2D view."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,           0,          1]])
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(el), -np.sin(el)   ],
                   [0, np.sin(el),  np.cos(el)   ]])
    return (coords_3d @ (Rz @ Rx).T)[:, :2]

pca = PCA(n_components=3)
Xs_pca = [pca.fit_transform(X) for X in Xs]
avg_pca_3d = sum(Xs_pca) / len(Xs_pca)

# Align each participant's 3D embedding to the group average via Procrustes
aligned_Xs_pca_3d = []
for coords in Xs_pca:
    _, aligned, _ = procrustes(avg_pca_3d, coords)
    aligned_Xs_pca_3d.append(aligned)

# Project to the chosen 2D view
aligned_Xs_pca = [_view_project(c, AZIMUTH_DEG, ELEVATION_DEG) for c in aligned_Xs_pca_3d]
avg_pca        = _view_project(avg_pca_3d, AZIMUTH_DEG, ELEVATION_DEG)
mean_coords    = np.mean(aligned_Xs_pca, axis=0)
std_coords     = np.std(aligned_Xs_pca, axis=0)


# 5. Plot the average matrix

print("\nCreating some plots of the average matrix and PCA results...")

from src.figure_style import APA_LABEL, APA_TICK, APA_TICK_ROI, APA_STAR, APA_ANNOT
apply_defaults()

# Render person last so it sits on top when overlapping with other categories
scatter_order = [c for c in categories if c != 'person'] + ['person']

# Plot the number of trials per pair of images, raw matrix and averaged matrix
matrix_palette = 'gray'
n_images = len(images)
n_categories = len(categories)
category_width = n_images // n_categories

## Figure version 1: simple, average matrix & average PCA

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.2, 0.8]}, figsize=(9, 4))

ax[0].imshow(average_matrix.values, aspect='equal', cmap=matrix_palette)
ax[0].axis('off')

# Draw category color bar along the bottom (x-axis)
for i in range(n_categories):
    category = categories[i]
    start = i * category_width
    end = (i + 1) * category_width if i < n_categories - 1 else n_images
    rect = Rectangle(
        (start - 0.5, n_images - 0.5),  # x, y: move bar further down to be outside the matrix
        end - start,                    # width
        2.0,                            # height: thin bar
        color=category_palette[category],
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

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
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)


# Plot the PCA positions
for category in scatter_order:
    i = categories.index(category)
    idx = slice(i * category_width, (i + 1) * category_width if i < n_categories - 1 else n_images)
    ax[1].scatter(
        avg_pca[idx, 0], avg_pca[idx, 1],
        color=category_palette[category],
        alpha=0.6,
        label=category,
        s=40,
        zorder=10 if category == 'person' else 5
    )

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])

ax[1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.suptitle(f'Average similarity RDM (n={num_matrices})', y=1.05, clip_on=False, fontweight='bold')
plt.savefig(os.path.join(fig_dir, 'average_similarity_rdm_pca.png'), dpi=300)
# plt.show()



## Version 2: average matrix and overlaid PCA from each participant

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.2, 0.8]}, figsize=(9, 4))

ax[0].imshow(average_matrix.values, aspect='equal', cmap=matrix_palette)
ax[0].axis('off')

# Draw category color bar along the bottom (x-axis)
for i in range(n_categories):
    category = categories[i]
    start = i * category_width
    end = (i + 1) * category_width if i < n_categories - 1 else n_images
    rect = Rectangle(
        (start - 0.5, n_images - 0.5),  # x, y: move bar further down to be outside the matrix
        end - start,                    # width
        2.0,                            # height: thin bar
        color=category_palette[category],
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

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
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)


# Plot the PCA positions — person images last so they render on top
person_idx = [i for i, img in enumerate(images) if 'person' in img]
other_idx  = [i for i, img in enumerate(images) if 'person' not in img]
for i in other_idx + person_idx:
    img = images[i]
    category = [cat for cat in categories if cat in img][0]
    ax[1].scatter(
        [pca[i][0] for pca in aligned_Xs_pca],
        [pca[i][1] for pca in aligned_Xs_pca],
        color=category_palette[category],
        alpha=0.4,
        label=category,
        s=40,
        edgecolor='none',
        zorder=10 if category == 'person' else 5
    )

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])

# Create a legend for the category palette
legend_patches = [
    Rectangle((0, 0), 1, 1, color=category_palette[cat], label=cat)
    for cat in categories
]
ax[1].legend(
    handles=legend_patches,
    labels=categories,
    title="Categories",
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    borderaxespad=0.,
    frameon=False,
)
plt.tight_layout()
plt.suptitle(f'Average similarity RDM (n={num_matrices}), PCA across ppts', y=1.05, clip_on=False, fontweight='bold')
plt.savefig(os.path.join(fig_dir, 'average_similarity_rdm_pca_img.png'), dpi=300)
# plt.show()



## Version 3: average matrix, PCA averaged across procrustes-aligned participants + SD

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.2, 0.8]}, figsize=(9, 4))

ax[0].imshow(average_matrix.values, aspect='equal', cmap=matrix_palette)
ax[0].axis('off')

# Draw category color bar along the bottom (x-axis)
for i in range(n_categories):
    category = categories[i]
    start = i * category_width
    end = (i + 1) * category_width if i < n_categories - 1 else n_images
    rect = Rectangle(
        (start - 0.5, n_images - 0.5),  # x, y: move bar further down to be outside the matrix
        end - start,                    # width
        2.0,                            # height: thin bar
        color=category_palette[category],
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

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
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

# Plot the SD as cicrles around the mean
for i, img in enumerate(images):
    category = [cat for cat in categories if cat in img][0]
    x_size, y_size = std_coords[i]
    circle = Ellipse(
        (mean_coords[i, 0], mean_coords[i, 1]),
        width=x_size,
        height=y_size,
        # facecolor='none',  # Transparent fill
        color=category_palette[category],
        alpha=0.05,         # More visible edge
        linewidth=0.0
    )
    ax[1].add_patch(circle)

# Plot the PCA positions
for category in scatter_order:
    i = categories.index(category)
    idx = slice(i * category_width, (i + 1) * category_width if i < n_categories - 1 else n_images)
    ax[1].scatter(
        mean_coords[idx, 0], mean_coords[idx, 1],
        color=category_palette[category],
        alpha=0.7,
        label=category,
        s=40,
        zorder=10 if category == 'person' else 5
    )

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])

# Create a legend for the category palette
legend_patches = [
    Rectangle((0, 0), 1, 1, color=category_palette[cat], label=cat)
    for cat in categories
]
ax[1].legend(
    handles=legend_patches,
    labels=categories,
    title="Categories",
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    borderaxespad=0.,
    frameon=False
)
plt.tight_layout()
plt.suptitle(f'Average similarity RDM (n={num_matrices})', y=1.05, clip_on=False, fontweight='bold')
plt.savefig(os.path.join(fig_dir, 'average_similarity_rdm_pca_std_opaque.png'), dpi=300)
# plt.show()


## Version 3bis: same as version 3 with circles instead of filled elipses

f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.2, 0.8]}, figsize=(9, 4))

ax[0].imshow(average_matrix.values, aspect='equal', cmap=matrix_palette)
ax[0].axis('off')

# Draw category color bar along the bottom (x-axis)
for i in range(n_categories):
    category = categories[i]
    start = i * category_width
    end = (i + 1) * category_width if i < n_categories - 1 else n_images
    rect = Rectangle(
        (start - 0.5, n_images - 0.5),  # x, y: move bar further down to be outside the matrix
        end - start,                    # width
        2.0,                            # height: thin bar
        color=category_palette[category],
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

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
        transform=ax[0].transData,
        clip_on=False,
        linewidth=0
    )
    ax[0].add_patch(rect)

# Plot the SD as cicrles around the mean
for i, img in enumerate(images):
    category = [cat for cat in categories if cat in img][0]
    x_size, y_size = std_coords[i]
    circle = Ellipse(
        (mean_coords[i, 0], mean_coords[i, 1]),
        width=x_size,
        height=y_size,
        facecolor='none',  # Transparent fill
        edgecolor=category_palette[category],
        alpha=0.2,         # More visible edge
        linewidth=0.5
    )
    ax[1].add_patch(circle)

# Plot the PCA positions
for category in scatter_order:
    i = categories.index(category)
    idx = slice(i * category_width, (i + 1) * category_width if i < n_categories - 1 else n_images)
    ax[1].scatter(
        mean_coords[idx, 0], mean_coords[idx, 1],
        color=category_palette[category],
        alpha=0.7,
        label=category,
        s=40,
        zorder=10 if category == 'person' else 5
    )

ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])

# Create a legend for the category palette
legend_patches = [
    Rectangle((0, 0), 1, 1, color=category_palette[cat], label=cat)
    for cat in categories
]
ax[1].legend(
    handles=legend_patches,
    labels=categories,
    title="Categories",
    loc='center left',
    bbox_to_anchor=(1.0, 0.5),
    borderaxespad=0.,
    frameon=False
)
plt.tight_layout()
plt.suptitle(f'Average similarity RDM (n={num_matrices})', y=1.05, clip_on=False, fontweight='bold')
plt.savefig(os.path.join(fig_dir, 'average_similarity_rdm_pca_std_line.png'), dpi=300)
# plt.show()

print("\nEnd of the script.")