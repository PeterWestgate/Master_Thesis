# -*- coding: utf-8 -*-
"""
Script to generate 8x8 theoretical Representational Dissimilarity Matrices (RDMs).

This script creates three RDMs based on theoretical models:
1. Animacy Model: Distinguishes between animate and inanimate categories.
2. Category Model: Each category is unique and dissimilar from all others.
3. GIST Model: Based on low-level visual features, averaged from an 80x80 RDM.

Author Peter Westgate
"""

## Imports

# Global imports
import os
import re
import pandas as pd
import numpy as np
import sys
import os

# Voeg de bovenliggende map toe aan het zoekpad van Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Hieronder staat je originele import (deze kun je laten staan)
from src import images, categories, category_palette
# Local imports
from src import categories, conditions, images

## Parameters

# Output directory where the generated 8x8 RDMs will be saved.
model_dir = 'models'
theoretical_models_dir = os.path.join(model_dir, 'theoretical_models')
os.makedirs(theoretical_models_dir, exist_ok=True)

def get_category_from_label(label):
    category = [c for c in categories if c in label][0]
    return category

## Create the animacy RDM

print("--- Generating 8x8 Animacy Model ---")
animacy_map = {
    'person': 'animate', 'cat': 'animate', 'bird': 'animate',
    'banana': 'inanimate', 'firehydrant': 'inanimate', 'tree': 'inanimate',
    'bus': 'inanimate', 'building': 'inanimate'
}
rdm = pd.DataFrame(index=categories, columns=categories, dtype=float)
for cat1 in categories:
    for cat2 in categories:
        rdm.loc[cat1, cat2] = 0.0 if animacy_map[cat1] == animacy_map[cat2] else 1.0

output_path = os.path.join(theoretical_models_dir, 'animacy8_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Animacy 8x8 RDM saved to: {output_path}\n")

print("--- Generating 40x40 Animacy Model ---")
rdm = pd.DataFrame(index=conditions, columns=conditions, dtype=float)
for cond1 in conditions:
    for cond2 in conditions:
        rdm.loc[cond1, cond2] = 0.0 if animacy_map[get_category_from_label(cond1)] == animacy_map[get_category_from_label(cond2)] else 1.0

output_path = os.path.join(theoretical_models_dir, 'animacy40_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Animacy 40x40 RDM saved to: {output_path}\n")


print("--- Generating 80x80 Animacy Model ---")
rdm = pd.DataFrame(index=images, columns=images, dtype=float)
for img1 in images:
    for img2 in images:
        rdm.loc[img1, img2] = 0.0 if animacy_map[get_category_from_label(img1)] == animacy_map[get_category_from_label(img2)] else 1.0

output_path = os.path.join(theoretical_models_dir, 'animacy80_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Animacy 80x80 RDM saved to: {output_path}\n")

## Create the category RDM

print("--- Generating 8x8 Category Model ---")
rdm = pd.DataFrame(1 - np.identity(len(categories)), index=categories, columns=categories)
output_path = os.path.join(theoretical_models_dir, 'category8_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Category 8x8 RDM saved to: {output_path}\n")


print("--- Generating 40x40 Category Model ---")
rdm = pd.DataFrame(index=conditions, columns=conditions, dtype=float)
for cond1 in conditions:
    for cond2 in conditions:
        rdm.loc[cond1, cond2] = 0.0 if get_category_from_label(cond1) == get_category_from_label(cond2) else 1.0

output_path = os.path.join(theoretical_models_dir, 'category40_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Category 40x40 RDM saved to: {output_path}\n")


print("--- Generating 80x80 Category Model ---")
rdm = pd.DataFrame(index=images, columns=images, dtype=float)
for img1 in images:
    for img2 in images:
        rdm.loc[img1, img2] = 0.0 if get_category_from_label(img1) == get_category_from_label(img2) else 1.0

output_path = os.path.join(theoretical_models_dir, 'category80_model.tsv')
rdm.to_csv(output_path, sep='\t', index=True)
print(f"Category 80x80 RDM saved to: {output_path}\n")

## Reduce the GIST 80x80 RDM to an 8x8 RDM


print("--- Generating 8x8 GIST Model ---")
gist_80x80_path = os.path.join(theoretical_models_dir, 'gist80_model.tsv')
gist_80x80_df = pd.read_table(gist_80x80_path, sep='\t', index_col=0)


sorted_labels = sorted(gist_80x80_df.index, key=get_category_from_label)
gist_80x80_reordered = gist_80x80_df.reindex(index=sorted_labels, columns=sorted_labels)

rdm_8x8 = pd.DataFrame(index=categories, columns=categories, dtype=float)
for i, cat1 in enumerate(categories):
    for j, cat2 in enumerate(categories):
        block = gist_80x80_reordered.iloc[i*10:(i+1)*10, j*10:(j+1)*10]
        if i == j:
            upper_triangle_values = block.values[np.triu_indices(10, k=1)]
            rdm_8x8.loc[cat1, cat2] = np.mean(upper_triangle_values)
        else:
            rdm_8x8.loc[cat1, cat2] = block.values.mean()

output_path = os.path.join(model_dir, 'gist8_model.csv')
rdm_8x8.to_csv(output_path)
print(f"GIST RDM saved to: {output_path}\n")



print("All theoretical models have been generated.")
