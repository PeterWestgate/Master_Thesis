
#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Calculate inter-participant correlation reliability of behavioural RDMs
'''
import sys
import os

# Voeg de bovenliggende map toe aan het zoekpad van Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import glob
import pandas as pd
import numpy as np
import json
from src.utils import upper_triangle, progress_bar

## Directories
out_dir = 'output'
out_data_dir = os.path.join(out_dir, 'rdms')
out_reliability_dir = os.path.join(out_dir, 'reliability')
out_filename = 'behaviour_reliabilities.json'

## Data

extension = '.tsv'
data_files = glob.glob(os.path.join(out_data_dir, f'*{extension}'))
data = {
    ppt_nb+1: pd.read_table(data_file, index_col = 0)
    for ppt_nb, data_file in enumerate(data_files)
}
participants = list(data.keys())
sample_size = len(participants)

# Calculate the size of each half (might not be an even number)
correlations = []

for ppt_idx, ppt in enumerate(participants):
    
    ppt_matrix = data[ppt].values
    
    other_ppts_avg_matrix = np.mean(
        [data[other_ppt].values for other_ppt in participants if other_ppt != ppt],
        axis=0
    )
    
    corr = np.corrcoef(
        upper_triangle(ppt_matrix),
        upper_triangle(other_ppts_avg_matrix)
    )[0, 1]
    
    correlations.append(corr)
    
    progress_bar(len(participants), ppt_idx)
    
# Print the result
print(f'Average correlation: {np.round(np.mean(correlations), 4)} ± {np.round(np.std(correlations), 4)} SD (sample size: {sample_size})')

# Save the results
output_file = os.path.join(out_reliability_dir, out_filename)
with open(output_file, 'w') as f:
    json.dump({
        'correlations': [np.round(corr, 4).item() for corr in correlations]
    }, f, indent=4)