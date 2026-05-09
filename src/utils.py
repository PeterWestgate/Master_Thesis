#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Utilities for inverse MDS analyses
'''

import numpy as np
import json
import pandas as pd

def upper_triangle(matrix):
    '''
    Return the flattened upper triangle of a matrix
    '''
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def progress_bar(n_iterations, current_iteration, n_chunks=5):
    '''
    Give some progress on the advancement of a for loop by
    printing a progress bar-looking string.
    '''
    # Ensure n_chunks does not exceed n_iterations and avoid division by zero
    chunks = min(n_chunks, n_iterations) if n_iterations > 0 else 1
    chunk_size = max(n_iterations // chunks, 1)

    if current_iteration == 0:
        portion = (current_iteration) / chunk_size
        progress_bar = '*' * int(portion) + '_' * (chunks - int(portion))
        print(f'Progress: [{progress_bar}] {current_iteration + 1}/{n_iterations} iteration complete')

    elif (current_iteration + 1) % chunk_size == 0 or (current_iteration + 1) == n_iterations:
        portion = (current_iteration + 1) / chunk_size
        progress_bar = '*' * int(portion) + '_' * (chunks - int(portion))
        print(f'Progress: [{progress_bar}] {current_iteration + 1}/{n_iterations} iterations complete')

        if (current_iteration + 1) == n_iterations:
            print(f'All {n_iterations} iterations completed')


# More comprehensive json file reading function
def read_json(json_file):
    '''Slightly enhanced reading of json files to process data faster'''
    # Load JSON file
    with open(json_file, "r") as f:
        results = json.load(f)
    # Keep only the one key from the dictionary
    if 'results' in results.keys():
        results = results['results']    
    return results

# Function to take in json file and reconstruct its content
def reconstruct_json(json_data):
    '''Use the reconstrucdataframe function on json loaded data'''
    # Convert each ROI's matrices
    for roi, roi_data in json_data.items():
        for matrix_name, matrix_data in roi_data.items():
            json_data[roi][matrix_name] = reconstruct_dataframe(matrix_data)
    
    return json_data

# Json reconstruction utility function
def reconstruct_dataframe(data):
    """ Convert a list of dictionaries into a Pandas DataFrame with labels """
    if isinstance(data, list) and isinstance(data[0], dict):
        col_labels = list(data[0].keys())  # Extract column names
        row_labels = col_labels.copy()  # Assuming it's square

        # Convert to matrix (replace None with NaN for missing values)
        matrix = np.array([[row[k] if row[k] is not None else np.nan for k in col_labels] for row in data])

        # Convert to DataFrame with row/column labels
        return pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    
    return data  # Return as is if already a matrix
