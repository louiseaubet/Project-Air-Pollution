# -*- coding: utf-8 -*-
"""some helper functions"""
import numpy as np
import csv
import pandas as pd
import scipy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


# Build indices for cross-validation using venetian blinds technique
def build_k_indices(y, nb_fold):
    "build indices for venetian blind "
    num_row = y.shape[0]-1
    interval = int(num_row / nb_fold)
    indices = np.linspace(1, num_row, num_row)
    for k in range(nb_fold):
        k_indices = np.array(indices[k:num_row:interval], dtype=int)
        yield k_indices, k_indices
        
        
# Show plot of prediction vs reference + fit      
def print_result(y_ref, y_pred, title):
    # Fit a line between reference and prediction
    z = np.polyfit(y_ref, y_pred, 1)
    plt.scatter(y_ref, y_pred, c='red')
    #Plot the best fit line
    plt.plot(y_ref, np.polyval(z, y_ref), c='green', linewidth=1, label='fit')
    plt.plot(y_ref, y_ref, color='black', linewidth=1, label='$y=x$') # line y = x 
    plt.xlabel('Reference [$\mu$g/cm²]')
    plt.ylabel('Prediction [$\mu$g/cm²]')
    plt.legend()
    plt.title(title)
    plt.show()