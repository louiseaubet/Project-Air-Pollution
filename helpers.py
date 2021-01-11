# -*- coding: utf-8 -*-
"""some helper functions"""
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from imblearn.over_sampling import RandomOverSampler

######################

# Build indices for cross-validation using venetian blinds technique
def build_k_indices(y, nb_fold):
    "build indices for venetian blind "
    num_row = y.shape[0]-1
    interval = int(num_row / nb_fold)
    indices = np.linspace(0, num_row, num_row+1)
    for k in range(nb_fold):
        test_indices = np.array(indices[k:num_row:nb_fold], dtype=int)
        train_indices = set(indices).symmetric_difference(set(test_indices))
        train_indices = np.array(list(train_indices)).astype(int)
        yield train_indices, test_indices

###################### 

def upsample_data(data_lab_train, data_amb_train):
    
    ros = RandomOverSampler(sampling_strategy='minority',random_state=42)
    
    data_train = data_lab_train.append(data_amb_train)
    label_train = data_train['Label']
    del data_train['Label']
    
    # Resampling
    data_train_res, label_train_res = ros.fit_sample(data_train, label_train)
    data_train_res['Label'] = label_train_res
    data_train_res = data_train_res.reset_index(drop=True)
    
    return data_train_res

######################   
        
        
def get_X_y(data):
    wavenb = np.arange(1,2785,1).astype(str)
    X = data[wavenb]
    y = data['Concentration']
    return X, y
        
     
######################   
        
            
# Show plot of prediction vs reference + fit      
def print_result(y_ref, y_pred):
    fig = plt.figure()
    plt.scatter(y_ref, y_pred)
    plt.plot(y_ref, y_ref, color='black', linewidth=1, label='$y=x$') # line y = x 
    plt.xlabel('Reference [$\mu$g/cm²]')
    plt.ylabel('Prediction [$\mu$g/cm²]')
    plt.legend()
    plt.show()
    return fig
    

######################    
    
    
    
    