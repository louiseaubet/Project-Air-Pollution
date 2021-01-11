# -*- coding: utf-8 -*-
"""some helper functions"""
import numpy as np
import csv
import pandas as pd


######################

def prep_data():
    # Input files

    DATA_FOLDER = "../data/"

    DATA_PATH_AMB_SPEC = DATA_FOLDER + "raw/IMPROVE_2011-2013_filterSubtractionV2.csv"
    DATA_PATH_AMB_CONC = DATA_FOLDER + "raw/IMPROVE_2011_2013_XRF_OC_ions_mug_cm2.csv"
    DATA_PATH_LAB_SPEC = DATA_FOLDER + "raw/matched_std_2784_baselined_filter_subtracted.csv"
    DATA_PATH_LAB_CONC = DATA_FOLDER + "raw/FG_Ruthenburg_std_responses.csv"
    
    OUTPUT_AMB = DATA_FOLDER + "processed/data_amb.csv"
    OUTPUT_LAB = DATA_FOLDER + "processed/data_lab.csv"
    
    ###################### 
    
    # Ambient samples
    
    # Import data
    file = pd.read_csv(DATA_PATH_AMB_SPEC, index_col="Unnamed: 0")
    data_amb_spec = pd.DataFrame(file)
    wavenb = data_amb_spec[['Wavenumber']] #dataframe of wavenumbers 

    file = pd.read_csv(DATA_PATH_AMB_CONC)
    data_amb_conc = pd.DataFrame(file)
    data_amb_conc = data_amb_conc[['Unnamed: 0','SO4f:Value']]
    data_amb_conc = data_amb_conc.dropna().reset_index(drop=True) # drop nan values

    MW_SO4 = 96.06
    MW_NH4SO4 = 132.14 # Molecular weights [g/mol]
    data_amb_conc['SO4f:Value'] = data_amb_conc['SO4f:Value'].multiply(MW_NH4SO4/MW_SO4)
    # Delete concentrations superior to the max of laboratory values :
    data_amb_conc = data_amb_conc.loc[data_amb_conc['SO4f:Value'] <= 130]
    data_amb_conc = data_amb_conc.rename(columns={'Unnamed: 0':'Sample', 'SO4f:Value':'(NH4)SO4'})

    # Selection of samples :
    samples_spec = pd.DataFrame(data_amb_spec.columns, columns=['Sample'])
    samples_spec = samples_spec.drop(samples_spec.index[[0]]).reset_index(drop=True)

    samples_conc = data_amb_conc[['Sample']].reset_index(drop=True)

    samples_both = pd.merge(samples_conc, samples_spec, how='inner').values.ravel()

    data_amb_spec = data_amb_spec[samples_both]
    data_amb_conc = data_amb_conc[['Sample','(NH4)SO4']].loc[data_amb_conc['Sample']\
                                                             .isin(samples_both)].reset_index(drop=True)
    
    data_amb_spec = data_amb_spec.transpose().reset_index()
    data_amb_spec = data_amb_spec.rename(columns={'index':'Sample', 'Wavenumber':'index'})
    data_amb_spec['Concentration'] = data_amb_conc['(NH4)SO4']

    # Separates samples into site and date
    new = pd.DataFrame(data_amb_spec['Sample'].str.split('_', 2).tolist())
    data_amb_spec['Site'] = new[0]
    data_amb_spec['Date'] = new[1]
    data_amb_spec = data_amb_spec.drop(columns =['Sample'])
    data_amb_spec['Label'] = 'amb'
    
    #Saving the data in file
    data_amb_spec.to_csv(OUTPUT_AMB, sep=',')
    
    ###################### 
    
    # Laboratory samples
    
    # Import data
    file = pd.read_csv(DATA_PATH_LAB_SPEC, index_col="Unnamed: 0")
    data_lab_spec = pd.DataFrame(file)
    data_lab_spec = data_lab_spec.sort_index(axis=1, ascending=True)

    file = pd.read_csv(DATA_PATH_LAB_CONC)
    data_lab_conc = pd.DataFrame(file)
    data_lab_conc['sample'] = data_lab_conc['sample'].astype(str)
    data_lab_conc = data_lab_conc[['sample','TRset','AmmNH_umole_cm2']]

    MW_NH4SO4 = 132.14 # Molecular weights [g/mol]
    data_lab_conc['AmmNH_umole_cm2'] = data_lab_conc['AmmNH_umole_cm2'].multiply(MW_NH4SO4/8)
    data_lab_conc['Site'] = 'X0'
    data_lab_conc['Sample'] = data_lab_conc['Site'] + data_lab_conc['sample']
    data_lab_conc = data_lab_conc[['Sample', 'Site', 'sample', 'TRset', 'AmmNH_umole_cm2']]
    data_lab_conc.columns = ['Sample','Site','Date','TRset','(NH4)SO4']

    # Selection of samples :
    samples_spec = pd.DataFrame(data_lab_spec.columns, columns = ['Sample'])
    samples_spec = samples_spec.drop(samples_spec.index[[0]]).reset_index(drop=True)

    samples_conc = data_lab_conc[['Sample']].reset_index(drop=True)

    samples_both = pd.merge(samples_conc, samples_spec, how='inner').values.ravel()

    data_lab_spec = data_lab_spec[samples_both]
    data_lab_conc = data_lab_conc[['Site','Date','TRset','(NH4)SO4']]\
                                                            .loc[data_lab_conc['Sample'].isin(samples_both)]
    
    data_lab_spec = data_lab_spec.transpose().reset_index()
    data_lab_spec = data_lab_spec.rename(columns={'index':'Sample', 'Wavenumber':'index'})
    
    data_lab_spec['Label'] = 'lab'
    data_lab_spec['Concentration'] = data_lab_conc['(NH4)SO4']
    data_lab_spec['TRset'] = data_lab_conc['TRset']
    data_lab_spec['Site'] = data_lab_conc['Site']
    data_lab_spec['Date'] = data_lab_conc['Date']
    data_lab_spec = data_lab_spec.drop(['Sample'], axis=1)

    # Saving the data in file
    data_lab_spec.to_csv(OUTPUT_LAB, sep=',')


###################### 

def load_data(INPUT_LAB, INPUT_AMB):
    # Import data
    file_lab = pd.read_csv(INPUT_LAB, index_col="Unnamed: 0")
    file_amb = pd.read_csv(INPUT_AMB, index_col="Unnamed: 0")

    data_lab = pd.DataFrame(file_lab)
    data_amb = pd.DataFrame(file_amb)
    
    return data_lab, data_amb

###################### 

# Separates dataset into train and test subsets using venetian blinds method
def train_test_split_venet(*arrays, test_size=0.2):
    
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
        
    Xv = arrays[0].values
    num_row = Xv.shape[0]-1
    interval = int(1/test_size)
    indices = np.linspace(1, num_row, num_row)
    test_indices = np.array(indices[0:num_row:interval], dtype=int)
    train_indices = set(indices).symmetric_difference(set(test_indices))
    train_indices = np.array(list(train_indices)).astype(int)
    X_train = pd.DataFrame(Xv[train_indices,], columns=arrays[0].columns)
    X_test = pd.DataFrame(Xv[test_indices,], columns=arrays[0].columns)
    
    if n_arrays == 2:
        yv = arrays[1].values
        y_train = pd.DataFrame(yv[train_indices])
        y_test = pd.DataFrame(yv[test_indices])
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test
    
######################  
    
def split_data(data_lab, data_amb):
    # separation into train and test sets using labels
    data_lab_train = data_lab.loc[data_lab['TRset'] == 'calibration'].drop(['TRset'], axis=1).reset_index(drop=True)
    data_lab_test = data_lab.loc[data_lab['TRset'] == 'test'].drop(['TRset'], axis=1).reset_index(drop=True)

    # separation into train and test sets using venetian blinds
    data_amb_train, data_amb_test = train_test_split_venet(data_amb, test_size=0.2)
    
    return data_lab_train, data_lab_test, data_amb_train, data_amb_test

###################### 

 








