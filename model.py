# -*- coding: utf-8 -*-

"""Definition of the model"""

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
plt.rcParams.update({'font.size': 14})

from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from sys import stdout

from helpers import *


class Model:
    
    # Constructor
    def __init__(self,  data_lab_train, data_lab_test, data_amb_train, data_amb_test, model_type='PCA', venet_blind=True, nb_folds=5, transd=True, train_amb=False):
        self.type = model_type
        self.venet_blind = venet_blind
        self.nb_folds = nb_folds
        self.transductive = transd
        self.training_amb = train_amb
        
        if (self.type == 'PLS'):
            self.parameter = 50
        else:
            self.parameter = 1e-7
        
        if transd:
            self.data_basisSet = upsample_data(data_lab_train, data_amb_train)
            self.data_test = data_lab_test.append(data_amb_test).reset_index(drop=True)
        else:
            self.data_basisSet = data_lab_train
            self.data_test = data_lab_test.append(data_amb_train.append(data_amb_test)).reset_index(drop=True)
        
        if train_amb:
            self.data_train = upsample_data(data_lab_train, data_amb_train)
        else:
            self.data_train = data_lab_train
            
        
    ######################   
        
        
    def compute_RMSE(self, parameters):
    
    # """Run corresponding model for different values of the parameters, and calculate RMSE"""
    
        rmsecv = []
        rmsec = []
        rmsep = []
        
        X_basisSet, y_basisSet = get_X_y(self.data_basisSet)
        X_train, y_train = get_X_y(self.data_train)
        X_test, y_test = get_X_y(self.data_test)
        
        if (self.type == 'PLS'):
            
            for p in parameters:
                model = PLSRegression(n_components=p, scale=False, tol=1e-6, max_iter=500)
        
                # Type of cross-validation
                if self.venet_blind:
                    # Venetian blind cross-validation
                    indices_venet = build_k_indices(y_basisSet, nb_fold=self.nb_folds)
                    score_cv = np.mean(cross_val_score(model, X_basisSet, y_basisSet, cv=indices_venet, \
                                                scoring='neg_root_mean_squared_error'))
                else:
                    # "classic" k-fold cross-validation : cv = nb of folds
                    score_cv = np.mean(cross_val_score(model, X_basisSet, y_basisSet, cv=self.nb_folds, \
                                                 scoring='neg_root_mean_squared_error'))
        
                rmsecv.append(-score_cv)
        
                # Training 
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                score_c = np.sqrt(mean_squared_error(y_pred_train, y_train))
                rmsec.append(score_c)
        
                # Prediction
                y_predict = model.predict(X_test)
                score_p = np.sqrt(mean_squared_error(y_predict, y_test))
                rmsep.append(score_p)
            
            
            
        elif (self.type == 'PCA'):
            
            for p in parameters:
                steps = [('pca', PCA()), ('m', Lasso(alpha=p))]
                model = Pipeline(steps=steps)
            
                model[0].fit(X_basisSet, y_basisSet)
                scores_train = model[0].transform(X_train)
                scores_test = model[0].transform(X_test)
        
                # Type of cross-validation
                if self.venet_blind:
                    # Venetian blind cross-validation
                    indices_venet = build_k_indices(y_train, nb_fold=self.nb_folds)
                    score_cv = np.mean(cross_val_score(model, scores_train, y_train, cv=indices_venet, \
                                                scoring='neg_root_mean_squared_error'))
                else:
                    # "classic" k-fold cross-validation : cv = nb of folds
                    score_cv = np.mean(cross_val_score(model, scores_train, y_train, cv=self.nb_folds, \
                                                 scoring='neg_root_mean_squared_error'))
        
                rmsecv.append(-score_cv)
        
                # Training 
                model[1].fit(scores_train, y_train)
                y_pred_train = model[1].predict(scores_train)
                score_c = np.sqrt(mean_squared_error(y_pred_train, y_train))
                rmsec.append(score_c)
        
                # Prediction
                y_predict = model[1].predict(scores_test)
                score_p = np.sqrt(mean_squared_error(y_predict, y_test))
                rmsep.append(score_p)
            
            
        else:
            raise Exception("Sorry, only 'PLS' or 'PCA' are accepted for model_type.")
        
        self.model = model
    
        return rmsecv, rmsec, rmsep    
        
    ###################### 
    
    
    def print_optimisation(self, parameters, rmsecv, rmsec, rmsep, num_min):
    
        fig = plt.figure()
        plt.plot(parameters, np.array(rmsecv), '-+', color = 'tab:blue', label='RMSECV')
        plt.plot(parameters, np.array(rmsec), '-x', color = 'tab:green', label='RMSEC')
        plt.plot(parameters, np.array(rmsep), '-*', color = 'tab:red', label='RMSEP')
        plt.plot(parameters[num_min], np.array(rmsecv)[num_min], 'P', ms=10, mfc='red')
        if (self.type == 'PLS'):
            plt.xlabel('Number of PLS components')
            plt.yscale('log')
        else:
            plt.xlabel('Coefficient alpha of Lasso')
            plt.xscale('log')
            plt.yscale('log')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()
    
        return fig
    
    ######################    
        
    def optimise_model(self, parameters):

        # Compute RMSE
        rmsecv, rmsec, rmsep = self.compute_RMSE(parameters)
    
        # Calculate and print the position of minimum in RMSE
        num_min = np.argmin(rmsecv)
        opt_param = parameters[num_min]
        rmsecv_min = np.array(rmsecv)[num_min]
        rmsep_min = np.array(rmsep)[num_min]
    
        print("Suggested number of components : {:.3e}".format(opt_param))
        stdout.write("\n")
        print("Minimum RMSECV : {:.3e}".format(rmsecv_min))
        print("Minimum RMSEP : {:.3e}".format(rmsep_min))
    
        fig = self.print_optimisation(parameters, rmsecv, rmsec, rmsep, num_min)
        
        # Update of the parameter attribute
        self.parameter = opt_param
    
        return fig, rmsecv, rmsec, rmsep, rmsecv_min, rmsep_min, opt_param
        
    
    ###################### 
    
    
    def graph_prediction(self, y_predict):
        
        _, y_test = get_X_y(self.data_test)
        
        # Plot reference versus prediction
        fig = plt.figure()
        plt.scatter(y_test.loc[self.data_test['Label'] == 'amb'], y_predict.loc[self.data_test['Label'] == 'amb'], c='tab:blue', label='amb')
        plt.scatter(y_test.loc[self.data_test['Label'] == 'lab'], y_predict.loc[self.data_test['Label'] == 'lab'], c='tab:red', label='lab')
        plt.plot(y_test.loc[self.data_test['Label'] == 'amb'], y_test.loc[self.data_test['Label'] == 'amb'], color='black', linewidth=1, label='$y=x$')
        plt.xlabel('Reference [$\mu$g/cm²]')
        plt.ylabel('Prediction [$\mu$g/cm²]')
        plt.legend()
        plt.show()
    
        return fig

    
    ######################
    
    def print_prediction(self, parameter):
        
        y_predict, rmse_amb, rmse_lab = self.compute_prediction(parameter)
        
        print('Ambient error : {:.3e}'.format(rmse_amb))
        print('Laboratory error : {:.3e}'.format(rmse_lab))
        
        fig = self.graph_prediction(y_predict)
        
        return fig, rmse_amb, rmse_lab
        
    
    ######################
        
    def compute_prediction(self, parameter):
        
        X_basisSet, y_basisSet = get_X_y(self.data_basisSet)
        X_train, y_train = get_X_y(self.data_train)
        X_test, y_test = get_X_y(self.data_test)
        
        # Type of model
        if (self.type == 'PLS'):
            model = PLSRegression(n_components=parameter, scale=False, tol=1e-6, max_iter=500)
            
            # Training
            model.fit(X_train, y_train)

            # Prediction
            y_predict = pd.DataFrame(model.predict(X_test))
            self.y_pred = y_predict
            
        elif (self.type == 'PCA'):
            steps = [('pca', PCA()), ('m', Lasso(alpha=parameter))]
            model = Pipeline(steps=steps)
            
            # Training
            model[0].fit(X_basisSet, y_basisSet)
            scores_train = model[0].transform(X_train)
            scores_test = model[0].transform(X_test)
            model[1].fit(scores_train, y_train)

            # Prediction
            y_predict = pd.DataFrame(model[1].predict(scores_test))
            self.y_pred = y_predict
            
        else:
            raise Exception("Sorry, only 'PLS' or 'PCA' are accepted for model_type.")
    
        rmse_amb = np.sqrt(mean_squared_error(y_predict.loc[self.data_test['Label'] == 'amb'], y_test.loc[self.data_test['Label'] == 'amb']))
        rmse_lab = np.sqrt(mean_squared_error(y_predict.loc[self.data_test['Label'] == 'lab'], y_test.loc[self.data_test['Label'] == 'lab']))
    
        self.data_test['Prediction'] = y_predict
        self.model = model
    
        return y_predict, rmse_amb, rmse_lab    
        
        
        
        
        
             