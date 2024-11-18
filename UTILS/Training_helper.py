import math
import torch
import numpy as np
# import gpytorch
import pandas as pd
import seaborn as sns
import os
import pickle
import shutil
import matplotlib 
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import KFold

import imblearn as imb
# print("imblearn version: ",imblearn.__version__)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import itertools

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys
sys.path.append('../')

from sklearn.model_selection import GridSearchCV

def calculate_metrics(y_true, y_pred): 
    

    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp, tn, fp, fn
    
def prediction_type(y_true, y_pred): 
    if (y_true == 0 and y_pred == 0): 
        return 'TN'
    elif (y_true == 0 and y_pred == 1): 
        return 'FP'
    elif (y_true == 1 and y_pred ==0): 
        return 'FN'
    elif (y_true == 1 and y_pred ==1): 
        return 'TP'
    else: 
        return 'error'

def specificity_score(tn, fp):
    val = (tn/(tn+fp))
    return val


def gather_all_test_results
    loops through all the files 

