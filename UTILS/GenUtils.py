import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil
import matplotlib 

import sklearn
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)


# sys.path.append('/Users/radhi/Desktop/GitHub/atom2024/atom2024/notebooks/')
from functools import wraps
from time import time
from imblearn.over_sampling import SMOTEN, ADASYN, SMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score,precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix,matthews_corrcoef

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import KFold

import imblearn as imb


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import itertools

# from scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV


# def calculate_metrics(y_true, y_pred):
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     tn = np.sum((y_true == 0) & (y_pred == 0))
#     fp = np.sum((y_true == 0) & (y_pred == 1))
#     fn = np.sum((y_true == 1) & (y_pred == 0))
#     return tp, tn, fp, fn


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r  took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap



def calculate_metrics(y_true, y_pred): 
    
    # tp = np.sum((y_true == 1) & (y_pred == 1))
    # tn = np.sum((y_true == 0) & (y_pred == 0))
    # fp = np.sum((y_true == 0) & (y_pred == 1))
    # fn = np.sum((y_true == 1) & (y_pred == 0))
    # return tp, tn, fp, fn
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    y_pred = pd.Series(y_pred) if not isinstance(y_pred, pd.Series) else y_pred
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp, tn, fp, fn
def specificity_score(tn, fp):
    val = (tn/(tn+fp))
    return val

def make_torch_tens_float(filepath=None, filename=None, rootname=None, df=None,full_path=None):
    
    if filepath is not None:
        trainX_df = pd.read_csv(filepath + filename + "_trainX.csv")
        trainy_df = pd.read_csv(filepath + filename + "_train_y.csv")
        testX_df = pd.read_csv(filepath + filename + "_testX.csv")
        testy_df = pd.read_csv(filepath + filename + "_test_y.csv")
    if rootname is not None:
        df = pd.read_csv(
            f"{filepath}{rootname}.csv"
        )  # NEK2_binding_MOE_none_scaled.csv
        train = df[df["subset"] == "train"]
        test = df[df["subset"] == "test"]
        drop_cols = ["NEK", "subset", "active", "base_rdkit_smiles", "compound_id"]
        if "fold" in df.columns:
            drop_cols.append("fold")
        trainX_df = train.drop(columns=drop_cols)
        trainy_df = train["active"]
        testX_df = test.drop(columns=drop_cols)
        testy_df = test["active"]
    if df is not None:
        train = df[df["subset"] == "train"]
        test = df[df["subset"] == "test"]
        drop_cols = ["NEK", "subset", "active", "base_rdkit_smiles", "compound_id"]
        if "fold" in df.columns:
            drop_cols.append("fold")
        trainX_df = train.drop(columns=drop_cols)
        trainy_df = train["active"]
        testX_df = test.drop(columns=drop_cols)
        testy_df = test["active"]

    train_x_temp = trainX_df.to_numpy().astype("double")  # double
    test_x_temp = testX_df.to_numpy().astype("double")  # double

    train_y_temp = trainy_df.to_numpy().flatten().astype("double")  # double
    test_y_temp = testy_df.to_numpy().flatten().astype("double")  # double

    trainX = torch.as_tensor(train_x_temp, dtype=torch.float32)
    trainy = torch.as_tensor(train_y_temp, dtype=torch.float32)
    testX = torch.as_tensor(test_x_temp, dtype=torch.float32)
    testy = torch.as_tensor(test_y_temp, dtype=torch.float32)
    return trainX, trainy, testX, testy

def make_torch_tens_float_simple(file_path_X=None,file_path_y=None, df_path=None):
    if df_path is not None: 
        df = pd.read_csv(df_path)
    # X_df = pd.read_csv(file_path_X)
    # y_df = pd.read_csv(file_path_y)

    drop_cols = ["NEK", "subset", "active", "base_rdkit_smiles", "compound_id"]
    if "fold" in df.columns:
        drop_cols.append("fold")
    # X_df = df.drop(columns=drop_cols)
    # y_df = df["active"]
     
    # x_temp = X_df.to_numpy().astype("double")  # double
    # y_temp =y_df.to_numpy().flatten().astype("double")  # double

    # X = torch.as_tensor(x_temp, dtype=torch.float32)
    # y = torch.as_tensor(y_temp, dtype=torch.float32)

    # return X,y
    X = df.drop(columns=drop_cols).to_numpy().astype("float32")
    y = df['active'].to_numpy().astype("float32")

    testX = torch.from_numpy(X)
    testy = torch.from_numpy(y)

    return testX, testy


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
