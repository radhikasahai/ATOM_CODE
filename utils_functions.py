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

# from scipy.stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import sys
sys.path.append('../')
# import utils
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, recall_score, PrecisionRecallDisplay

# Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=18)
    fig.tight_layout();
    # return ax

# Plot Heatmap
def plot_heatmap(data, title="Heatmap", xlabel="X-axis", ylabel="Y-axis"):
    """
    Plots a heatmap for the given data.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt="d", cmap='BuPu')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Plot t-SNE
def plot_tsne(features, labels, title='t-SNE'):
    """
    Plots a t-SNE visualization for the given features and labels.
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=sns.color_palette("hsv", 10),
        legend="full",
        alpha=0.3
    )
    plt.title(title)

# Plot Histogram
def plot_hist(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram for the given data.
    """
    plt.hist(data, bins=bins, alpha=0.6, color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Plot Histogram for True Positives and True Negatives
def plot_hist_tp_tn(y_true, y_pred_prob, threshold=0.5):
    """
    Plots histograms for true positives and true negatives based on a given threshold.
    """
    tp = y_pred_prob[(y_true == 1) & (y_pred_prob >= threshold)]
    tn = y_pred_prob[(y_true == 0) & (y_pred_prob < threshold)]

    plt.hist(tp, bins=30, alpha=0.5, label='True Positives', color='green')
    plt.hist(tn, bins=30, alpha=0.5, label='True Negatives', color='red')
    plt.legend(loc='upper right')
    plt.show()

# Plot Confusion Matrix Distribution
def plot_cm_dist(y_true, y_pred, title="Confusion Matrix Distribution"):
    """
    Plots distributions for each quadrant of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot KDE for Confusion Matrix Distribution
def plot_cm_dist_kde(y_true, y_pred_prob, title="KDE for Confusion Matrix Distribution"):
    """
    Plots Kernel Density Estimation (KDE) for true positives and false positives.
    """
    sns.kdeplot(y_pred_prob[y_true == 1], label='True Positives', fill=True)
    sns.kdeplot(y_pred_prob[y_true == 0], label='False Positives', fill=True)
    plt.title(title)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_class_and_probability_grids(y_true, probabilities, title_prefix=''):
    """
    Plots grids for actual classes and prediction probabilities side by side.

    Parameters:
    - y_true: Actual class labels (numpy array).
    - probabilities: Prediction probabilities (assumed to be a tensor, requires .numpy() method).
    - title_prefix: Optional prefix for the plot titles.
    """
    # Ensure `y_true` is a numpy array and convert `probabilities` to numpy array
    actual_classes = np.array(y_true)
    probabilities_np = probabilities.numpy() if hasattr(probabilities, 'numpy') else probabilities

    # Calculate the side length of the grid for square arrangement
    num_samples = actual_classes.shape[0]
    side_length = int(np.ceil(np.sqrt(num_samples)))

    # Prepare actual class grid data
    actual_grid_data = np.full((side_length, side_length), np.nan)
    actual_grid_data.flat[:num_samples] = actual_classes

    # Assume probabilities_np is structured with probabilities in the second axis
    active_probabilities = probabilities_np[:, 1] if probabilities_np.ndim > 1 else probabilities_np
    prediction_grid_data = np.full((side_length, side_length), np.nan)
    prediction_grid_data.flat[:num_samples] = active_probabilities

    # Plotting both grids side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})

    # Plot actual class grid
    im0 = axs[0].imshow(actual_grid_data, cmap='RdYlGn', origin='lower', aspect='equal')
    fig.colorbar(im0, ax=axs[0], ticks=[0, 1], label='Actual Class')
    axs[0].set_title(f'{title_prefix}Actual Class Grid')
    axs[0].axis('off')

    # Plot prediction probabilities grid
    im1 = axs[1].imshow(prediction_grid_data, cmap='RdYlGn', origin='lower', aspect='equal')
    fig.colorbar(im1, ax=axs[1], label='Probability of Active Class')
    axs[1].set_title(f'{title_prefix}Prediction Probabilities Grid')
    axs[1].axis('off')

    plt.show()

def plot_kde(observed_pred, title): 
    plt.figure(figsize=(8, 6))
    var = observed_pred.variance.numpy().tolist()
    class0_var = observed_pred.variance[0].numpy() 
    class1_var = observed_pred.variance[1].numpy() 
    
    sns.kdeplot(class0_var, label=f'Class 0')
    sns.kdeplot(class1_var, label=f'Class 1')
    plt.xlabel('Variance')
    plt.ylabel('Density')
    plt.title(f'{title} KDE Variances for Each Class')
    plt.legend()
    plt.grid(True)
    plt.show();

def look_at_data(filepath):
    """5-fold on majority and minority separately, then concat into one df""" 
    df = pd.read_csv(filepath)

    print("Dataset shape:",df.shape)
    print(df.active.value_counts())
    print(df['fold'].unique())
    num_gap = (df.loc[df['active']==0].shape[0]) - (df.loc[df['active']==1].shape[0])
    print("\nDifference in class sample sizes: ",num_gap)

    num_minority = df.loc[df['active']==1].shape[0]
    print("Number of minority samples: ",num_minority)
    # print(df.describe())
    print(f"active/inactive: {df['active'].value_counts()}")
    print(f"active/inactive: {df['active'].value_counts()}")
    counts_per_fold = df.groupby('fold')['active'].value_counts()
    print(counts_per_fold)
    return df


def plot_cm_dist_kdedensity(observed_pred, predictions, true_labels, title, max_yaxis): 
    """Plot KDE density plot for each classification on CM: TP, FP, TN, FP
    observed_pred: likelihood, comes from likelihood(model(input))
    predictions: class 0 or 1 predicted label, comes from model(input).loc.max(0)[1]
    true_labels: 0 or 1 true labels 
    title (str): plot title
    max_yaxis: max density (so all subplots on same y axis)
    """

    true_labels = true_labels.numpy()
    
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()
    
    # max_var = max(var_tp.max(), var_tn.max(), var_fp.max(), var_fn.max())
    # min_var = min(var_tp.min(), var_tn.min(), var_fp.min(), var_fn.min())
    max_y_lim = max_yaxis
    plt.figure(figsize=(10, 10))
    # to add same scale
    # bins = np.linspace(0, max(max(var_tp), max(var_tn), max(var_fp), max(var_fn)), 50)
    # bins = np.linspace(min_var, max_var, 50)
    plt.subplot(2, 2, 4)
    sns.histplot(var_tp, kde=True,color='green', bins=10, stat='density')
    plt.title('True Positives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 1)
    sns.histplot(var_tn, kde=True,color='blue', bins=10, stat='density')
    plt.title('True Negatives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 2)
    sns.histplot(var_fp, kde=True,color='red', bins=10, stat='density')
    plt.title('False Positives',fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)

    plt.subplot(2, 2, 3)
    sns.histplot(var_fn, kde=True, color='orange', bins=10, stat='density')
    plt.title('False Negative', fontsize=12)
    plt.xlabel('Variance')
    plt.ylim(0, max_y_lim)
    
    plt.tight_layout()
    plt.suptitle(f'{title}', fontsize=16, y=1.05)
    plt.show();

def plot_prob_hist(probabilities, y_labels, title, bind_inhib): 
    """Histogram of prediction probabilities
    probabilities (tensor): sample from output distribution, and transform to probabilities
    y_labels: true labels 
    title: plot title
    bind_inhib (str): binding or inhibition for x axis label"""
    fig_width = 10
    fig_height = 8
    
    idx_1 = np.where(y_labels == 1)[0]
    idx_0 = np.where(y_labels == 0)[0]
    # Histogram predictions without error bars:
    fig, ax = plt.subplots(1,figsize=(fig_width, fig_height))
    ax.hist(probabilities.numpy()[1,][idx_1], histtype='step', linewidth=3, label='Binding')
    ax.hist(probabilities.numpy()[1,][idx_0], histtype='step', linewidth=3, label='No binding')
    ax.set_xlabel(f'Prediction ({bind_inhib} probability)')
    ax.set_ylabel('Number of compounds (in log scale)')
    plt.title(title, fontsize=24)
    plt.legend(fontsize=18)
    plt.yscale('log')
    plt.grid(True)
    plt.show(); 

def plot_swarmplot(predictions, true_labels, observed_pred, title):
    true_labels = true_labels.numpy()
    
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()

    data = {
        'Variance': np.concatenate([var_tp, var_tn, var_fp, var_fn]),
        'Category': ['TP'] * len(var_tp) + ['TN'] * len(var_tn) + ['FP'] * len(var_fp) + ['FN'] * len(var_fn)
    }

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Variance', data=df)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Variance')
    plt.show();


def probabilities_vs_var(true_labels, probabilities, observed_pred,title, bind_inhib):
    """Scatter plot of probabilities vs variance
    probabilities: extracted from samples
    """
    idx_1 = np.where(true_labels == 1)[0]
    idx_0 = np.where(true_labels == 0)[0]
    fig_width = 10
    fig_height = 8
    fig, ax = plt.subplots(1,figsize=(fig_width, fig_height))
    ax.scatter(probabilities.numpy()[1,][idx_1],
               observed_pred.variance.numpy()[1,][idx_1],
               label=bind_inhib, marker='^', s=80, alpha=0.75)

    ax.scatter(probabilities.numpy()[1,][idx_0],
               observed_pred.variance.numpy()[1,][idx_0],
               label=f'No {bind_inhib}', marker='o', s=80, alpha=0.75)
    
    ax.set_xlabel(f'Prediction ({bind_inhib} probability)')
    ax.set_ylabel(f'{bind_inhib} variance')
    plt.title(title, fontsize=24)
    plt.legend(fontsize=18)
    
    plt.show();


def swarm_prob(model, x_input, true_labels, title):
    """Swarm plot of probabilities (I used it for the rf models)
    model: rf model
    x_input: x labels 
    true_labels: matching y labels"""
    predictions = model.predict(x_input)
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    prob = model.predict_proba(x_input)
    a = prob[true_pos, 1]
    b = prob[true_neg, 0]
    c = prob[false_pos, 1]
    d = prob[false_neg, 0]
    data = {
        'Probability': np.concatenate([a,b,c,d]),
        'Category': ['TP'] * len(a) + ['TN'] * len(b) + ['FP'] * len(c) + ['FN'] * len(d)
    }

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Probability', data=df)
    plt.title(title)
    plt.xlabel('Classification Type')
    plt.ylabel('Probability')
    plt.show();
    
    

def plot_prec_recall(true_labels, probabilities_class1, title):
    precision, recall, thresholds = precision_recall_curve(true_labels, probabilities_class1)
    plt.figure(figsize=(8,6))
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.title(title)
    plt.show();



def swarm_by_var_and_prob(predictions, true_labels, observed_pred, probabilities, title):
    true_labels = true_labels.numpy()
   
    true_pos = np.where((predictions == 1) & (true_labels == 1))[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0))[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0))[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1))[0] 

    var_tp = observed_pred.variance[1, true_pos].numpy()
    var_tn = observed_pred.variance[0, true_neg].numpy()
    var_fp = observed_pred.variance[1, false_pos].numpy()
    var_fn = observed_pred.variance[0, false_neg].numpy()
    prob_class0 = probabilities.numpy()[0,]
    prob_class1 = probabilities.numpy()[1,]
    prob_tp = probabilities.numpy()[1,][true_pos]
    prob_tn = probabilities.numpy()[0,][true_neg]
    prob_fp = probabilities.numpy()[1,][false_pos]
    prob_fn = probabilities.numpy()[0,][false_neg]
    
    
    data = {
        'Variance': np.concatenate([var_tp, var_tn, var_fp, var_fn]),
        'Probability Class 0 or Class 1': np.concatenate([prob_tp, prob_tn, prob_fp, prob_fn]),
        'Category': ['TP'] * len(var_tp) + ['TN'] * len(var_tn) + ['FP'] * len(var_fp) + ['FN'] * len(var_fn)
    }


    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='Variance', data=df,hue='Probability Class 0 or Class 1')
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Variance')
    plt.show();
        
    
def swarm_stdprob_RF(predictions, x_input, true_labels, std1, std0, title=None):
    # predictions = model.predict(x_input)
    true_pos = np.where((predictions == 1) & (true_labels == 1).flatten())[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0).flatten())[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0).flatten())[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1).flatten())[0] 
    # std0 = test_proba0_df.std(axis=1)
    # std1 = test_proba1_df.std(axis=1)
    a = std1[true_pos]
    b = std0[true_neg]
    c = std1[false_pos]
    d = std1[false_neg]
    data = {
        'std of probabilities': np.concatenate([a,b,c,d]),
        'Category': ['TP'] * len(a) + ['TN'] * len(b) + ['FP'] * len(c) + ['FN'] * len(d)
    }
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Category', y='std of probabilities', data=df)
    
    plt.title(title)
    plt.xlabel('Classification Type')
    plt.ylabel('std of tree probability')
    plt.show();

def get_swarm_data(predictions, true_labels, std1, std0, set_name):
    true_pos = np.where((predictions == 1) & (true_labels == 1).flatten())[0] 
    true_neg = np.where((predictions == 0) & (true_labels == 0).flatten())[0]
    false_pos = np.where((predictions == 1) & (true_labels == 0).flatten())[0] 
    false_neg = np.where((predictions == 0) & (true_labels == 1).flatten())[0] 
    a = std1[true_pos]
    b = std0[true_neg]
    c = std1[false_pos]
    d = std1[false_neg]

    
    data = {
        'std of probabilities': np.concatenate([a,b,c,d]),
        'Category': ['TP'] * len(a) + ['TN'] * len(b) + ['FP'] * len(c) + ['FN'] * len(d),
        'Set': [set_name] * (len(a) + len(b) + len(c) + len(d))
    }
    return pd.DataFrame(data)
def plot_swarm_std_prob_RF(df, figure_path, title):
    "for one plot at a time"

    plt.figure(figsize=(10, 6))
    category_order = ['TP', 'TN', 'FP', 'FN']
    color={'moe':'mediumpurple' ,'mfp':'paleredviolet'}
    color_palette2 = sns.color_palette('Accent',n_colors=2)
    sns.swarmplot(x='Category', y='std of probabilities', hue='Set', data=df,dodge=True,palette=color_palette2,order=category_order)
    plt.ylim(0,0.5)
    plt.xlabel(category_order)
    plt.title(title)
    plt.xlabel('Classification Type')
    
    plt.ylabel('std of tree probability')
    # plt.legend(title)
    plt.savefig(f'{figure_path}{title}.png')
    plt.show();
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

def rf_results(model, x_input, y_labels): 
    """Make predictions adn get probabilities
    @params
    model: fitted model (fitted to train set)
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays)
    @returns
    train/test predictions
    train/test accuracies 
    train/test probabilities"""
    pred = model.predict(x_input)
    tp, tn, fp, fn = calculate_metrics(y_labels, pred)
    acc = accuracy_score(y_labels, pred)
    precision = precision_score(y_labels, pred)
    recall = recall_score(y_labels, pred)
    specificity = specificity_score(tn, fp)
    prob = model.predict_proba(x_input)

    # print(f'accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}')
    return pred, acc, precision, recall, specificity, prob

def rf_models(train_x, train_y, test_x, test_y, rf_type, parameters):
    """Fit a RF model, make predictions, get probabilities
    @params: 
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays) 
    rf_type: model type: RandomForestClassifier, RandomForestClassifier with class_weight:'balanced', or BalancedRandomForestClassifier
        default is RFC 
    parameters: dict for model params 
    dataset_type: binding or inhibition
    @returns: dict with model, train/test prections and probabilities
    """
    n_estimators = parameters.get('n_estimators', 100)
    random_state = parameters.get('random_state', 42) 
    criterion = parameters.get('criterion', 'gini')
    max_depth = parameters.get('max_depth', 100)
    min_samples_split = parameters.get('min_samples_split', 2) 
    min_samples_leaf = parameters.get('min_samples_leaf', 1) 
    bootstrap = parameters.get('bootstrap', False) 
    max_features = parameters.get('max_features', None) 
    class_weight = parameters.get('class_weight', None)
    # RF, RF_BCW, BRFC, BRFC_BCW
    # if (rf_type == 'balanced class_weight'): 
    #     model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
    #                             , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
    # elif (rf_type == 'balanced RF'):
    #     model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
    #                             , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    # else:
    #     model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
    #                             , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    if (rf_type == 'RF'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    elif (rf_type == 'RF_BCW'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
    elif (rf_type == 'BRFC'):
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    elif (rf_type == 'BRFC_BCW'): 
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')

    else:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    
    model.fit(train_x, train_y)
   
    train_pred, train_acc, train_precision, train_recall, train_specificity, train_prob = rf_results(model, train_x, train_y)
    test_pred, test_acc, test_precision, test_recall, test_specificity, test_prob = rf_results(model, test_x, test_y)
    print(f'TRAIN: accuracy: {train_acc:.3f}, precision: {train_precision:.3f}, recall: {train_recall:.3f}, specificity: {train_specificity:.3f}')
    print(f'TEST: accuracy: {test_acc:.3f}, precision: {test_precision:.3f}, recall: {test_recall:.3f}, specificity: {test_specificity:.3f}')
    # return {'model': model, 'train_pred':train_pred, 'test_pred': test_pred,
    #          'train_prob':train_prob, 'test_prob': test_prob}
    return model 



def find_best_models(train_x, train_y, test_x, test_y, rf_type, parameters, param_dist,  verbose_val=None):
    """uses GridSearchCV not random grid search
    Grid search to find the best model, make predictions (train and test), get probability (train and test), and plot CM 
    Save best model to pickle file 
    @params:
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays) 
    rf_type: model type: RandomForestClassifier, RandomForestClassifier with class_weight:'balanced', or BalancedRandomForestClassifier
        default is RFC 
    parameters: dict for model params 
    param_dist: parameters for grid search
    dataset_type: binding or inhibition
    @returns: dict with model, train/test prections and probabilities
    """
    n_estimators = parameters.get('n_estimators', 100)
    random_state = parameters.get('random_state', None) 
    criterion = parameters.get('criterion', 'gini')
    max_depth = parameters.get('max_depth', None)
    min_samples_split = parameters.get('min_samples_split',2 ) 
    min_samples_leaf = parameters.get('min_samples_leaf', 1) 
    bootstrap = parameters.get('bootstrap', True) 
    max_features = parameters.get('max_features', None) 
    class_weight = parameters.get('class_weight', None)
    bootstrap = parameters.get('bootstrap', True)
    if (verbose_val==None): 
        verbose_val = 0
    if (rf_type == 'RF'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
        model = RandomForestClassifier()
    elif (rf_type == 'RF_BCW'): 
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')
        model = RandomForestClassifier()
    elif (rf_type == 'BRFC'):
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight=class_weight)
    elif (rf_type == 'BRFC_BCW'): 
        model = BalancedRandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split
                                , min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, max_features=max_features, class_weight='balanced')


    rand_search = GridSearchCV(estimator =model, param_grid = param_dist,cv=5, n_jobs=8, verbose=verbose_val)
    rand_search.fit(train_x, train_y) 
    best_rf = rand_search.best_estimator_

    train_pred, train_acc, train_precision, train_recall, train_specificity, train_prob = rf_results(best_rf, train_x, train_y)
    test_pred, test_acc, test_precision, test_recall, test_specificity, test_prob = rf_results(best_rf, test_x, test_y)

    # return {'best_model': best_rf, 'train_pred':train_pred, 'test_pred': test_pred, 'train_prob':train_prob, 'test_prob': test_prob}
    return rand_search


def rf_plots(train_x, train_y, test_x, test_y, max_depths, n_estimators, max_features, rf_type, parameters, dataset_type): 
    """model_resuults is the dictionary with model, predictions, etc."""
    train_aucs = []
    test_aucs = []

    for depth in max_depths:
            parameters['max_depth'] = depth
            results = rf_models(train_x, train_y, test_x, test_y, rf_type, parameters, dataset_type)
            train_auc = roc_auc_score(train_y, results['train_pred'])
            test_auc = roc_auc_score(test_y, results['test_pred'])
            train_aucs.append(train_auc)
            test_aucs.append(test_auc)

    plt.plot(max_depths, train_aucs, label='Train AUC')
    plt.plot(max_depths, test_aucs, label='Test AUC')
    plt.xlabel('Tree Depth')
    plt.ylabel('AUC Score')
    plt.title('Tree Depth vs AUC Score')
    plt.legend()
    plt.show();

    train_aucs_est = []
    test_aucs_est = []

    for estimators in n_estimators:
        parameters['n_estimators'] = estimators
        results = rf_models(train_x, train_y, test_x, test_y, rf_type, parameters, dataset_type)
        train_auc_est = roc_auc_score(train_y, results['train_pred'])
        test_auc_est = roc_auc_score(test_y, results['test_pred'])
        train_aucs_est.append(train_auc_est)
        test_aucs_est.append(test_auc_est)

    plt.plot(n_estimators, train_aucs_est, label='Train AUC')
    plt.plot(n_estimators, test_aucs_est, label='Test AUC')
    plt.xlabel('Number of Estimators')
    plt.ylabel('AUC Score')
    plt.title('Number of Estimators vs AUC Score')
    plt.legend()
    plt.show();

    train_aucs_feats = []
    test_aucs_feats = []

    for features in max_features:
        parameters['max_features'] = features
        results = rf_models(train_x, train_y, test_x, test_y, rf_type, parameters, dataset_type)
        train_aucfeats = roc_auc_score(train_y, results['train_pred'])
        test_auc_feats = roc_auc_score(test_y, results['test_pred'])
        train_aucs_feats.append(train_aucs_feats)
        test_aucs_feats.append(test_auc_feats)

    plt.plot(max_features, train_aucs_feats, label='Train AUC')
    plt.plot(max_features, test_aucs_feats, label='Test AUC')
    plt.xlabel('Max Features')
    plt.ylabel('AUC Score')
    plt.title('Max Features vs AUC Score')
    plt.legend()
    plt.show();



def rf_results2(model, train_x, train_y, test_x, test_y): 
    """Make predictions adn get probabilities
    @params
    model: fitted model (fitted to train set)
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays)
    @returns dict 
    train/test predictions
    train/test accuracies 
    train/test probabilities"""
    train_pred = model.predict(train_x) 
    test_pred = model.predict(test_x)
    train_acc = accuracy_score(train_y, train_pred) 
    test_acc = accuracy_score(test_y, test_pred) 
    
    precision_train = precision_score(train_y, train_pred)
    precision_test = precision_score(test_y, test_pred)

    recall_train = recall_score(train_y, train_pred)
    recall_test = recall_score(test_y, test_pred)

    tp_train, tn_train, fp_train, fn_train = calculate_metrics(train_y, train_pred)
    tp_test, tn_test, fp_test, fn_test = calculate_metrics(test_y, test_pred)
    sensitivity_train = tp_train / (tp_train  + fn_train)
    sensitivity_test = tp_test / (tp_test + fn_test)


    specificity_train = tn_train / (tn_train  + fp_train)
    specificity_test = tn_test / (tn_test + fp_test)

    train_prob = model.predict_proba(train_x) 
    test_prob = model.predict_proba(test_x) 

    print(f'TRAIN: accuracy: {train_acc:.3f}, precision: {precision_train:.3f}, recall: {recall_train:.3f},  specificity: {specificity_train:.3f}')
    print(f'TEST: accuracy: {test_acc:.3f}, precision: {precision_test:.3f}, recall: {recall_test:.3f}, specificity: {specificity_test:.3f}')

    

    return {'train_pred':train_pred, 'test_pred':test_pred,
            'train_acc':train_acc, 'test_acc':test_acc,
            'train_prob':train_prob, 'test_prob':test_prob, 
            'train_acc': train_acc, 'test_acc': test_acc,
            'train_prec':precision_train, 'test_prec': precision_test, 
            'train_recall': recall_train, 'test_recall': recall_test, 
            'train_sensitivity': sensitivity_train, 'test_sensitivity': sensitivity_test,
            'train_specificity': specificity_train, 'test_specificity': specificity_test}


def gather_rf_results(model, x_input, true_labels):
    """Save rf model results to DF"""
    # results = rf_results(model, x_input, true_labels)
    pred, acc, precision, recall, specificity, prob = rf_results(model, x_input, true_labels)
    # results_df = pd.DataFrame(results)
    results = {
        'prediction': pred,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'prob_class0': prob[:, 0],
        'prob_class1': prob[:, 1]
    }
    results_df = pd.DataFrame(results)
    results_df['y'] = true_labels
    tp, tn, fp, fn = calculate_metrics(true_labels, pred)
    results_df['TN'] = tn  
    results_df['FN'] = fn
    results_df['FP'] = fp 
    results_df['TP'] = tp     
    # 'TN': tn, 'FN': fn, 'FP': fp, 'TP': tp

    # results_df['prob_class0'] = model.predict_proba(x_input)[:,0] 
    # results_df['prob_class1'] = model.predict_proba(x_input)[:,1] 
    return results_df 

def get_arrays(file_path, df_filename, filename_type=None, save=False, printOut=False):
    """use dataframes to get trainX, trainy, testX, testy out. Optional: save those files to csv
    file_path: directory
    df_filename: dataframe NEK#_binding_moe_{sampling}_df.csv (sampling: scaled, UNDER, SMOTE, ADASYN)
    split dataframe to train and test, and x and y
    save: bool, option to save splits to separate csv files (train X, train y, test X, test y) 
    returns: numpy arrays train X, train y, testX, test y"""
    df = pd.read_csv(file_path+df_filename)
    train_df= df[df['subset']=='train']
    test_df = df[df['subset']=='test']
    train_y = train_df['active'].to_numpy().reshape(-1)
    test_y=test_df['active'].to_numpy().reshape(-1)
    train_x_df = train_df.drop(columns='active')

  
    test_x_df = test_df.drop(columns='active')
    
    train_x_df = train_df.drop(columns='active')
    test_x_df = test_df.drop(columns='active')
    trainX = train_x_df.select_dtypes(include='number').to_numpy()
    testX = test_x_df.select_dtypes(include='number').to_numpy()
    if (printOut): 

        print(f'train X shape: {trainX.shape}, y: {train_y.shape}, test X: {testX.shape}, y:{test_y.shape}')
    if (save and filename_type is not None): 
        trainxdf = pd.DataFrame(trainX)
        trainxdf.to_csv(file_path+filename_type+'_trainX.csv', index=False)
        # train_x_df.to_csv(filename_type+'_trainX.csv', index=False)
        trainy_df = pd.DataFrame(train_y)
        trainy_df.to_csv(file_path+filename_type+'_train_y.csv', index=False) 
        # test_x_df.to_csv(filename_type+'_testX.csv', index=False)
        testxdf = pd.DataFrame(testX)
        testxdf.to_csv(file_path+filename_type+'_testX.csv', index=False)
        testy_df = pd.DataFrame(test_y)
        testy_df.to_csv(file_path+filename_type+'_test_y.csv', index=False) 
        
    return trainX, train_y, testX, test_y