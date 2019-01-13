###### importing libraries ################################################
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import utilities as util
import pandas as pd 
import numpy as np

""" This script holds functions for feature reduction
    and selection.
"""

###### feature reduction ##################################################
def pca_reduction(dataframes, n_components = None, whiten = False, svd_solver = "auto", random_state = None, columns = None, pca_instance = None):
    """Feature Reduction using sklearns' PCA.

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to reduce.
        n_components (int): Number of components to keep.
        whiten (bool): When True, the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
        svd_solver (string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}) : The solver used in PCA.
        random_state(int): To seed the Random instance.
        columns (list str): Used to set the column names for the reduced instances. 
        pca_instance (PCA): Can pass an instance of PCA to be used instead of initializing a new instance.

    Returns:
        list of pandas' dataframes: A list of pandas' dataframes with reduced features.
        PCA: The PCA instance used in the function.
    """

    # convert dataframe to dataframes if necessary
    dfs = util.df_to_dfs(dataframes)
    
    # the list which will be returned including reduced features dataframes
    dfs_reduced = []

    # initialize PCA instance 
    if pca_instance == None:    
        pca_instance = PCA(n_components = n_components, whiten = whiten, svd_solver = svd_solver, random_state = random_state)
    
    for i in range(len(dfs)):
        # use pca to reduce features 
        dimentionality_reduction = pca_instance.fit_transform(dfs[i][dfs[i].columns[0:-1]]) 
    
        # create a new pandas dataframe with reduced features       
        tmp_reduced_df = pd.DataFrame(data = dimentionality_reduction)
        tmp_reduced_y_df = pd.concat([tmp_reduced_df, dfs[i].iloc[:,-1]], axis = 1)

        # set column names         
        if i == 0 and columns == None:
            columns = np.append(dfs[i].columns[: tmp_reduced_df.shape[1]], dfs[i].columns[-1])

        # setting column names 
        tmp_reduced_y_df.columns = columns

        # append reduced feature dfs to list
        dfs_reduced.append(tmp_reduced_y_df)

    # return dataframes with reduced features 
    return dfs_reduced, pca_instance

###### feature selection ##################################################
def chi2_scores(x, y, keys):    
    """Ranks features using slearns' Chi2.

    Args:
        x (numpy array): The instances used for RFE X.
        y (numpy array): A 1D numpy array with the respective labels for X.
        keys (list of str): A list of names of the respective features.
        
    Returns:
        dictionary: A dictionary of features sorted by rank.
        list of str: A list of column names sorted by rank.
    """

    selector = SelectKBest(chi2, k = "all").fit(x, y)
    scores = selector.scores_
    score_dictionary = dict(zip(keys, scores))
    sorted_by_value = sorted(score_dictionary.items(), key=lambda kv: kv[1], reverse=True)

    sorted_column_names = [sorted_by_value[i][0] for i in range(len(sorted_by_value))]
    return sorted_by_value, sorted_column_names

def rfe_ranking(x, y, estimator, f, keys, step = 0.1):
    """Ranks features using slearns' Recursive Feature Elimination.

    Args:
        x (numpy array): The instances used for RFE X.
        y (numpy array): A 1D numpy array with the respective labels for X.
        estimator(model which inherits from BaseEstimator, ClassifierMixin) Estimator used in RFE.
        f (int): The number of features to select. If None, half of the features are selected.
        keys (list of str): A list of names of the respective features.
        step (int or float): If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
        
    Returns:
        dictionary: A dictionary of features sorted by rank.
        list of str: A list of column names sorted by rank.
    """

    # initialize recursive feature elimination
    rfe = RFE(estimator, f, step = step)
    # fit RFE
    rfe = rfe.fit(x, y)
    # get the ranking for each feature 
    ranking = rfe.ranking_
    # create ranking dictionary 
    ranking_dictionary = dict(zip(keys, ranking))
    sorted_by_value = sorted(ranking_dictionary.items(), key=lambda kv: kv[1], reverse=True)

    # sort column name 
    sorted_column_names = [sorted_by_value[i][0] for i in range(len(sorted_by_value))]
    return sorted_by_value, sorted_column_names

###########################################################################
