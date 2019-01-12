###### importing libraries ################################################
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import utilities as util
import pandas as pd 
import numpy as np

###### feature reduction ##################################################
def pca_reduction(dataframes, n_components = None, whiten = False, svd_solver = "auto", random_state = None, columns = None, pca_instance = None):
    
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
    selector = SelectKBest(chi2, k = "all").fit(x, y)
    scores = selector.scores_
    score_dictionary = dict(zip(keys, scores))
    sorted_by_value = sorted(score_dictionary.items(), key=lambda kv: kv[1], reverse=True)

    sorted_column_names = [sorted_by_value[i][0] for i in range(len(sorted_by_value))]
    return sorted_by_value, sorted_column_names

def rfe_ranking(x, y, estimator, f, keys, step = 0.1):

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
