###### importing libraries ################################################
# for dimensionality reduction using pca
from sklearn.decomposition import PCA 
# for utility functions
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
        tmp_reduced_y_df = pd.concat([tmp_reduced_df, dfs[0].iloc[:,-1]], axis = 1)

        # set column names         
        if i == 0 and columns == None:
            columns = np.append(dfs[i].columns[: tmp_reduced_df.shape[1]], dfs[i].columns[-1])

        # setting column names 
        tmp_reduced_y_df.columns = columns

        # append reduced feature dfs to list
        dfs_reduced.append(tmp_reduced_y_df)

    # return dataframes with reduced features 
    return dfs_reduced, pca_instance

###########################################################################
