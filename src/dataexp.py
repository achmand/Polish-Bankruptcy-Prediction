###### importing libraries ################################################
import numpy as np
import pandas as pd 
import utilities as util
import missingno as msno 

""" This script holds functions which output stats
    and different plots about the features. 
    This is used for feature exploration.
"""

###### statistics #########################################################
def missing_stats(dataframes):
    """ Output statistics on missing values. The following stats are shown;
    total instances, total instances with missing values, total instances without missing values 
    and the data loss percentage if the values with missing values were to be removed.  

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to explore.

    Returns:
        pandas' dataframe: A pandas dataframe with missing values stats for each dataframe passed. 
    """

    # convert to pandas' list if required
    df_missing = util.df_to_dfs(dataframes)
    # initialize array to hold stats            
    missing_stats = np.zeros((len(df_missing), 4))
    # loop and calculate statistics for each dataframe passed
    for i in range(len(df_missing)):
        instances_no_missing = df_missing[i].dropna().shape[0]
        missing_stats[i][0] = df_missing[i].shape[0]
        missing_stats[i][1] = df_missing[i].shape[0] - instances_no_missing
        missing_stats[i][2] = instances_no_missing
        missing_stats[i][3] = round((missing_stats[i][1]/missing_stats[i][0]),4)
    
    # create new pandas' dataframe which holds these stats
    columns = ["total_instances", "total_instances_with_missing_values", "total_instances_without_missing_values", "data_loss"]
    df_missing_stats = pd.DataFrame(data = missing_stats, columns = columns)
    
    # return missing values stats as a pandas' dataframe
    return df_missing_stats

def imbalanced_stats(dataframes, outcome):
    """ Output statistics on imbalanced datasets. The following stats are shown;
    the total count of each label, minority class, minority class percentage

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to explore.
    
    Returns:
        pandas' dataframe: A pandas dataframe with imbalanced labels stats for each dataframe passed. 
    """

    # convert to pandas' list if required
    df_imbalanced = util.df_to_dfs(dataframes)
    # group by labels
    grouping = np.unique(df_imbalanced[0]["outcome"], return_counts = False)
    # count the different labels
    total_classes = len(grouping)
    # initialize array to hold stats            
    imbalanced_stats = np.zeros((len(df_imbalanced), 2 + total_classes))
    # loop and calculate statistics for each dataframe passed
    for i in range(len(df_imbalanced)):
        group, occurrences = np.unique(df_imbalanced[i]["outcome"], return_counts = True)
        outcomes = dict(zip(group, occurrences))
        total_outcomes = 0
        for x in range(total_classes):
            imbalanced_stats[i][x] = outcomes[x]
            total_outcomes += outcomes[x]
        
        imbalanced_stats[i][0 + total_classes] = min(outcomes, key=outcomes.get)
        imbalanced_stats[i][1 + total_classes] = min(outcomes.values()) / total_outcomes

    columns = ["minortiy_label" , "minority_percentage"]
    
    # add label columuns (for each unique label)
    for x in range(total_classes):
        columns = ["label_" + str(grouping[(total_classes - 1) - x])] + columns
    
    # create new pandas' dataframe which holds these stats
    df_imbalanced_stats = pd.DataFrame(data = imbalanced_stats, columns = columns)
    
    # return imbalanced label stats as a pandas' dataframe
    return df_imbalanced_stats

###### plots ##############################################################
def nullity_matrix(dataframes, figsize = (20,5), include_all = False):
    """ Plots the nullity matrix of the missinggo library for the datasets.

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to plot.
        figsize (tuple(int,int)): The size of the plot
        include_all (bool): if true show all features if false shows only features with missing values.
    """

    # convert to pandas' list if required
    dfs = util.df_to_dfs(dataframes) 
    # loop and plot the nullity matrix for each dataframe passed
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna().any()].tolist()]
        msno.matrix(tmp_df, labels = True, figsize = figsize)
    
def nullity_heatmap(dataframes, figsize = (20,20), include_all = False):
    """ Plots the nullity heatmap of the missinggo library for the datasets.

    Args:
        dataframes (pandas' dataframe or a list of pandas' dataframes): The instances or a list of different instance to plot.
        figsize (tuple(int,int)): The size of the plot
        include_all (bool): if true show all features if false shows only features with missing values.
    """

    # convert to pandas' list if required
    dfs = util.df_to_dfs(dataframes) 
    # loop and plot the nullity heatmap for each dataframe passed
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna().any()].tolist()]
        msno.heatmap(tmp_df, labels = True, figsize = figsize)

###########################################################################
