import numpy as np
import pandas as pd 
import utilities as util
import missingno as msno 

## TODO WRITE COMMENTS 

def missing_stats(dataframes):
    df_missing = util.df_to_dfs(dataframes)            
    missing_stats = np.zeros((len(df_missing), 4))
    for i in range(len(df_missing)):
        instances_no_missing = df_missing[i].dropna().shape[0]
        missing_stats[i][0] = df_missing[i].shape[0]
        missing_stats[i][1] = df_missing[i].shape[0] - instances_no_missing
        missing_stats[i][2] = instances_no_missing
        missing_stats[i][3] = round((missing_stats[i][1]/missing_stats[i][0]),4)
    
    columns = ["total_instances", "total_instances_with_missing_values", "total_instances_without_missing_values", "data_loss"]
    df_missing_stats = pd.DataFrame(data = missing_stats, columns = columns)
    return df_missing_stats

def imbalanced_stats(dataframes, outcome):
    df_imbalanced = util.df_to_dfs(dataframes)
    grouping = np.unique(df_imbalanced[0]["outcome"], return_counts = False)
    total_classes = len(grouping)
    imbalanced_stats = np.zeros((len(df_imbalanced), 2 + total_classes))
    
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
    
    for x in range(total_classes):
        columns = ["label_" + str(grouping[(total_classes - 1) - x])] + columns
    
    df_imbalanced_stats = pd.DataFrame(data = imbalanced_stats, columns = columns)
    return df_imbalanced_stats

def nullity_matrix(dataframes, figsize = (20,5), include_all = False):
    dfs = util.df_to_dfs(dataframes) 
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna().any()].tolist()]
        msno.matrix(tmp_df, labels = True, figsize = figsize)
    
def nullity_heatmap(dataframes, figsize = (20,20), include_all = False):
    dfs = util.df_to_dfs(dataframes) 
    for i in range(len(dataframes)):
        tmp_df = dfs[i] if include_all == True else dfs[i][dfs[i].columns[dfs[i].isna().any()].tolist()]
        msno.heatmap(tmp_df, labels = True, figsize = figsize)