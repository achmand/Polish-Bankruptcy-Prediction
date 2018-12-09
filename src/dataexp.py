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