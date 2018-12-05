import numpy as np
import pandas as pd 

def missing_stats(dataframes):
    df_missing = []
    if isinstance(dataframes, pd.DataFrame):
        df_missing = [dataframes]
    else:
        df_missing = dataframes
            
    missing_values_stats = np.zeros((len(df_missing), 4))
    for i in range(len(df_missing)):
        instances_no_missing = df_missing[i].dropna().shape[0]
        missing_values_stats[i][0] = df_missing[i].shape[0]
        missing_values_stats[i][1] = df_missing[i].shape[0] - instances_no_missing
        missing_values_stats[i][2] = instances_no_missing
        missing_values_stats[i][3] = round((missing_values_stats[i][1]/missing_values_stats[i][0]),4)
    
    columns = ["total_instances", "total_instances_with_missing_values", "total_instances_without_missing_values", "data_loss"]
    df_missing_values = pd.DataFrame(data = missing_values_stats, columns = columns)
    return df_missing_values