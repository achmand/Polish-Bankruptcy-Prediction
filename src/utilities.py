import pandas as pd

def df_to_dfs(dataframes):
    dfs = []
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes
    return dfs