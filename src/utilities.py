###### importing libraries ################################################
import pandas as pd

""" This script holds utility functions.
"""

###### utility functions ##################################################
def df_to_dfs(dataframes):
    """ Checks if the dataframe/s passed is a list if not convert it to a list of pandas's dataframes. 

    Args:
        dataframes (pandas' dataframe or list of pandas' dataframe): A single instance of pandas' dataframe or a list.

    Returns:
        list of pandas' dataframes: Returns a list of pandas' dataframes if one instance is passed or returns the list passed instead.
    """

    # initialize list to hold pandas' dataframes
    dfs = []
    # if it is not a list add to list
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes
    
    # return a list of pandas' dataframes
    return dfs
    
###########################################################################
