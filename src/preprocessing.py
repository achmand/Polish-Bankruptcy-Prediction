import numpy as np
import pandas as pd
import impyute as impy 
from sklearn.impute import SimpleImputer

def __sklearn_imputation(dataframes, strategy):
    dfs = []
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes

    imp_sklearn_dfs = []
    sklearn_imputer = SimpleImputer(missing_values=np.nan, strategy= strategy)
    for i in range(len(dfs)):
        imp_sklearn_dfs.append(
            pd.DataFrame(
                sklearn_imputer.fit_transform(dfs[i]), columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))            

    return imp_sklearn_dfs

def mean_imputation(dataframes):
    """Imputes missing values found in pandas dataframe/s using sklearn mean imputation.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using mean imputation.
    """
    return __sklearn_imputation(dataframes, "mean")

def median_imputation(dataframes):
    """Imputes missing values found in pandas dataframe/s using sklearn median imputation.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using median imputation.
    """
    return __sklearn_imputation(dataframes, "median")

def mode_imputation(dataframes):
    """Imputes missing values found in pandas dataframe/s using sklearn mode imputation.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using mode imputation.
    """
    return __sklearn_imputation(dataframes, "most_frequent")


def em_imputation(dataframes, dtype, loops = 50):
    """Imputes missing values found in pandas dataframe/s using impyute expectation maximization.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.
        dtype (str(‘int’,’float’)): Type of data.
        loops (int, optional): Number of expectation maximization iterations to run before breaking.  Defaults to 50.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using expectation maximization.
    """
    dfs = []
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes
    
    imp_em_dfs = []
    for i in range(len(dfs)):
        tmp_em_df = impy.imputation.cs.em(dfs[i].values, loops = loops, dtype = dtype) 
        imp_em_dfs.append(
            pd.DataFrame(
                tmp_em_df, columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))

    return imp_em_dfs