import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def mean_imputation(dataframes):
    """Imputes missing values found in pandas dataframe/s using sklearn mean imputation.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using mean imputation.
    """
    dfs = []
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = dataframes

    imp_mean_dfs = []
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    for i in range(len(dfs)):
        imp_mean_dfs.append(
            pd.DataFrame(
                mean_imputer.fit_transform(dfs[i]), columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))            

    return imp_mean_dfs