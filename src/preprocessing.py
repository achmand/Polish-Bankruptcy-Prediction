###### importing libraries ################################################
import numpy as np
import pandas as pd
import utilities as util
import impyute as impy 
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

###### data imputation ####################################################
def __sklearn_imputation(dataframes, strategy):
    dfs = util.df_to_dfs(dataframes)
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
    dfs = util.df_to_dfs(dataframes)
    imp_em_dfs = []
    for i in range(len(dfs)):
        tmp_em_df = impy.imputation.cs.em(dfs[i].values, loops = loops, dtype = dtype) 
        imp_em_dfs.append(
            pd.DataFrame(
                tmp_em_df, columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))

    return imp_em_dfs

def mice_imputation(dataframes, dtype):
    """Imputes missing values found in pandas dataframe/s using impyute expectation maximization.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.
        dtype (str(‘int’,’float’)): Type of data.
        loops (int, optional): Number of expectation maximization iterations to run before breaking.  Defaults to 50.

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using expectation maximization.
    """
    dfs = util.df_to_dfs(dataframes)
    imp_mice_dfs = []
    for i in range(len(dfs)):
        tmp_mice_df = impy.imputation.cs.mice(dfs[i].values, dtype = dtype) 
        imp_mice_dfs.append(
            pd.DataFrame(
                tmp_mice_df, columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))

    return imp_mice_dfs

def knn_imputation(dataframes, dtype, k = 100):
    """Imputes missing values found in pandas dataframe/s using impyute knn.

    Args:
        dataframes (pandas dataframe or list of dataframes): The dataframe/s to impute missing values for.
        dtype (str(‘int’,’float’)): Type of data.
        k (int): Number of neighbours used in KNN. 

    Returns:
        list of pandas dataframe: A list of pandas dataframe imputted using knn.
    """
    dfs = util.df_to_dfs(dataframes)
    imp_knn_dfs = []
    for i in range(len(dfs)):
        tmp_knn_df = impy.imputation.cs.fast_knn(dfs[i].values, k = k, dtype = dtype) 
        imp_knn_dfs.append(
            pd.DataFrame(
                tmp_knn_df, columns = dfs[i].columns
                ).astype(dfs[i].dtypes.to_dict()))

    return imp_knn_dfs

###### over sampling  ####################################################
def oversample_smote(dataframes, sampling_strategy = "auto", random_state = 40, k = 8, columns = None, verbose = False):
    
    # convert df to dataframes
    dfs = util.df_to_dfs(dataframes)
    # initialize smote object
    smote = SMOTE(sampling_strategy = sampling_strategy, random_state = random_state, k_neighbors = k)
    
    # loop in each dataframe 
    oversampled_dfs = []
    for i in range(len(dfs)):
        n = dfs[i].shape[1] - 1
        
        # get the features for the df
        x = dfs[i].iloc[:,0:n] 
        # get the lables for the df 
        y = dfs[i].iloc[:,n]
        
        # output log (original)
        if(verbose):
            group, occurrences = np.unique(y, return_counts = True)
            outcomes = dict(zip(group, occurrences))
            print("original dataset (labels): " + str(outcomes))
            print("total: " + str(sum(outcomes.values())))
        
        # apply smote 
        x_resampled, y_resampled = smote.fit_sample(x,y)
             
        # output log (oversampled)
        if(verbose):
            group, occurrences = np.unique(y_resampled, return_counts = True)
            outcomes = dict(zip(group, occurrences))
            print("resampled dataset (labels): " + str(outcomes))
            print("total: " + str(sum(outcomes.values())) + "\n")
        
        # convert oversampled arrays back to dataframes
        oversampled_instances = np.concatenate((x_resampled, np.matrix(y_resampled).T), axis=1)
        oversampled_df = pd.DataFrame(data = oversampled_instances, columns = columns)
        oversampled_df.iloc[:,n] = oversampled_df.iloc[:,n].astype(int)
        oversampled_dfs.append(oversampled_df)
        
    # return oversampled dataframes
    return oversampled_dfs
                                     
###### re-scaling data  ##################################################
def scale_range(x, min, max):
    return np.interp(x, (x.min(), x.max()) , (min, max))

def standardization(x):
    """Scales values in array using standardization and replaces the values by their Z scores (x - x_mean / std). 
    This technique redistributes the array with mean = 0 and STD = 1. 

    Args:
        x (numpy array): A 1D numpy numeric array which will be scaled using standardization.

    Returns:
        numpy array: A 1D numpy numeric array scaled using standardization.
    """

    return ((x - np.mean(x)) / np.std(x))
##########################################################################