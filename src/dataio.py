import os
import sys
import arff as ar 
import pandas as pd
import utilities as util
from scipy.io import arff

def file_paths (path, extensions):
    """Get file names for specific extensions in the specified path.

    Args:
        path (str): The path to search for files with the specified extension/s.
        extensions (str or tuple of str): The extensions to look for in the specified path.

    Returns:
        list of str: A list of a file paths with the specified extension.
    """

    dirs = os.listdir(path)
    dirs.sort()
    dirs_list = []
    for i in range(len(dirs)):
        tmp_dir = dirs[i]
        if(tmp_dir.lower().endswith(extensions)):
            dirs_list.append(path + dirs[i])
    return dirs_list

def arff2df (paths, include_path = False):    
    """Convert .arff files to dataframes.

    Args:
        paths (list of str): The path to the .arff files to convert into pandas dataframes. 
        include_path (bool, optional): If set to True the list returned will include the path from where the dataframe was loaded.
                        Defaults to False.

    Returns:
        list of pandas dataframe: A list of dataframes converted from the specified .arff files.
    """
    df_list = []    
    for i in range(len(paths)):
        if(include_path == True):
            df_list.append((paths[i] , pd.DataFrame(arff.loadarff(paths[i])[0]))) 
        else:
            df_list.append(pd.DataFrame(arff.loadarff(paths[i])[0])) 
    return df_list

def df2arff_function(path, filenames, function, *args):
    create_dir(path)
    files_exist = [f for f in filenames if os.path.isfile(path + f + ".arff")]
    if(len(files_exist) != len(filenames)):
        dfs = function(*args) 
        df2arff(dfs, path, filenames)

# does not take care of race conditions... 
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# TODO -> Also accept single file name (str) not only list of str...
def df2arff(dataframes, path, file_names):
    dfs = util.df_to_dfs(dataframes)
    for i in range(len(dfs)):
        ar.dump(path + file_names[i] + '.arff', dfs[i].values, names=dfs[i].columns)
        