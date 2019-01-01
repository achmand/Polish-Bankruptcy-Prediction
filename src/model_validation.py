###### importing libraries ################################################
# for cross validation
from sklearn.model_selection import KFold

###### k fold  ############################################################
def kfold_split(X, y, k = 10, shuffle = False, random_seed = None):

    """Splits data into K folds using sklearn KFold. This is used for cross validation.

    Args:
        n_splits (int): The number of folds this must be a value greater or equal to 2.
        X (numpy array): A numpy array with the features for a specific dataset.
        y (numpy array): A numpy array with the outcome/label for a specific dataset.
        shuffle (bool): Whether to shuffle the data before splitting into batches.
        random_seed (int): A seed used for random when applying Kfold. If None random is not seeded.

    Returns:
        list of numpy array: A list with numpy arrays used for training (features).
        list of numpy array: A list with numpy arrays used for training (label).
        list of numpy array: A list with numpy arrays used for testing (features).
        list of numpy array: A list with numpy arrays used for testing (label).
    """

    # initialize sklearn KFold
    kfold = KFold(n_splits = k, shuffle = shuffle, random_state = random_seed)

    # initialize lists for splits 
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # loop through the indices returned by split
    for train_i, test_i in kfold.split(X):
        X_train.append(X[train_i])
        X_test.append(X[test_i])
        y_train.append(y[train_i])
        y_test.append(y[test_i])

    # return lists with split data using kfold 
    return X_train, y_train, X_test, y_test


###########################################################################
