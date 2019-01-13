###### importing libraries ################################################
import numpy as np 
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score

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

###### data modeling  #####################################################
def data_modeling(classifiers, datasets, k = 10, transform_func = None, transform_axis = 0, transform_func_args = None, verbose = False):
    
    # initialize dictionary to output results for each model
    results = OrderedDict()
    
    # iterate over classifiers dictionary 
    for model_key, model_value in classifiers.items():
        
        if verbose == True:
            print("Model: {0}".format(model_key))

        # initialize dictionary for each imputation datasets result
        imputation_results = OrderedDict()
        
        # iterate over different datasets imputted using different techniques
        for dataset_key, dataset_value in datasets.items():
            
            if verbose == True:
                print("\tDataset: {0}".format(dataset_key))

            # initialize dictionary for dataset result
            dataset_results = OrderedDict()
            
            # iterate over each dataframe in datasets 
            for i in range(len(dataset_value)):
                
                if verbose == True:
                    print("\t\tDataset No: {0}".format(i))

                # get dataframe at the specified index
                tmp_dataframe = dataset_value[i]
                
                # split dataset to features and labels 
                tmp_X = tmp_dataframe.iloc[:,0:-1]
                
                # check if we should apply any transformation 
                if transform_func != None:
                    model_apply_transform = model_value[1]
                    # in this case we should apply transform on features 
                    if model_apply_transform == True:
                        tmp_X = tmp_X.apply(transform_func, axis = transform_axis, args = transform_func_args)
                
                tmp_X = tmp_X.values
                tmp_y = tmp_dataframe.iloc[:,-1].values  
                
                # get labels to be used in confusion matrix
                labels = np.unique(tmp_y) 

                # use kfold to split dataset for cross validation
                tmp_X_train, tmp_y_train, tmp_X_test, tmp_y_test = kfold_split(X = tmp_X, y = tmp_y, k = k, shuffle=True)
                
                # arrays for different metric results for each fold
                # array for results for each fold 
                # accuracy array
                folds_accuracy = np.zeros([k])
                # recall array 
                folds_recall = np.zeros([k, 2])
                # precision array
                folds_precision = np.zeros([k, 2])
                # f1 score array 
                folds_f1 = np.zeros([k, 2])
                # true negatives array
                folds_tn = np.zeros([k])
                # false postives array 
                folds_fp = np.zeros([k])
                # false negatives array 
                folds_fn = np.zeros([k])
                # true postives array 
                folds_tp = np.zeros([k])
                
                # initialize dictionary for different evaluation results
                evaluation_results = OrderedDict()
                
                # iterate over different split/folds 
                for j in range(k):
                    
                    if verbose == True:
                        print("\t\t\tFold No: {0}".format(j))

                    # get current fold 
                    fold_tmp_X_train = tmp_X_train[j]
                    fold_tmp_y_train = tmp_y_train[j]
                    fold_tmp_X_test = tmp_X_test[j]
                    fold_tmp_y_test = tmp_y_test[j]
                    
                    # get model depending if transoformation is applied 
                    if transform_func == None:
                        tmp_model = model_value
                    # if transformation function is passed get model from tuple at index 0
                    else:
                        tmp_model = model_value[0]

                    # train the current model 
                    tmp_model.fit(fold_tmp_X_train, fold_tmp_y_train)
                    
                    # predict after training
                    fold_predicted_y = tmp_model.predict(fold_tmp_X_test)
                    
                    # calculating accuracy for the current fold 
                    fold_accuracy = accuracy_score(fold_tmp_y_test, fold_predicted_y, normalize = True) 
                    folds_accuracy[j] = fold_accuracy
                    
                    # calculating recall for the current fold 
                    fold_recall = recall_score(fold_tmp_y_test, fold_predicted_y, average = None)
                    folds_recall[j] = fold_recall
                    
                    # calculating precision for the current fold 
                    fold_precision = precision_score(fold_tmp_y_test, fold_predicted_y, average = None)
                    folds_precision[j] = fold_precision
                    
                    # calculating f1 score for the current fold 
                    fold_f1 = f1_score(fold_tmp_y_test, fold_predicted_y, average=None)
                    folds_f1[j] = fold_f1

                    # calculating confusion matrix 
                    fold_cm = confusion_matrix(fold_tmp_y_test, fold_predicted_y, labels)
                    folds_tn[j] = fold_cm[0][0]
                    folds_fp[j] = fold_cm[0][1]
                    folds_fn[j] = fold_cm[1][0]
                    folds_tp[j] = fold_cm[1][1]
                
                # get averages results after cross validation 
                evaluation_results["Accuracy"] = np.mean(folds_accuracy)
                evaluation_results["Recall"] = np.mean(folds_recall, axis = 0)
                evaluation_results["Precision"] = np.mean(folds_precision, axis = 0)
                evaluation_results["F1 Score"] = np.mean(folds_f1, axis = 0)
                evaluation_results["True Negative"] = np.mean(folds_tn)
                evaluation_results["False Postive"] = np.mean(folds_fp)
                evaluation_results["False Negative"] = np.mean(folds_fn)
                evaluation_results["True Postive"] = np.mean(folds_tp)
                
                # add evaluation result to result for this dataset
                dataset_results[str(i + 1)] = evaluation_results
            
            # add results for different imputation techniques for each dataset
            imputation_results[dataset_key] = dataset_results
        
        # add results to each machine learning model
        results[model_key] = imputation_results
                
    # returns results for models
    return results

###########################################################################

