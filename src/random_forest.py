###### importing libraries ################################################
import math
import random
import numpy as np
from decision_tree import DecisionTree

""" This script holds all the classes needed, 
    to build a RandomForest.
"""

###### random forest ######################################################
class RandomForest:
    def __init__(self, n_estimators = 10, max_features = "auto", bag_ratio = 0.6, bag_features = True, random_seed = None):
        """Constructor for RandomFores.

        Args:
            n_estimators (int): The number of estimators to be used (DecisionTrees).
            max_features (str or int or float) If str (auto or sqrt uses sqrt, log2 uses log base 2 ) uses function for max_features, if int uses max_features as the int set, if float takes percentage.
            bag_ratio (float): The percentage of the dataset to use when bagging the dataset.
            bag_features (bool): If true dataset bagging will be used, if not dataset bagging wont be used.
            random_seed (int): If passed a random seed would be used when random is used (bagging/random subspace method).
        """

        self.n_estimators = 10 if n_estimators <= 0 else n_estimators
        self.random = random 

        # seed random if passed     
        if type(random_seed) is int:    
            self.random.seed(random_seed)

        self.max_features = max_features
        
        # if bag_features is true make sure that the bag_ratio passed is correct 
        self.bag_features = bag_features
        if self.bag_features == True:
            self.bag_ratio = 0.6 if bag_ratio > 1 or bag_ratio <= 0 else bag_ratio
    
    def __sample_instances(self, instances):

        # initialize subsample list
        subsample = []

        # loop to create a random subsample from the instances using replacement
        for _ in range(self.bag_instances_len):
            # get a random index and append to subsample 
            random_index = self.random.randrange(self.instances_len)
            subsample.append(instances[random_index])

        # return subsample as numpy array
        return np.array(subsample)
    
    def __construct_forest(self, instances):
            
        # compute the total size of samples 
        self.instances_len = len(instances)

        # if bag_features is true compute the length of each bag using bag_ratio
        if self.bag_features == True:
            self.bag_instances_len = round(self.instances_len * self.bag_ratio)

        # initialize forest
        forest = []
        
        # loop and build random forest 
        for _ in range(self.n_estimators):

            # generate random indexes for random subspace (selecting random features)
            tmp_indexes = self.random.sample(range(0, self.total_features), self.subspace_len)
            
            # append index for the outcome columns
            tmp_indexes.append(self.total_features)
            
            # sort indexes
            tmp_indexes.sort()
            
            # convert random indexes into numpy array 
            subspace_indexes = np.array(tmp_indexes)
            
            # get instances with randomly picked features
            subspace_instances = instances[:, subspace_indexes]

            # if bootrap is true get sample from subspace_instances (with replacement)
            if self.bag_features == True:
                training_instances = self.__sample_instances(subspace_instances)
            else:
                training_instances = subspace_instances

            # initialize DecisionTree   
            decision_tree = DecisionTree()
            
            # train decision tree 
            subspace_x = training_instances[:,0:-1]
            subspace_y = training_instances[:,-1] 
            decision_tree.fit(subspace_x, subspace_y)

            # append tree to forest
            forest.append((decision_tree,subspace_indexes[0:-1]))

        # return forest containing tree
        return forest

    def fit(self, x, y):
        """Fits the RandomForest model (training).

        Args:
            x (numpy array): The instances used for training X.
            y (numpy array): A 1D numpy array with the respective labels for X.
        """

        # set the number of features to used in RandomForest (attribute bagging)
        self.total_features = x.shape[1]
        if type(self.max_features) is str:
            if self.max_features == "auto" or self.max_features == "sqrt":
                self.subspace_len = int(round(math.sqrt(self.total_features)))
            elif self.max_features == "log2":
                self.subspace_len = int(round(np.log2(self.total_features)))
            else: 
                self.subspace_len = int(round(math.sqrt(self.total_features)))
        elif type(self.max_features) is int: 
            self.subspace_len = self.total_features if self.max_features > self.total_features or self.max_features <= 0 else self.max_features
        elif type(self.max_features is float):
            self.subspace_len = self.total_features if self.max_features > 1 or self.max_features <= 0 else int(round(self.total_features * self.max_features))
        else: 
            self.subspace_len = int(round(math.sqrt(self.total_features)))
        
        # concatenate x and y   
        instances = np.array(np.concatenate((x, np.matrix(y).T), axis=1))
        
        # construct forest (train)
        self.forest = self.__construct_forest(instances)     

    def __majority_vote(self, array):
        return np.bincount(array).argmax()

    def predict(self, x):
        """Predicts unforeseen instances.

        Args:
            x (numpy array): The instances used for predictions X.
           
        Returns:
            numpy array: A numpy 1D array with the predicted outcomes.
        """

        predictions = np.array([tree[0].predict(x[:,tree[1]]) for tree in self.forest]) 
        return np.apply_along_axis(self.__majority_vote, 0, predictions)
        
##########################################################################



