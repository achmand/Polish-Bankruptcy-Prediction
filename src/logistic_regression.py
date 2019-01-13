###### importing libraries ################################################
import math
import numpy as np 
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

""" This script holds all the classes needed, 
    to build the LogisticRegression.
"""

###### logistic regression ################################################
class LogisticRegression(BaseEstimator, ClassifierMixin):
    """ The LogisticRegression class, which adjusts thetas in training
        and predicts unforseen instances, once its trained.
    """

    def __init__(self, alpha = 0.0001, threshold = 0.5, max_epoch = 1000, penalty = "l2", lambda_t = 0.01, verbose = False):
        """Constructor for LogisticRegression.

        Args:
            alpha (float): The learning rate used in Gradient Descent by default set to 0.0001.
            threshold (float): The threshold used in classification by default set to 0.5.
            max_epoch (int): The number of iterations to be used when training the model.
            penalty (string {'l1' or 'l2'}) : The type of regularization to be used by default set to l2.
            lambad_t (float): The regularaization term to be used (lambda) by default set to 0.01.
            verbose (bool): If set True outputs some information during training.
        """

        self.threshold = threshold
        self.alpha = alpha
        self.max_epoch = max_epoch
        #self.epsilon = epsilon 
        self.penalty = penalty 
        self.lambda_t = lambda_t         
        self.verbose = verbose
    
    def __sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def __criteria_met(self):
        if self.curr_epoch >= self.max_epoch:
            return True
        else: 
            return False
        
    def __cost_function(self, y, h, thetas):

        if self.penalty == "l1":
            return -np.mean(y * np.log(h) + ((1.0 - y) * np.log(1.0 - h))) + ((self.lambda_t / (2 * len(y))) * np.sum(np.abs(thetas[1:])))
        elif self.penalty == "l2":
            return -np.mean(y * np.log(h) + ((1.0 - y) * np.log(1.0 - h))) + ((self.lambda_t  / (2 * len(y))) * np.sum(np.power(thetas[1:], 2)))
        else:
            -np.mean(y * np.log(h) + ((1.0 - y) * np.log(1.0 - h)))

    def fit(self, X, y):
        """Fits the LogisticRegression model (training).

        Args:
            x (numpy array): The instances used for training X.
            y (numpy array): A 1D numpy array with the respective labels for X.
        """

        # set current epoch to 0 
        self.curr_epoch = 0

        # set current epsilon 
        self.curr_epsilon = float("inf")

        # number of instances
        m = float(len(y))
       
        # add bias and initialize as 1 (x0 * 1 = x0), for vectorization purposes
        bias = np.ones((X.shape[0], 1))

        # concatenate bias with features, for vectorization purposes 
        x_bias = np.concatenate((bias, X), axis = 1)

        # initialize thetas to 0s
        self.thetas = np.zeros(x_bias.shape[1])

        # verbose print 
        verbose_print = int(1/10 * self.max_epoch)

        #current_cost = 0.0
        #prev_cost = float("inf")

        # keep looping until termination is met 
        # also finding best thetas/weights using gradient descent (decreasing errors)
        while self.__criteria_met() == False: 

            # theta transpose * X
            z = np.dot(x_bias, self.thetas)

            # hypothesis 
            h = self.__sigmoid(z)

            # calculate gradient 
            gradient = (np.dot(x_bias.T, (h - y)) / m)

            # apply penalty 
            if self.penalty == "l1":
                gradient[1:] = gradient[1:] + (abs(self.lambda_t) / math.sqrt(np.power(self.lambda_t, 2))) * (m) * self.thetas[1:]
            elif self.penalty == "l2":
                l2_term = (self.lambda_t / m) * self.thetas[1:]
                gradient[1:] = gradient[1:] + l2_term

            # adjust thetas/weights
            self.thetas -= (self.alpha * gradient)
            
            # calculate current cost 
            current_cost = self.__cost_function(y, h, self.thetas)

            # print current loss if verbose is on 
            if self.verbose == True and self.curr_epoch % verbose_print == 0:
                print("Current Loss: {0}".format(current_cost)) 

            # increment current epoch 
            self.curr_epoch += 1     

        self.coef_ = self.thetas[1:]
        
        # print details
        if self.verbose:
            print("\nCurrent Epoch: {0}".format(self.curr_epoch))
            print("Current Loss: {0}".format(current_cost))

    def predict(self, X):
        """Predicts unforeseen instances.

        Args:
            x (numpy array): The instances used for predictions X.
           
        Returns:
            numpy array: A numpy 1D array with the predicted outcomes.
        """

        # add bias and initialize as 1 (x0 * 1 = x0), for vectorization purposes
        bias = np.ones((X.shape[0], 1))       
        
        # concatenate bias with features, for vectorization purposes 
        x_bias = np.concatenate((bias, X), axis = 1)
        
        # theta transpose * X
        z = np.dot(x_bias, self.thetas)

        # use sigmoid to classify 
        return (self.__sigmoid(z) >= self.threshold).astype(np.int8)
        
###########################################################################
