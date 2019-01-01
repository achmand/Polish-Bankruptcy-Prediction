###### importing libraries ################################################
import numpy as np 
import math

###### logistic regression ################################################
class LogisticRegression:

    def __init__(self, alpha = 0.0001, threshold = 0.5,  max_epoch = None, epsilon = 0.00001, penalty = "l2", lambda_t = 1.0, verbose = False):
        
        self.threshold = threshold
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.epsilon = epsilon 
        self.penalty = penalty 
        self.lambda_t = lambda_t 

        # check if we should use max epoch for termination
        self.use_max_epoch = False 
        if type(max_epoch) is int and max_epoch > 0:
            self.use_max_epoch = True 
        
        self.verbose = verbose
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __criteria_met(self):

        # if max_epoch was passed must checck for both epsilon and current epochs
        if self.use_max_epoch == True:
            if self.curr_epoch >= self.max_epoch or self.curr_epsilon < self.epsilon:
                return True
            else: 
                return False
        # else only check for epsilon 
        else: 
            if self.curr_epsilon < self.epsilon:
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


    # TODO -> Must add epsilon as termination criteria using cost 
    def fit(self, x, y):
        
        # set current epoch to 0 
        self.curr_epoch = 0

        # set current epsilon 
        self.curr_epsilon = float("inf")

        # number of instances
        m = len(y)
       
        # add bias and initialize as 1 (x0 * 1 = x0), for vectorization purposes
        bias = np.ones((x.shape[0], 1))

        # concatenate bias with features, for vectorization purposes 
        x_bias = np.concatenate((bias, x), axis = 1)

        # initialize thetas to 0s
        thetas = np.zeros(x_bias.shape[1])

        # verbose print 
        if self.use_max_epoch:
            verbose_print = int(1/10 * self.max_epoch)

        current_cost = 0.0
        prev_cost = float("inf")

        # keep looping until termination is met 
        # also finding best thetas/weights using gradient descent (decreasing errors)
        while self.__criteria_met() == False: 
            
            # set prev to current
            prev_cost = current_cost

            # theta transpose * X
            z = np.dot(x_bias, thetas)

            # hypothesis 
            h = self.__sigmoid(z)

            # calculate gradient 
            gradient = (np.dot(x_bias.T, (h - y)) / m)

            # apply penalty 
            if self.penalty == "l1":
                gradient[1:] = gradient[1:] + (abs(self.lambda_t) / math.sqrt(np.power(self.lambda_t, 2))) * (m) * thetas[1:]
            elif self.penalty == "l2":
                gradient[1:] = gradient[1:] + (self.lambda_t / m) * thetas[1:]

            # adjust thetas/weights
            thetas -= self.alpha * gradient

            # calculate current cost 
            current_cost = self.__cost_function(y, h, thetas)

            # calculate current epsilon
            self.curr_epsilon = np.abs(prev_cost - current_cost)

            # print current loss if verbose is on 
            if self.verbose == True and self.curr_epoch % verbose_print == 0:
                print("Current Loss: {0}".format(current_cost)) 

            # increment current epoch 
            self.curr_epoch += 1     

        # print details
        if self.verbose:
            print("\nCurrent Epoch: {0}".format(self.curr_epoch))
            print("Current Epsilon: {0}".format(self.curr_epsilon))
            print("Current Loss: {0}".format(current_cost))

        # set thetas to thetas we found aftar training
        self.thetas = thetas

    def predict(self, x):
        
        # add bias and initialize as 1 (x0 * 1 = x0), for vectorization purposes
        bias = np.ones((x.shape[0], 1))       
        
        # concatenate bias with features, for vectorization purposes 
        x_bias = np.concatenate((bias, x), axis = 1)
        
        # theta transpose * X
        z = np.dot(x_bias, self.thetas)

        # use sigmoid to classify 
        return (self.__sigmoid(z) >= self.threshold).astype(np.int8)

###########################################################################
