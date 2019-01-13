###### importing libraries ################################################
import numpy as np 
import decision_tree_criterion as dc

""" This script holds all the classes needed, 
    to build a DecisionTree.
"""

###### Leaf Node Class ###################################################
class LeafNode:
    """ The LeafNode class, holds the final node in
        the DecisionTree. This is used to make the final 
        prediction.
    """
    
    def __init__(self, y):
        """Constructor for LeafNode.

        Args:
            y (numpy array): A numpy 1D array with labels for this LeafNode.
        """

        # get unique lables with count
        category, occurrences = np.unique(y, return_counts = True)
        # convert to dictionary
        outcomes = dict(zip(category, occurrences))
        # set outcome/prediction to the maximum count of the label 
        self.outcome = max(outcomes, key=outcomes.get)

###### Internal Node Class ###############################################
class InternalNode:
    """ The InternalNode class, the nodes found in the DecisionTree.
        Each InternalNode has the left and right node references which,
        can be another InternalNode or a LeafNode. The first InternalNode 
        in the DecisionTree is the root node.
    """
    
    def __init__(self, decision, left_branch, right_branch):
        """Constructor for InternalNode.

        Args:
            decision (numpy array): A numpy 1D array, index 0 feature/column index and index 1 value for split.
            left_branch (InternalNode or LeafNode): A reference to the left child node.
            right_branch (InternalNode or LeafNode): A reference to the right child node.
        """

        # set to the parameters passed in the constructor.
        self.decision = decision
        self.left_branch = left_branch
        self.right_branch = right_branch

###### Decision Tree Class ###############################################
class DecisionTree:
    """ The DecisionTree class, which constructs a tree (training)
        and predicts unforseen instances, once its trained.
    """

    def __get_split(self, instances):
        """Gets the best split for the instances passed.

        Args:
            instances (numpy array): A numpy array with the instances used to get the split for.
       
        Returns:
            float: the best gain found.
            numpy array: A 1D numpy array which contains the decision. Index 0 is for feature index, index 1 the value used for split.
        """

        # total length of features
        x_len = instances.shape[1] - 1

        # calculate starting impurity
        base_gini = dc.binary_gini_impurity(instances[:,x_len].copy(order='C').astype(np.int8))
            
        # define and init best 
        best_gain = 0 
        best_decision = [0,0]
        
        # loop through all features
        for feature in range(x_len):
            
            # seperate each feature together with it's classification
            feature_values = instances[:, [feature, x_len]]
            
            # sort features by value 
            feature_values = feature_values[feature_values[:,0].argsort()]
            
            # get candidates values which seperate classes (0 , 0 , 0 , 1, 1 , 0) => (0 ,1 , 0)  
            candidates = feature_values[np.insert(np.diff(feature_values[:,1]).astype(np.bool),0,True)][:,0]
            
            # get the value in between 
            candidates = (candidates[1:] + candidates[:-1]) / 2
            
            # loop threshold/value of candidates 
            for threshold in candidates:
                            
                # split instances using current threshold
                left_instances = feature_values[feature_values[:,0] >= threshold][:,1].astype(np.int8)
                right_instances = feature_values[feature_values[:,0] < threshold][:,1].astype(np.int8)

                # if there was no split continue 
                if(left_instances.shape[0] == 0 or right_instances.shape[0] == 0):
                    continue
                
                # calculate information gain for this split 
                current_gain = dc.binary_information_gain(left_instances, right_instances, base_gini)
        
                # if the information gain is higher than set new best
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_decision[0] = feature
                    best_decision[1] = threshold
                
        # return best decision and gain (split)
        return best_gain, best_decision

    def __construct_tree(self, instances):
        """Constructs the DecisionTree using recurssion.

        Args:
            instances (numpy array): A numpy array with the instances used to construct the tree.
       
        Returns:
            InternalNode or LeafNode: Returns LeafNode is no more gain can be reached or InternalNode otherwise.
        """

        # split dataset and get the decision with the highest gain
        gain, decision = self.__get_split(instances)

        # there is no more highest gain return LeafNode
        if(gain == 0):
            return LeafNode(instances[:,instances.shape[1] - 1])

        # if there is more gain to get from other features, re-split 
        true_instances, false_instances = dc.decision_split(instances, tuple(decision))

        # use recursion to continue construct the tree
        true_branch = self.__construct_tree(true_instances)
        false_branch = self.__construct_tree(false_instances)

        return InternalNode(decision, true_branch, false_branch)   

    def fit(self,x, y):
        """Fits the DecisionTree model (training).

        Args:
            x (numpy array): The instances used for training X.
            y (numpy array): A 1D numpy array with the respective labels for X.
        
        """

        # concatenate x and y   
        instances = np.array(np.concatenate((x, np.matrix(y).T), axis=1))
        # construct tree (train)
        self.internal_node = self.__construct_tree(instances)

    def __classify(self, instance, node):
        """Traverse DecisionTree nodes using recurssion.
           Until a LeafNode is found and a prediction is made. 

        Args:
            instance (numpy array): A 1D numpy array to represent an instance.
            node (InternalNode or LeafNode): The current Node being searched.
           
        Returns:
            int: The predicted classification encoded in an int.
        """

        # if lead node is reach output outcome 
        if isinstance(node, LeafNode):
            return node.outcome

        # traverse using recursion until u find outcome (LeafNode)
        node_decision = node.decision
        if(instance[node_decision[0]] >= node_decision[1]):
            return self.__classify(instance, node.left_branch)
        else: 
            return self.__classify(instance, node.right_branch)

    def predict(self, x):
        """Predicts unforeseen instances.

        Args:
            x (numpy array): The instances used for predictions X.
           
        Returns:
            numpy array: A numpy 1D array with the predicted outcomes.
        """

        predictions = np.array([self.__classify(i, self.internal_node) for i in x])
        return predictions.astype(np.int8)

##########################################################################