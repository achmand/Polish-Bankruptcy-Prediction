import numpy as np 
import decision_tree_criterion as dc

##########################################################################
class LeafNode:
    def __init__(self, y):
        category, occurrences = np.unique(y, return_counts = True)
        outcomes = dict(zip(category, occurrences))
        self.outcome = max(outcomes, key=outcomes.get)

##########################################################################
class InternalNode:
    def __init__(self, decision, left_branch, right_branch):
        self.decision = decision
        self.left_branch = left_branch
        self.right_branch = right_branch

##########################################################################
class DecisionTree:
    def __get_split(self, instances):

        # total length of features
        x_len = instances.shape[1] - 1

        # calculate starting impurity
        base_gini = dc.gini_impurity(instances[:,x_len].copy(order='C').astype(np.int8))
            
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
                current_gain = dc.information_gain(left_instances, right_instances, base_gini)
        
                # if the information gain is higher than set new best
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_decision[0] = feature
                    best_decision[1] = threshold
                
        # return best decision and gain (split)
        return best_gain, best_decision

    def __construct_tree(self, instances):
        
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
        # concatenate x and y   
        instances = np.array(np.concatenate((x, np.matrix(y).T), axis=1))
        # construct tree (train)
        self.internal_node = self.__construct_tree(instances)

    def __classify(self, instance, node):
        
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
        predictions = np.array([self.__classify(i, self.internal_node) for i in x])
        return predictions.astype(np.int8)

##########################################################################
