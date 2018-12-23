# importing libraries 
import numpy as np

##########################################################################
class Decision: 
    
    def __init__(self, col, val):
        self.col = col
        self.val = val 

    def compare(self, instance):
        # getting value for instance depending on the column being evaluated 
        instance_val = instance[self.col]
  
        # checking if the instance value is numeric 
        if(isinstance(instance_val, (int, float))):
            return instance_val >= self.val
        else: # else instance value is categorical so check if it is equal to
            return instance_val == self.val

    def __repr__(self):
        # output of the Decision as a str 
        compare_condition = ">=" if isinstance(self.val, (int, float)) == True else "=="
        return "compare value: {0} {1}".format(compare_condition, str(self.val))

##########################################################################
class LeafNode:
    def __init__(self, y):
        self.outcomes = category_counts(y)

##########################################################################
class InternalNode:
    def __init__(self, decision, left_branch, right_branch):
        self.decision = decision
        self.left_branch = left_branch
        self.right_branch = right_branch

##########################################################################

def category_counts(y):
    category, occurrences = np.unique(y, return_counts = True)
    return dict(zip(category, occurrences))

def gini_impurity(y):
    """A metric to measure the relative frequency of a category (j) at node (i).
    Maximum: Least interesting info as all categories (y) are equally distributed. 
    Minimum: Most interesting info as all catgeories (y) are the same. 
    Basically a way to 

    Args:
        y (numpy array): A numpy 1 dimensional array of categorical inputs (y/labels/outcome).

    Returns:
        float: The gini impurity measurement. Quantification of how much uncertainty is there at a specific node. 
    """

    # get count of rows
    rows_count = len(y)

    # if empty array return 0 impurity 
    if(rows_count) == 0:
        return 0.0 
    
    # gets dict of categorical occurences 
    occurrences = category_counts(y)

    # calculate summation for probabilities 
    prob_category = np.sum([(occurrences[key] / rows_count) ** 2 for key in occurrences])

    # calulate and return gini impurity 
    return 1 - prob_category

# TODO -> Add comments ... 
def information_gain(left_instances, right_instances, current_gini):

    # calculate the number of rows in left row 
    # (left_node_count/left_node_count + right_node_count)
    lnode_count = float(len(left_instances)) / float(len(left_instances) + len(right_instances))
    rnode_count = 1 - lnode_count

    #print(lnode_count)
    #print(rnode_count)

    # get outcome column index 
    outcome_col = left_instances.shape[1] - 1

    # calculate the weighted avg gini impurity for the two nodes
    lnode_gini_index = lnode_count * gini_impurity(left_instances[:,outcome_col])
    rnode_gini_index = rnode_count * gini_impurity(right_instances[:,outcome_col])

    # calculate and return information gain
    return current_gini - lnode_gini_index - rnode_gini_index

# TODO -> Add comments
def decision_split(instances, decision):
    
    # define two lists to append instances which satisfies the decision or not
    true_instances = []
    false_instances = []

    # loop through instances and append according to the satisfied decision
    for instance in instances:
        tmp_list = true_instances if decision.compare(instance) == True else false_instances
        tmp_list.append(instance)

    # return lists split by decision
    return np.array(true_instances, dtype = object), np.array(false_instances, dtype = object)

def get_split(instances):

    # get the total number of features
    total_features = instances.shape[1] - 1
    
    # index with total features to get last column (outcome)
    outcome_col = instances[:,total_features]

    # calculate the impurtity of the current instances
    current_gini = gini_impurity(outcome_col)  
    
    # initialize best gain and decision which gives back best gain
    best_gain = 0 
    best_decision = None

    # loop through each feature to find the best split
    for col_no in range(total_features):
        
        # get unique values in a specific indexed column
        feature_values = np.unique(instances[:,col_no])
        
        # loop through each feature value to check for best information gain
        for feature_value in feature_values:

            # apply a decision split and get two lists (true & false)
            current_decision = Decision(col_no, feature_value)
            true_instances, false_instances = decision_split(instances, current_decision)

            # check if the data was divided and not one sided 
            no_split = (len(true_instances) == 0 or len(false_instances) == 0)
            # if no split continue iteration
            if(no_split):
                continue 
            
            # calculate information gain for this split 
            current_info_gain = information_gain(true_instances, false_instances, current_gini)

            # if information gain is higher than best set new best 
            if current_info_gain > best_gain:
                best_gain = current_info_gain
                best_decision = current_decision

    # return best split
    return best_gain, best_decision

def construct_tree(instances):

    # split dataset and get the decision with the highest gain
    gain, decision = get_split(instances)
    
    # there is no more highest gain return LeafNode
    if(gain == 0):
        # get outcome column index 
        outcome_col = instances.shape[1] - 1
        return LeafNode(instances[:,outcome_col])

    # if there is more gain to get from other features, re-split 
    true_instances, false_instances = decision_split(instances, decision)

    # use recursion to continue construct the tree
    true_branch = construct_tree(true_instances)
    false_branch = construct_tree(false_instances)

    # returns an internal node 
    return InternalNode(decision, true_branch, false_branch)

def classify(instance, node):

    # if lead node is reach output outcome 
    if isinstance(node, LeafNode):
        return node.outcomes
    
    # traverse using recursion the decision tree until u find outcome 
    branch = node.left_branch if node.decision.compare(instance) else node.right_branch
    return classify(instance, branch) 

