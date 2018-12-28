import numpy as np 
cimport numpy as np 
from cpython cimport bool

cpdef double gini_impurity(np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y):
    """A metric to measure the relative frequency of a category (j) at node (i).
    Maximum: Least interesting info as all categories (y) are equally distributed. 
    Minimum: Most interesting info as all catgeories (y) are the same. 
    Basically a way to 

    Args:
        y (numpy.ndarray): A numpy 1 dimensional array of categorical inputs (y/labels/outcome).

    Returns:
        double: The gini impurity measurement. Quantification of how much uncertainty is there at a specific node. 
    """

    # get count of rows
    cdef double y_len = len(y)
    cdef int empty = 0 

    # if empty array return 0 impurity 
    if(y_len) == empty: 
        return 0.0
    
    # get number of occurences
    cdef np.ndarray[np.int64_t, ndim = 1] occurrences = np.unique(y, return_counts = True)[1]

    # probability for catgeory 
    cdef double prob_category = 0
  
    # loop an calculate probability
    cdef int i 
    cdef int occurrences_len = len(occurrences)
    for i in range(occurrences_len):
        prob_category += (occurrences[i] / y_len) * (occurrences[i] / y_len)

    cdef double z = 1 
    return z - prob_category

cpdef double binary_gini_impurity(np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y):
    """A metric to measure the relative frequency of a category (j) at node (i).
    Maximum: Least interesting info as all categories (y) are equally distributed. 
    Minimum: Most interesting info as all catgeories (y) are the same. 
    Basically a way to 

    Args:
        y (numpy.ndarray): A numpy 1 dimensional array of categorical inputs (y/labels/outcome).

    Returns:
        double: The gini impurity measurement. Quantification of how much uncertainty is there at a specific node. 
    """

    # if empty array return 0 impurity 
    cdef unsigned int empty = 0
    cdef unsigned int y_range = len(y)
    if y_range == empty: 
        return 0.0

    cdef int i
    cdef unsigned int counter_a = 0
    cdef unsigned int counter_b = 0
    for i in range(y_range):    
        if(y[i] == 0):
            counter_a += 1 
        else:
            counter_b += 1

    cdef double y_len = len(y)
    cdef double prob_category = 1.0 - (((counter_a / y_len) * (counter_a / y_len)) + ((counter_b / y_len) * (counter_b / y_len)))
    return prob_category

cpdef double information_gain(np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y_left, np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y_right, double gini):
    
     # calculate the percentage of y (left and right) 
    cdef double y_left_len = len(y_left)
    cdef double y_right_len = len(y_right)
    cdef double y_left_perc = y_left_len / (y_left_len + y_right_len)
    cdef double y_right_perc = 1 - y_left_perc
    
    # calculate the weighted avg gini impurity for the two nodes
    cdef double l_impurity = y_left_perc * gini_impurity(y_left)
    cdef double r_impurity = y_right_perc * gini_impurity(y_right)
    
    # calculate and return information gain
    return gini - l_impurity - r_impurity

cpdef double binary_information_gain(np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y_left, np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y_right, double gini):
    
     # calculate the percentage of y (left and right) 
    cdef double y_left_len = len(y_left)
    cdef double y_right_len = len(y_right)
    cdef double y_left_perc = y_left_len / (y_left_len + y_right_len)
    cdef double y_right_perc = 1 - y_left_perc
    
    # calculate the weighted avg gini impurity for the two nodes
    cdef double l_impurity = y_left_perc * binary_gini_impurity(y_left)
    cdef double r_impurity = y_right_perc * binary_gini_impurity(y_right)
    
    # calculate and return information gain
    return gini - l_impurity - r_impurity

cpdef tuple decision_split(np.ndarray x, (int, double) decision):
    cdef int col_pos = decision[0]
    cdef double decision_val = decision[1]
    cdef int rows_len = x.shape[0]
    
    cdef list true_instances = []
    cdef list false_instances = []

    cdef int i
    cdef double current_val
    for i in range(rows_len):
        current_val = x[i][col_pos] 
        if(current_val >= decision_val):
            true_instances.append(x[i])
        else:   
            false_instances.append(x[i])
    
    return np.array(true_instances), np.array(false_instances)
