import numpy as np 
cimport numpy as np 

cpdef double gini_impurity(np.ndarray[np.int8_t, ndim = 1, mode = 'c'] y):
    
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

    cdef int z = 1 
    return z - prob_category
    