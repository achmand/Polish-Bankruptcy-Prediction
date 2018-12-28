import numpy as np 
cimport numpy as np 
import decision_tree_criterion as dc
cimport cython

##########################################################################
cdef class DecisionTree2:
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef tuple __get_split(self, np.ndarray instances):
        
        # total length of features
        cdef int x_len = instances.shape[1] - 1

        # calculate starting impurity
        cdef double base_gini = dc.gini_impurity(instances[:,x_len].astype(np.int8))

        # define and init best 
        cdef double best_gain = 0 
        cdef list best_decision = [0,0]
        
        cdef int feature
        cdef double [:, :] feature_values, sorted_values
        cdef np.ndarray[np.double_t, ndim = 1] candidates
        cdef double threshold , current_gain

        # loop through all features
        for feature in range(x_len):

            # seperate each feature together with it's classification
            x_feature_values = instances[:, [feature, x_len]]
            feature_values = x_feature_values[x_feature_values[:,0].argsort()]
                
            candidates = feature_values[np.insert(np.diff(feature_values[:1]).astype(np.bool),0,True)][:0]
            
            # get the value in between 
            candidates = (candidates[1:] + candidates[:-1]) / 2

        # return best decision and gain (split)
        return best_gain, best_decision
