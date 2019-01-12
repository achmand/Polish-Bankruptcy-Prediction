###### importing libraries ################################################
from distutils.core import setup 
from Cython.Build import cythonize
import numpy

""" This script is used to cythonize the decision_tree_criterion.pyx script.
"""

setup(
    ext_modules=cythonize("decision_tree_criterion.pyx"),
    include_dirs=[numpy.get_include()]
)    

#setup(
#    ext_modules=cythonize("decision_tree.pyx"),
#    include_dirs=[numpy.get_include()]
#)    