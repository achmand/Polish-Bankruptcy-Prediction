import numpy as np

def min_max_norm(x, a, b, offset):
    x_norm = ((((x - np.min(x))/(np.max(x) - np.min(x))) * (b - a)) + offset)
    return x_norm

def z_score_norm(x):
    x_norm = ((x - np.mean(x)) * np.std(x))
    return x_norm
