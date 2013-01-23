import numpy as np
import matplotlib.pyplot as plt
import random

class Mog_2d_3class:
    
    def sample(self, n_per):
        X_1 = np.random.standard_normal(size=(n_per, 2)) + np.array([3, 1])
        X_2 = np.random.standard_normal(size=(n_per, 2)) + np.array([1.5, -2])
        X_3 = np.random.standard_normal(size=(n_per, 2)) + np.array([-1, 0.5])
        Y_1 = np.zeros(n_per)
        Y_2 = np.ones(n_per)
        Y_3 = np.ones(n_per) * 2
        X = np.concatenate([X_1, X_2, X_3], axis=0)
        Y = np.concatenate([Y_1, Y_2, Y_3], axis=0)
        Xfloat32 = np.array( X, dtype=np.float32)
        y = np.array( Y, dtype=np.int32 )
        row_indices = range(n_per*3)
        random.shuffle( row_indices )
        shuffled_Xfloat32 = Xfloat32[row_indices, :]
        shuffled_y = y[row_indices]
        return shuffled_Xfloat32, shuffled_y