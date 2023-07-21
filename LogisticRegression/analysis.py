########################################################################################################################
####---------------------------------------------------Analysis-----------------------------------------------------####
########################################################################################################################

## Used to calculate different types of errors for all algorithms

import numpy as np
from numpy import linalg as LA

class error:
    def __init__(self, problem, model_optimal, cost_optimal):
        self.pr = problem                                       ## problem class
        self.N = self.pr.N                                      ## total number of data samples
        self.X = self.pr.X_train                                ## feature vectors
        self.Y = self.pr.Y_train                                ## label vector             
        self.theta_opt = model_optimal
        self.F_opt = cost_optimal
        
    def path_cls_error(self, iterates):
        iterates = np.array( iterates )          
        Y_predict = np.matmul( self.X, iterates.T )
        error_matrix = np.multiply( Y_predict, self.Y[:,np.newaxis] ) < 0
        return np.sum( error_matrix, axis = 0 ) / self.N

    def point_cls_error(self, theta):
        Y_predict = np.matmul( self.X, theta )
        error = Y_predict * self.Y < 0 
        return sum(error)/self.N

    def theta_gap_path(self, iterates):
        iterates = np.array( iterates )
        if iterates.ndim == 2:  # if iterates is a 2D array, reshape it to 3D. simulate a network with only one node
            iterates = iterates[:,np.newaxis,:]
        norms = np.apply_along_axis( LA.norm, 2, iterates - self.theta_opt )**2
        return np.sum( norms, axis = 1 ) / norms.shape[1]
    
    def cost_gap_point(self, theta):
        return self.pr.F_val(theta) - self.F_opt

    def grad_gap_path(self, iterates):
        iterates = np.array( iterates )
        if iterates.ndim == 2:  # if iterates is a 2D array, reshape it to 3D. simulate a network with only one node
            iterates = iterates[:,np.newaxis,:]
        norms = np.apply_along_axis( LA.norm, 2, iterates )**2
        return np.sum( norms, axis = 1 ) / norms.shape[1]
    
    def cost_gap_path(self, iterates, gap_type = "F"):
        K = len(iterates)
        result = [ ]
        if gap_type == "F":
            for k in range(K):  # TODO: use numpy vectorization
                result.append( error.cost_gap_point(self,iterates[k]) )
        elif gap_type == "theta":
            result = error.theta_gap_path(self, iterates)
        elif gap_type == "grad":
            result = error.grad_gap_path(self, iterates)
        else:
            raise ValueError("gap_type must be either F or theta")
        
        return np.array(result)
        return result





