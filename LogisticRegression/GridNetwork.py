

## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt

"""
Data processing for MNIST
"""
n = 4                                                      ## number of nodes 
lr_0 = LR_L2(n, limited_labels = False, balanced = True )   ## instantiate the problem class 
p = lr_0.p                                                  ## dimension of the model 
L = lr_0.L                                                  ## L-smooth constant
N = lr_0.N                                                  ## total number of training samples
b = lr_0.b                                                  ## average number of local samples
step_size = 1/L/2                                           ## selecting an appropriate step-size

"""
Initializing variables
"""
CEPOCH_base = 3000
depoch = 100
theta_c0 = np.random.normal(0,1,p)
theta_0 = np.random.normal(0,1,(n,p)) 
UG = Exponential_graph(n).directed()
B = Weight_matrix(UG).column_stochastic()
"""
Centralized solutions
"""
## solve the optimal solution of Logistic regression
_, theta_opt, F_opt = copt.CGD(lr_0,10*1/L, CEPOCH_base,theta_c0) 
error_lr_0 = error(lr_0,theta_opt,F_opt)

"""
Decentralized Algorithms
"""
## GP
theta_GP = dopt.SGD()
res_F_GP = error_lr_0.cost_gap_path( np.sum(theta_GP,axis = 1)/n)
