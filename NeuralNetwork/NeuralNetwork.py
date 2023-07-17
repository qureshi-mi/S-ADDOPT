########################################################################################################################
####-------------------------------------------------Neural Network-------------------------------------------------####
########################################################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Problems.my_neural_network_mnist import NN_mnist
from Problems.my_neural_network_cifar import NN_cifar
from graph import Weight_matrix, Geometric_graph
from Optimizers import DOPTIMIZER as dopt

########################################################################################################################
####--------------------------------------------CIFAR-10 Classification---------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Geometric directed graphs using neural networks.

start = time.time()
exp_log_path = input('Enter the path to save the experiment log: ')

"""
Data processing for MNIST
"""
n = 500                                                 # number of nodes
hidden = 64                                             # number of neurons in the hidden layer   

"""
Initializing variables
"""
UG = Geometric_graph(n).directed(0.07, 0.03)
B = Weight_matrix(UG).column_stochastic()   

"""
Data processing for CIFAR
"""
nn_1 = NN_cifar(n, hidden, limited_label = True)      # neural network class 
m = nn_1.b                                            # number of local data samples
d = nn_1.dim

"""
Initializing variables
"""
depoch = 150
theta_0 = np.random.randn( n,d )/10
step_size = 0.5

"""
Decentralized Algorithms
"""
## SGP
theta_SGP = dopt.SGP(nn_1,B,step_size,int(depoch*m),theta_0)            
loss_SGP, acc_SGP = NN_cifar.loss_accuracy_path(nn_1, theta_SGP)
## SADDOPT     
theta_SADDOPT = dopt.SADDOPT(nn_1,B,B,step_size,int(depoch*m),theta_0) 
loss_SADDOPT, acc_SADDOPT = NN_cifar.loss_accuracy_path(nn_1, theta_SADDOPT)

"""
Save data
"""
np.savetxt(f'{exp_log_path}/CIFAR_SGP_Acc.txt',acc_SGP)
np.savetxt(f'{exp_log_path}/CIFAR_SGP_Loss.txt',loss_SGP)
np.savetxt(f'{exp_log_path}/CIFAR_SADDOPT_Acc.txt',acc_SADDOPT)
np.savetxt(f'{exp_log_path}/CIFAR_SADDOPT_Loss.txt',loss_SADDOPT)

"""
Save plot
"""
font = FontProperties()
font.set_size(23.5)

font2 = FontProperties()
font2.set_size(15)

mark_every = 15
plt.figure(3)
plt.plot(loss_SGP,'-dy', markevery = mark_every)
plt.plot(loss_SADDOPT,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Loss', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.legend(('SGP', 'SADDOPT'), prop=font2)
plt.title('CIFAR-10', fontproperties=font)
plt.savefig('plots/cifar_loss.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')

plt.figure(4)
plt.plot(acc_SGP,'-dy', markevery = mark_every)
plt.plot(acc_SADDOPT,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Accuracy', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.legend(('SGP', 'SADDOPT'), prop=font2)
plt.title('CIFAR-10', fontproperties=font)
plt.savefig('plots/cifar_acc.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')


end = time.time()
print('Time elapsed: ', end - start, 'seconds')


