################################################################################################################################
##---------------------------------------------------Decentralized Optimizers-------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
from numpy import linalg as LA



def D_SGD(prd, weight, learning_rate, K, theta_0, batch_size):
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]

    for k in range(K):
        sample_vec = [np.random.choice(prd.data_distr[i]) for i in range(prd.n)]
        sample_vec = [
            (
                val, max(val + batch_size, prd.data_distr[i])   # tuple of index for training batch i -> from index val to index val + batch size
            ) for i, val in enumerate(sample_vec)
        ]
        grad = prd.networkgrad( theta, batch_idx = sample_vec )

        theta = np.matmul( weight, theta ) - learning_rate * grad  

        ut.monitor('D_SGD', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )

    return theta_epoch
    

def D_RR(prd, weight, learning_rate, K, theta_0, batch_size):
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]

    # sample_vec = [np.random.permutation(prd.data_distr[i]) for i in range(prd.n)]    # each client needs a local batch index to perform updates
    # slices = [0] * len(sample_vec)
    # grad = prd.networkgrad( theta, permute = sample_vec )

    slices = [prd.data_distr[i] for i in range(prd.n)]
    sample_vec = [np.random.permutation(prd.data_distr[i]) for i in range(prd.n)]

    for k in range(K):
        permutes = []
        for i, sequence in enumerate(sample_vec):
            if slices[i] >= prd.data_distr[i]:
                slices[i] = 0
                sample_vec[i] = np.random.permutation(prd.data_distr[i])

            permutes.append(sequence[slices[i]:slices[i] + batch_size])         # same batch size across all clients TODO: is this good?
            slices[i]+=batch_size

        grad = prd.networkgrad( theta, permute = permutes )

        theta = np.matmul( weight, theta ) - learning_rate * grad  

        ut.monitor('D_RR', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )

    return theta_epoch

def DPG_RR():
    # DRR with different communication frequency
    pass


def SADDOPT(prd,B1,B2,learning_rate,K,theta_0):   
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad( theta, sample_vec )
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta = np.matmul( B1, theta ) - learning_rate * tracker  
        grad_last = cp.deepcopy(grad)
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad = prd.networkgrad( z, sample_vec )
        tracker = np.matmul( B2, tracker ) + grad - grad_last
        ut.monitor('SADDOPT', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )
    return theta_epoch

def GP(prd,B,learning_rate,K,theta_0):
    theta = [cp.deepcopy( theta_0 )]
    grad = prd.networkgrad( theta[-1] )
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta.append( np.matmul( B, theta[-1] ) - learning_rate * grad ) 
        Y = np.matmul( B, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta[-1] )
        grad = prd.networkgrad( z )
        ut.monitor('GP', k, K)
    return theta

def ADDOPT(prd,B1,B2,learning_rate,K,theta_0):   
    theta = [ cp.deepcopy(theta_0) ]
    grad = prd.networkgrad( theta[-1] )
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta.append( np.matmul( B1, theta[-1] ) - learning_rate * tracker ) 
        grad_last = cp.deepcopy(grad)
        Y = np.matmul( B1, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta[-1] )
        grad = prd.networkgrad( z )
        tracker = np.matmul( B2, tracker ) + grad - grad_last 
        ut.monitor('ADDOPT', k ,K)
    return theta

def SGP(prd,B,learning_rate,K,theta_0):   
    theta = cp.deepcopy( theta_0 )
    theta_epoch = [ cp.deepcopy(theta) ]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad( theta, sample_vec )
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta = np.matmul( B, theta ) - learning_rate * grad 
        Y = np.matmul( B, Y )
        YY = np.diag(Y)
        z = np.matmul( LA.inv(YY), theta )
        sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
        grad = prd.networkgrad( z, sample_vec )
        ut.monitor('SGP', k, K)
        if (k+1) % prd.b == 0:
            theta_epoch.append( cp.deepcopy(theta) )
    return theta_epoch

