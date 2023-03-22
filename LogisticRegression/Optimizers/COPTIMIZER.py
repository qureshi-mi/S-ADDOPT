################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut

def SGD(pr, learning_rate, K, theta_0, batch_size):
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  

    update_round = pr.N / batch_size        # in order to line up with RR case

    for k in range(K):                      # TODO: each k here is actually one batch ... 
        batch_idx = np.random.choice(pr.N - batch_size + 1)
        theta.append( 
            theta[-1] - learning_rate * pr.grad(
                theta[-1], batch_idx = (batch_idx, batch_idx + batch_size)
            ) 
        )
        ut.monitor('SGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

def C_RR(pr, learning_rate, K, theta_0, batch_size):
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  
    
    cnt = pr.N
    for k in range(K):
        cnt = cnt + batch_size
        if cnt >= pr.N:
            cnt = 0
            permutation = np.random.permutation(pr.N)
        
        theta.append( 
            theta[-1] - learning_rate * pr.grad(
                theta[-1], permute = permutation[cnt:cnt + batch_size]
            ) 
        )
        ut.monitor('C_RR',k,K)
        
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

## Centralized gradient descent
def CGD(pr,learning_rate,K,theta_0):
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  
    for k in range(K):
        theta.append( theta[-1] - learning_rate * pr.grad(theta[-1]) )
        ut.monitor('CGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

## Centralized gradient descent with momentum
def CNGD(pr,learning_rate,momentum,K,theta_0):
    theta = [theta_0]  
    theta_aux = cp.deepcopy(theta_0)
    for k in range(K):
        grad = pr.grad(theta[-1])
        theta_aux_last = cp.deepcopy(theta_aux)
        theta_aux = theta[-1] - learning_rate * grad 
        theta.append( theta_aux + momentum * ( theta_aux - theta_aux_last ) )
        ut.monitor('CNGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

## Centralized stochastic gradient descent
def CSGD(pr,learning_rate,K,theta_0):
    N = pr.N
    theta = cp.deepcopy(theta_0)
    theta_epoch = [ theta_0 ]
    for k in range(K):
        idx = np.random.randint(0,N)
        grad = pr.grad(theta,idx)
        theta -= learning_rate * grad 
        if (k+1) % N == 0:
            theta_epoch.append( cp.deepcopy(theta) )
        ut.monitor('CSGD',k,K)
    return theta_epoch
