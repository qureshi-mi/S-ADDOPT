################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut
import time
import math

## Centralized gradient descent
def CGD(pr,learning_rate,K,theta_0):
    start = time.time()
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  
    for k in range(K):
        theta.append( theta[-1] - learning_rate * pr.grad(theta[-1]) )
        ut.monitor('CGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    print(f"Time Span: {time.time() - start}")
    return theta, theta_opt, F_opt

def SGD(pr, learning_rate, K, theta_0, batch_size):
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  

    update_round = math.ceil(pr.N / batch_size)        # in order to line up with RR case
    print(f"update round {update_round} | pr.N {pr.N}")

    start = time.time()
    for k in range(K):                      # TODO: each k here is actually one batch ... 
        temp = theta[-1]
        for i in range(update_round):
            permutation = np.random.permutation(pr.N)
            temp = temp - learning_rate * pr.grad(
                temp, permute = permutation[0:batch_size], permute_flag = True
            )

        theta.append(temp)
        ut.monitor('SGD',k,K)

    print(f"{k} Round | {update_round}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

def C_RR(pr, learning_rate, K, theta_0, batch_size):
    theta_copy = cp.deepcopy( theta_0 )
    theta = [theta_copy]  
    
    start = time.time()
    for k in range(K):
        cnt = 0
        permutation = np.random.permutation(pr.N)
        temp = theta[-1]
        while cnt < pr.N:
            temp = temp - learning_rate * pr.grad(
                temp, permute = permutation[cnt:cnt + batch_size], permute_flag=True
            ) 
            cnt = cnt + batch_size

        ut.monitor('C_RR',k,K)
        theta.append(temp)
        
    print(f"{k} Round | {cnt / batch_size}# Updates | {batch_size} Batch Size")
    print(f"Time Span: {time.time() - start}")
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
