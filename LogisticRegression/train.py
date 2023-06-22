"""train.py

Launch the training process of the logistic regression problem on given graph network.
"""

import os
import time
import copy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from graph import Weight_matrix, Geometric_graph, Exponential_graph, Grid_graph
from analysis import error
from Problems.logistic_regression import LR_L2
from Problems.log_reg_cifar import LR_L4
from Optimizers import COPTIMIZER as copt
from Optimizers import DOPTIMIZER as dopt
from utilities import (
    load_state,
)
from ExponentialNet import centralized_algo, decentralized_algo

np.random.seed(0)

"""
Data processing for MNIST
"""
node_num = 16  ## number of nodes
logis_model = LR_L2(
    node_num, limited_labels=False, balanced=True
)  ## instantiate the problem class
dim = logis_model.p  ## dimension of the model
L = logis_model.L  ## L-smooth constant
total_train_sample = logis_model.N  ## total number of training samples
avg_local_sample = logis_model.b  ## average number of local samples

"""
Initializing variables
"""
model_para_central = np.random.normal(
    0, 1, dim
)  # initialize the model parameter for central algorithms
model_para_dis = np.array([cp.deepcopy(model_para_central) for i in range(node_num)])

graph = "exponential"
if graph == "exponential":
    undir_graph = Exponential_graph(
        node_num
    ).undirected()  # generate the undirected graph
elif graph == "grid":
    undir_graph = Grid_graph(np.sqrt(node_num, dtype=np.int32)).undirected()

communication_matrix = Weight_matrix(
    undir_graph
).column_stochastic()  # generate the communication matrix
communication_rounds = [
    1,
    10,
    20,
]  # list of number of communication rounds for decentralized algorithms experiments

C_algos = []  # "SGD", "CRR"
D_algos = ["DRR", "DSGD"]  # "DRR", "DSGD"

CEPOCH_base = 1600  # number of epochs for central algorithms
DEPOCH_base = 1600  # number of epochs for decentralized algorithms

C_lr = [1 / 8000]  # list of learning rate for central algorithms experiments
D_lr = [1 / 8000]  # list of learning rate for decentralized algorithms experiments

C_batch_size = [1]  # list of batch size for central algorithms experiments
D_batch_size = [1]  # list of batch size for decentralized algorithms experiments

C_lr_dec = False  # whether to decay the learning rate for central algorithms
D_lr_dec = False  # whether to decay the learning rate for decentralized algorithms
C_train_load = (
    False  # whether to load the optimal model parameter for central algorithms
)
D_train_load = (
    False  # whether to load the optimal model parameter for decentralized algorithms
)
C_stop_at_converge = False  # whether to stop the training when the model converges for central algorithms
D_stop_at_converge = False  # whether to stop the training when the model converges for decentralized algorithms
save_theta_path = False  # whether to save the model parameter training path for central and decentralized algorithms
grad_track = True  # whether to track the gradient norm for decentralized algorithms
load_init_theta = False  # whether to load the initial model parameter from pretrained model

line_formats = [  # list of line formats for plotting
    "-vb",
    "-^m",
    "-dy",
    "-sr",
    "-1k",
    "-2g",
    "-3r",
    "-.kp",
    "-+c",
    "-xm",
    "-|y",
    "-_r",
]
exp_name = "test_distributed_algo"
exp_log_path = f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/{exp_name}"  # path to save the experiment results
ckp_load_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/optimum"  # path to load the optimal model parameter
plot_every = 50  # plot every 250 epochs
save_every = 500  # save the model parameter every 5000 epochs

"""
Optimum solution
"""
if ckp_load_path is not None:
    theta_CSGD_0, theta_opt = load_state(ckp_load_path, "optimum", "optimal")
theta_CSGD_0 = None
error_lr_0 = error(
    logis_model, theta_opt, logis_model.F_val(theta_opt)
)  # instantiate the error class


if os.path.exists(f"{exp_log_path}/DSGD"):
    print("experiments have been done")
    exit()

print(f"{'-'*50}")
print(f"{graph} Graph")
print(f"node num = {node_num}")
print(f"dim = {dim}")
print(f"L = {L}")
print(f"total train sample = {total_train_sample}")
print(f"avg local sample {avg_local_sample}")
print(f"CEPOCH base = {CEPOCH_base}")
print(f"DEPOCH base {DEPOCH_base}")
print(f"communication rounds = {communication_rounds}")
print(f"C lr = {C_lr}")
print(f"D lr = {D_lr}")
print(f"C batch size = {C_batch_size}")
print(f"D batch size = {D_batch_size}")
print(f"C lr dec = {C_lr_dec}")
print(f"D lr dec = {D_lr_dec}")
print(f"C train load = {C_train_load}")
print(f"D train load = {D_train_load}")
print(f"C stop at converge = {C_stop_at_converge}")
print(f"D stop at converge = {D_stop_at_converge}")
print(f"save theta path = {save_theta_path}")
print(f"grad track = {grad_track}")

print(f"exp name = {exp_name}")
print(f"exp log path = {exp_log_path}")
print(f"ckp load path = {ckp_load_path}")
print(f"plot every = {plot_every}")
print(f"save every = {save_every}")

print(f"{'-'*50}", flush=True)
start = time.time()

for algo in C_algos:
    centralized_algo(
        logis_model,
        model_para_central,
        C_lr,
        C_lr_dec,
        C_batch_size,
        CEPOCH_base,
        exp_log_path,
        save_every,
        error_lr_0,
        line_formats,
        plot_every,
        C_train_load,
        save_theta_path,
        algo,
    )
for algo in D_algos:
    decentralized_algo(
        logis_model,
        model_para_dis,
        D_lr,
        D_lr_dec,
        D_batch_size,
        DEPOCH_base,
        communication_matrix,
        communication_rounds,
        grad_track,
        exp_log_path,
        save_every,
        error_lr_0,
        line_formats,
        plot_every,
        D_train_load,
        save_theta_path,
        algo,
    )

end = time.time()
print(f"{'-'*50}")
print(f"Total time: {end-start:.2f} seconds")
print(f"{'-'*50}", flush=True)
