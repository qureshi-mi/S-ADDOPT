########################################################################################################################
####----------------------------------------------Exponential Network-----------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

import os
import time
import copy as cp
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
from utilities import plot_figure_path, save_npy, save_state, load_state, initDir, load_optimal

np.random.seed(0)

########################################################################################################################
####----------------------------------------------MNIST Classification----------------------------------------------####
########################################################################################################################


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
CEPOCH_base = 10000 # number of epochs for central algorithms
DEPOCH_base = 10000 # number of epochs for decentralized algorithms

model_para_central = np.random.normal(0, 1, dim) # initialize the model parameter for central algorithms
# model_para_dis = np.random.normal(0, 1, (node_num, dim)) # initialize the model parameter for decentralized algorithms
model_para_dis = np.array([cp.deepcopy(model_para_central) for i in range(node_num)])
undir_graph = Exponential_graph(node_num).undirected() # generate the undirected graph
communication_matrix = Weight_matrix(undir_graph).column_stochastic() # generate the communication matrix
communication_rounds = [5,10,20] # list of number of communication rounds for decentralized algorithms experiments

C_lr = [1/100] # list of learning rate for central algorithms experiments
D_lr = [1/100] # list of learning rate for decentralized algorithms experiments
C_batch_size = [12000] # list of batch size for central algorithms experiments
D_batch_size = [1] # list of batch size for decentralized algorithms experiments
C_lr_dec = False # whether to decay the learning rate for central algorithms
D_lr_dec = False # whether to decay the learning rate for decentralized algorithms
C_train_load = False # whether to load the optimal model parameter for central algorithms
D_train_load = False # whether to load the optimal model parameter for decentralized algorithms
C_stop_at_converge = False # whether to stop the training when the model converges for central algorithms
D_stop_at_converge = False # whether to stop the training when the model converges for decentralized algorithms
save_theta_path = False # whether to save the model parameter training path for central and decentralized algorithms
grad_track = True # whether to track the gradient norm for decentralized algorithms

line_formats = [ # list of line formats for plotting
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
exp_name = "init_param"
exp_log_path = f"/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/{exp_name}" # path to save the experiment results
ckp_load_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/optimum" # path to load the optimal model parameter
plot_every = 100 # plot every 250 epochs
save_every = 5000 # save the model parameter every 5000 epochs

"""
Optimum solution
"""
if ckp_load_path is not None:
    theta_CSGD_0, theta_opt = load_state(ckp_load_path, "optimum", "optimal")
theta_CSGD_0 = None
error_lr_0 = error(logis_model, theta_opt, logis_model.F_val(theta_opt)) # instantiate the error class


def centralized_algo(
    logis_model,
    model_para,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
    train_load,
    save_theta_path,
    algo,
):
    exp_save_path = f"{exp_log_path}/central_{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    model_para_cp = cp.deepcopy(model_para)
    params = []
    for bz in C_batch_size:
        for lr in C_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        model_para = model_para_cp
        if os.path.exists(f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_theta.npy"):
            continue
        if train_load:
            model_para = load_optimal(exp_save_path, f"{algo}_opt_theta_bz{bz}_lr{lr:.6f}.npy")

        if algo == "SGD":
            theta, theta_opt, F_opt = copt.SGD(
                logis_model,
                lr,
                CEPOCH_base,
                model_para,
                bz,
                C_lr_dec,
                train_log_path,
                f"{algo}_bz{bz}_lr{lr:.3f}_check",
                save_every,
                error_lr_0,
            )
        elif algo == "CRR":
            theta, theta_opt, F_opt = copt.C_RR(
                logis_model,
                lr,
                CEPOCH_base,
                model_para,
                bz,
                C_lr_dec,
                train_log_path,
                f"{algo}_bz{bz}_lr{lr}_check",
                save_every,
                error_lr_0,
            )

        np.save(f"{exp_save_path}/{algo}_opt_theta_bz{bz}_lr{lr:.6f}.npy", theta_opt)

        res_F = error_lr.cost_gap_path(theta, gap_type="theta")
        np.save(f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_theta.npy", res_F)
        res_F_F = error_lr.cost_gap_path(theta, gap_type="F")
        np.save(f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_F.npy", res_F_F)

        if save_theta_path:
            np.save(f"{exp_save_path}/{algo}_theta_bz{bz}_lr{lr:.6f}.npy", theta)

    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_bz{bz}_lr{lr:.6f}_theta.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)] + ["CGD"],
        f"{exp_save_path}/convergence_{algo}_theta_{exp_name}.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_bz{bz}_lr{lr:.6f}_F.npy" for idx, (bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)] + ["CGD"],
        f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
        plot_every,
    )

def decentralized_algo(
    logis_model,
    model_para,
    D_lr,
    D_lr_dec,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    communication_rounds,
    grad_track,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
    train_load,
    save_theta_path,
    algo,
):
    exp_save_path = f"{exp_log_path}/{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    model_para_cp = cp.deepcopy(model_para)
    exp_names = []
    legends = []
    params = []
    for bz in D_batch_size:
        for lr in D_lr:
            for cr in communication_rounds:
                params.append((bz, lr, cr))

    for idx, (bz, lr, cr) in enumerate(params):
        model_para = model_para_cp
        if os.path.exists(f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy"):
            continue
        if train_load:
            model_para = load_optimal(exp_save_path, f"{algo}_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy")

        if algo == "DSGD":
            theta_D = dopt.D_SGD(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para,
                bz,
                cr,
                D_lr_dec,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr_0,
            )
        elif algo == "DRR":
            theta_D = dopt.D_RR(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para,
                bz,
                cr,
                D_lr_dec,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr_0,
            )
            
        exp_names.append(f"bz{bz}_ur{cr}_lr{lr}")
        legends.append(f"bz = {bz}, ur = {cr}, lr = {lr}")
        np.save(
            f"{exp_save_path}/{algo}_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy",
            theta_opt,
        )

        res_F_D = error_lr.cost_gap_path(theta_D, gap_type="theta")
        np.save(
            f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy",
            res_F_D,
        )
        res_F_D_F = error_lr.cost_gap_path(np.sum(theta_D, axis=1) / node_num, gap_type="F")
        np.save(
            f"{exp_save_path}/{algo}_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy",
            res_F_D_F,
        )

        if save_theta_path:
            np.save(f"{exp_save_path}/{algo}_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy", theta_D)

    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy" for idx, (bz, lr, cr) in enumerate(params)],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_{algo}_theta_{exp_name}.pdf",
        plot_every,
    )
    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy" for idx, (bz, lr, cr) in enumerate(params)],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
        plot_every,
    )
