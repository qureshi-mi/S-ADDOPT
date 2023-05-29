## Generates all the plots to compare different algorithms over Exponential directed graphs using logistic regression.

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
from utilities import initDir, save_npy, plot_figure, load_state

"""
Data processing for MNIST
"""
node_num = 16  ## number of nodes
side_length = 4  ## side length of the square
logis_model = LR_L2(
    node_num, limited_labels=False, balanced=True
)  ## instantiate the problem class
dim = logis_model.p  ## dimension of the model
L = logis_model.L  ## L-smooth constant
total_train_sample = logis_model.N  ## total number of training samples
avg_local_sample = logis_model.b  ## average number of local samples
step_size = 1 / L / 2  ## selecting an appropriate step-size

"""
Initializing variables
"""
CEPOCH_base = 10000
DEPOCH_base = 10000

model_para_central = np.random.normal(0, 1, dim)
model_para_dis = np.random.normal(0, 1, (node_num, dim))
undir_graph = Grid_graph(side_length).undirected()
communication_matrix = Weight_matrix(undir_graph).column_stochastic()

C_lr = [0.0001 / L]
D_lr = [0.0001 / L]  ## selecting an appropriate step-size
C_batch_size = [100]
D_batch_size = [100]

line_formats = [
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
exp_log_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/grid_ur"
ckp_load_path = "/afs/andrew.cmu.edu/usr7/jiaruil3/private/DRR/experiments/optimum"
plot_every = 250
save_every = 2000

"""
Optimum solution
"""
if ckp_load_path is not None:
    theta_CSGD_0, theta_opt = load_state(ckp_load_path, "optimum")
theta_CSGD_0 = None
error_lr_0 = error(logis_model, theta_opt, logis_model.F_val(theta_opt))


def CSGD_check(
    logis_model,
    model_para,
    C_lr,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/central_SGD"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    all_res_F_SGD = []
    params = []
    for bz in C_batch_size:
        for lr in C_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        theta_SGD, theta_opt, F_opt = copt.SGD(
            logis_model,
            lr,
            CEPOCH_base,
            model_para,
            bz,
            train_log_path,
            f"CSGD_bz{bz}_lr{lr:.3f}_check",
            save_every,
        )
        res_F_SGD = error_lr.cost_gap_path(theta_SGD)
        all_res_F_SGD.append(res_F_SGD)

    save_npy(
        all_res_F_SGD,
        exp_save_path,
        [f"bz{bz}_lr{lr:.3f}" for idx, (bz, lr) in enumerate(params)],
    )
    plot_figure(
        all_res_F_SGD,
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_SGD.pdf",
        plot_every,
    )


def CRR_check(
    logis_model,
    model_para,
    C_lr,
    C_batch_size,
    CEPOCH_base,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/central_CRR"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    all_res_F_CRR = []
    params = []
    for bz in C_batch_size:
        for lr in C_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        theta_CRR, theta_opt, F_opt = copt.C_RR(
            logis_model,
            lr,
            CEPOCH_base,
            model_para,
            bz,
            train_log_path,
            f"CRR_bz{bz}_lr{lr}_check",
            save_every,
        )
        res_F_CRR = error_lr.cost_gap_path(theta_CRR)
        all_res_F_CRR.append(res_F_CRR)

    save_npy(
        all_res_F_CRR,
        exp_save_path,
        [f"bz{bz}_lr{lr}" for idx, (bz, lr) in enumerate(params)],
    )
    plot_figure(
        all_res_F_CRR,
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (bz, lr) in enumerate(params)],
        f"{exp_save_path}/convergence_CRR.pdf",
        plot_every,
    )


"""
Decentralized Algorithms
"""


def DSGD_check(
    logis_model,
    model_para,
    D_lr,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/DSGD"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    all_res_F_DSGD = []
    update_rounds = [1, 2, 5, 10, 15]
    exp_names = []
    legends = []
    params = []
    for bz in D_batch_size:
        for lr in D_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        for comm_round in update_rounds:
            theta_D_SGD = dopt.D_SGD(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para,
                bz,
                comm_round,
                train_log_path,
                f"DSGD_bz{bz}_ur{comm_round}_lr{lr}",
                save_every,
            )

            res_F_D_SGD = error_lr.cost_gap_path(
                np.sum(theta_D_SGD, axis=1) / logis_model.n
            )

            all_res_F_DSGD.append(res_F_D_SGD)
            exp_names.append(f"bz{bz}_ur{comm_round}_lr{lr}")
            legends.append(f"bz = {bz}, ur = {comm_round}, lr = {lr}")

    save_npy(all_res_F_DSGD, exp_save_path, exp_names)
    plot_figure(
        all_res_F_DSGD,
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DSGD.pdf",
        plot_every,
    )


def DRR_check(
    logis_model,
    model_para,
    D_lr,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
):
    exp_save_path = f"{exp_log_path}/DRR"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    all_res_F_DRR = []
    update_rounds = [1, 2, 5, 10, 15]
    exp_names = []
    legends = []
    params = []
    for bz in D_batch_size:
        for lr in D_lr:
            params.append((bz, lr))

    for idx, (bz, lr) in enumerate(params):
        for comm_round in update_rounds:
            comm_round = int(comm_round)
            theta_D_RR = dopt.D_RR(
                logis_model,
                communication_matrix,
                lr,
                int(DEPOCH_base),
                model_para,
                bz,
                comm_round,
                train_log_path,
                f"DRR_bz{bz}_ur{comm_round}_lr{lr}",
                save_every,
            )
            res_F_D_RR = error_lr.cost_gap_path(
                np.sum(theta_D_RR, axis=1) / logis_model.n
            )

            all_res_F_DRR.append(res_F_D_RR)
            exp_names.append(f"bz{bz}_ur{comm_round}_lr{lr}")
            legends.append(f"bz = {bz}, ur = {comm_round}, lr = {lr}")

    save_npy(all_res_F_DRR, exp_save_path, exp_names)
    plot_figure(
        all_res_F_DRR,
        line_formats,
        legends,
        f"{exp_save_path}/convergence_DRR.pdf",
        plot_every,
    )


# CSGD_check(
#     logis_model,
#     model_para_central,
#     C_lr,
#     C_batch_size,
#     CEPOCH_base,
#     exp_log_path,
#     save_every,
#     error_lr_0,
#     line_formats,
#     plot_every,
# )
# CRR_check(
#     logis_model,
#     model_para_central,
#     C_lr,
#     C_batch_size,
#     CEPOCH_base,
#     exp_log_path,
#     save_every,
#     error_lr_0,
#     line_formats,
#     plot_every,
# )

DSGD_check(
    logis_model,
    model_para_dis,
    D_lr,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)
DRR_check(
    logis_model,
    model_para_dis,
    D_lr,
    D_batch_size,
    DEPOCH_base,
    communication_matrix,
    exp_log_path,
    save_every,
    error_lr_0,
    line_formats,
    plot_every,
)
