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

def centralized_algo(
    logis_model,
    model_para,
    C_lr,
    C_lr_dec,
    C_batch_size,
    CEPOCH_base,
    exp_name,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
    plot_first,
    train_load,
    load_init_theta,
    init_theta_path,
    save_theta_path,
    algo,
):
    exp_save_path = f"{exp_log_path}/central_{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    if load_init_theta:
        _, model_para_cp = load_state(init_theta_path, "", type="init")
    else:
        model_para_cp = cp.deepcopy(model_para)

    params = []
    for idx, bz in enumerate(C_batch_size):
        for lr in C_lr:
            params.append((CEPOCH_base[idx], bz, lr))

    exp_names = []
    legends = []
    for idx, (epoch, bz, lr) in enumerate(params):
        exp_names.append(f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta.npy")
        legends.append(f"{algo}: bz = {bz}, lr = {lr}")
        model_para = model_para_cp

        if os.path.exists(f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta.npy"):
            print(f"Already exists {exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta.npy")
            continue
        if train_load:
            model_para = load_optimal(exp_save_path, f"{algo}_opt_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy")

        if algo == "SGD":
            theta, theta_opt, F_opt = copt.SGD(
                logis_model,
                lr,
                epoch,
                model_para,
                bz,
                C_lr_dec,
                train_log_path,
                f"{algo}_bz{bz}_lr{lr:.3f}_check",
                save_every,
                error_lr,
            )
        elif algo == "CRR":
            theta, theta_opt, F_opt = copt.C_RR(
                logis_model,
                lr,
                epoch,
                model_para,
                bz,
                C_lr_dec,
                train_log_path,
                f"{algo}_bz{bz}_lr{lr}_check",
                save_every,
                error_lr,
            )

        np.save(f"{exp_save_path}/{algo}_opt_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy", theta_opt)

        res_F = error_lr.cost_gap_path(theta, gap_type="theta")
        np.save(f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_theta.npy", res_F)
        res_F_F = error_lr.cost_gap_path(theta, gap_type="F")
        np.save(f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_F.npy", res_F_F)

        if save_theta_path:
            np.save(f"{exp_save_path}/{algo}_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}.npy", theta)

        
        
    plot_figure_path(
        exp_save_path,
        exp_names,
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (epoch, bz, lr) in enumerate(params)] + ["CGD"],
        f"{exp_save_path}/convergence_{algo}_theta_{exp_name}.pdf",
        plot_every,
        plot_first,
    )
    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_F.npy" for idx, (epoch, bz, lr) in enumerate(params)],
        line_formats,
        [f"bz = {bz}, lr = {lr}" for idx, (epoch, bz, lr) in enumerate(params)] + ["CGD"],
        f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
        plot_every,
        plot_first,
    )
    
    exp_names = [f"central_{algo}/{name}"  for name in exp_names]
    return exp_names, legends

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
    exp_name,
    exp_log_path,
    save_every,
    error_lr,
    line_formats,
    plot_every,
    plot_first,
    train_load,
    load_init_theta,
    init_theta_path,
    save_theta_path,
    algo,
):
    exp_save_path = f"{exp_log_path}/{algo}"
    initDir(exp_save_path)
    train_log_path = f"{exp_save_path}/training"
    initDir(train_log_path)

    if load_init_theta:
        _, model_para_cp = load_state(init_theta_path, "", type="init")
        model_para_cp = np.array([model_para_cp for i in range(logis_model.n)])
    else:
        model_para_cp = cp.deepcopy(model_para)

    params = []
    for idx, bz in enumerate(D_batch_size):
        for lr in D_lr:
            for cr in communication_rounds:
                params.append((DEPOCH_base[idx], bz, lr, cr))

    exp_names = []
    legends = []
    for idx, (epoch, bz, lr, cr) in enumerate(params):
        exp_names.append(f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy")
        legends.append(f"{algo}: bz = {bz}, ur = {cr}, lr = {lr}")
        model_para = model_para_cp

        if os.path.exists(f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy"):
            print(f"Already exists {exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy")
            continue
        if train_load:
            model_para = load_optimal(exp_save_path, f"{algo}_opt_theta_bz{bz}_lr{lr:.6f}_ur{cr}.npy")

        if algo == "DSGD":
            theta_D = dopt.D_SGD(
                logis_model,
                communication_matrix,
                lr,
                int(epoch),
                model_para,
                bz,
                cr,
                D_lr_dec,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr,
            )
        elif algo == "DRR":
            theta_D = dopt.D_RR(
                logis_model,
                communication_matrix,
                lr,
                int(epoch),
                model_para,
                bz,
                cr,
                D_lr_dec,
                grad_track,
                train_log_path,
                f"{algo}_bz{bz}_ur{cr}_lr{lr}",
                save_every,
                error_lr,
            )

        res_F_D = error_lr.cost_gap_path(theta_D, gap_type="theta")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_theta.npy",
            res_F_D,
        )
        res_F_D_F = error_lr.cost_gap_path(np.sum(theta_D, axis=1) / logis_model.n, gap_type="F")
        np.save(
            f"{exp_save_path}/{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy",
            res_F_D_F,
        )

        if save_theta_path:
            np.save(f"{exp_save_path}/{algo}_theta_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}.npy", theta_D)

    plot_figure_path(
        exp_save_path,
        exp_names,
        line_formats,
        legends,
        f"{exp_save_path}/convergence_{algo}_theta_{exp_name}.pdf",
        plot_every,
        plot_first,
    )
    plot_figure_path(
        exp_save_path,
        [f"{algo}_gap_epoch{epoch}_bz{bz}_lr{lr:.6f}_ur{cr}_F.npy" for idx, (epoch, bz, lr, cr) in enumerate(params)],
        line_formats,
        legends,
        f"{exp_save_path}/convergence_{algo}_F_{exp_name}.pdf",
        plot_every,
        plot_first,
    )

    exp_names = [f"{algo}/{name}"  for name in exp_names]
    return exp_names, legends