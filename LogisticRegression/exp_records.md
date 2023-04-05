
``` python
# C_SGD sanity check using different batch sizes
def CSGD_check():
    all_res_F_SGD = []
    batch_sizes = [2000, 3000, 4000, 6000, 12000]
    for bz in batch_sizes:
        theta_SGD, theta_opt, F_opt = copt.SGD(
            logis_model, lr, CEPOCH_base, model_para_central, bz
        ) 
        res_F_SGD = error_lr_0.cost_gap_path(theta_SGD)
        all_res_F_SGD.append(res_F_SGD)

    exp_save_path = f"{exp_log_path}/central_SGD"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(
        all_res_F_SGD, exp_save_path,
        [f"bz{bz}" for bz in batch_sizes]
    )
    plot_figure(
        all_res_F_SGD, line_formats, 
        [f"bz = {bz}" for bz in batch_sizes],
        f"{exp_save_path}/convergence.pdf",
        plot_every
    )
```

```python
# CRR sanity check using different batch sizes
def CRR_check():
    all_res_F_CRR = []
    batch_sizes = [2000, 3000, 4000, 6000, 12000]
    for bz in batch_sizes:
        theta_CRR, theta_opt, F_opt = copt.C_RR(
            logis_model, lr, CEPOCH_base, model_para_central, bz
        ) 
        res_F_CRR = error_lr_0.cost_gap_path(theta_CRR)
        all_res_F_CRR.append(res_F_CRR)

    exp_save_path = f"{exp_log_path}/central_CRR"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(
        all_res_F_CRR, exp_save_path,
        [f"bz{bz}" for bz in batch_sizes]
    )
    plot_figure(
        all_res_F_CRR, line_formats, 
        [f"bz = {bz}" for bz in batch_sizes],
        f"{exp_save_path}/convergence.pdf",
        plot_every
    )
```

```python
def DSGD_check():
    all_res_F_DSGD = []
    batch_sizes = [1000, 2000, 3000]
    for bz in batch_sizes:
        theta_D_SGD = dopt.D_SGD(
            logis_model, communication_matrix, step_size, int(DEPOCH_base), model_para_dis, bz, 1
        )  
        res_F_D_SGD = error_lr_0.cost_gap_path( np.sum(theta_D_SGD,axis = 1)/node_num)

        all_res_F_DSGD.append(res_F_D_SGD)

    exp_save_path = f"{exp_log_path}/central_DSGD"
    if not os.path.exists(exp_save_path):
        os.mkdir(exp_save_path)

    save_npy(
        all_res_F_DSGD, exp_save_path,
        [f"bz{bz}" for bz in batch_sizes]
    )
    plot_figure(
        all_res_F_DSGD, line_formats, 
        [f"bz = {bz}" for bz in batch_sizes],
        f"{exp_save_path}/convergence.pdf",
        plot_every
    )
```


