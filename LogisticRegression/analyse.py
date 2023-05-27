import numpy as np
from utilities import plot_figure, save_npy, load_optimum, load_state, loadPathAndPlot, loadGapAndPlot
from Problems.logistic_regression import LR_L2

# exp_path = "/home/ubuntu/Desktop/DRR/experiment/debug"


# node_num = 4  ## number of nodes
# logis_model = LR_L2(
#     node_num, limited_labels=False, balanced=True
# )  ## instantiate the problem class

# error_lr_0 = load_optimum(logis_model, exp_path, "optimum")
# loadPathAndPlot(exp_path, ["optimum"], error_lr_0, 2000)


exp_path = "/home/ubuntu/Desktop/DRR/experiment/lr_bz_ratio"

batch_size = [50, 200, 500]
update_round = [i for i in range(1, 2)]
exp_names = []
legends = []
for idx, bz in enumerate(batch_size):
    for ur in update_round:
        exp_names.append(f"central_SGD/bz{bz}")
        exp_names.append(f"central_CRR/bz{bz}")
        exp_names.append(f"DSGD/bz{bz}_ur{ur}")
        exp_names.append(f"DRR/bz{bz}_ur{ur}")
        legends.append(f"SGD, bz={bz}")
        legends.append(f"CRR, bz={bz}")
        legends.append(f"DSGD, bz={bz} ur = {ur}")
        legends.append(f"DRR, bz={bz} ur = {ur}")

loadGapAndPlot(exp_path, exp_names, legends, 500, "fixed_lr")


