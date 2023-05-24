import numpy as np
from utilities import plot_figure, save_npy, load_optimum, load_state, loadPathAndPlot
from Problems.logistic_regression import LR_L2

exp_path = "/home/ubuntu/Desktop/DRR/experiment/debug"


node_num = 4  ## number of nodes
logis_model = LR_L2(
    node_num, limited_labels=False, balanced=True
)  ## instantiate the problem class

error_lr_0 = load_optimum(logis_model, exp_path, "optimum")
loadPathAndPlot(exp_path, ["optimum"], error_lr_0, 2000)

