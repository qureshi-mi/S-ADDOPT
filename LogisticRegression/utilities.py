########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################

## Used to pre-set networkx-class properties

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from analysis import error
import os


def monitor(name, current, total, start_time):
    if (current + 1) % (total / 10) == 0:
        print(name + " %d%% completed" % int(100 * (current + 1) / total), flush=True)
        print(f"Time Span: {time.time() - start_time}", flush=True)
        return time.time()
    else:
        return start_time


def nx_options():
    options = {
        "node_color": "skyblue",
        "node_size": 10,
        "edge_color": "grey",
        "width": 0.5,
        "arrows": False,
        "node_shape": "o",
    }
    return options


def save_npy(npys, root_path, exp_name):
    print("saving experiment results...")
    for i, array in enumerate(npys):
        np.save(f"{root_path}/{exp_name[i]}.npy", array)


def plot_figure_path(exp_save_path, exp_names, formats, legend, save_path, plot_every):
    print("plotting the figure...", flush=True)
    plt.clf()
    mark_every = 1
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    for i, name in enumerate(exp_names):
        line = np.load(f"{exp_save_path}/{name}")
        xaxis = np.linspace(0, len(line)-1, num=len(line), dtype=int)
        yaxis = [abs(point) for point in  line[::plot_every]]   # the F_val could be negative
        if i >= len(formats):
            plt.plot(
                xaxis[::plot_every], yaxis, markevery=mark_every
            )
        else:
            plt.plot(
                xaxis[::plot_every], yaxis, formats[i], markevery=mark_every
            )

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.tick_params(labelsize="large", width=3)
    plt.title("MNIST", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Optimality Gap", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")

def plot_figure_data(data, formats, legend, save_path, plot_every):
    print("plotting the figure...", flush=True)
    plt.clf()
    mark_every = 1
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    for i, line in enumerate(data):
        xaxis = np.linspace(0, len(line)-1, num=len(line), dtype=int)
        yaxis = [abs(point) for point in  line[::plot_every]]   # the F_val could be negative
        if i >= len(formats):
            plt.plot(
                xaxis[::plot_every], yaxis, markevery=mark_every
            )
        else:
            plt.plot(
                xaxis[::plot_every], yaxis, formats[i], markevery=mark_every
            )

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.tick_params(labelsize="large", width=3)
    plt.title("MNIST", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Optimality Gap", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")

def save_state(theta, save_path, exp_name):
    print("saving experiment results...")
    np.save(f"{save_path}/{exp_name}_theta", theta)


def load_state(save_path, exp_name):
    print("loading experiment results...")
    theta = np.load(f"{save_path}/{exp_name}_theta.npy")
    return theta, theta[-1]  # return the state sequence and the last state (optimum)

def load_optimal(save_path, exp_name):
    return np.load(save_path, exp_name)
    
def loadPathAndPlot(save_path, exp_names, error_lr, plot_every):
    load_thetas = []
    print("loading theta training results...")
    for i, name in enumerate(exp_names):
        load_thetas.append(np.load(f"{save_path}/{name}_theta.npy"))
    gaps = [error_lr.cost_gap_path(theta) for theta in load_thetas]

    print("plotting error gap results...")
    plot_figure(
        gaps,
        ["-vb", "-^m", "-dy", "-sr", "-1k", "-2g", "-3C", "-4w"],
        [f"optimum{i}" for i in range(len(exp_names))],
        f"{save_path}/cost_gap_all.pdf",
        plot_every
    )

def loadGapAndPlot(save_path, exp_names, legands, plot_every, fig_name):
    gaps = []
    print("loading error gap results...")
    for i, name in enumerate(exp_names):
        gaps.append(np.load(f"{save_path}/{name}.npy"))
    
    # gaps = [gap[:5000] for gap in gaps]

    print("plotting error gap results...")
    plot_figure(
        gaps,
        ["-vb", "-^m", "-dy", "-sr", "-1k", "-2g", "-3r", "-.kp", "-+c", "-xm", "-|y", "-_r"],
        legands,
        f"{save_path}/{fig_name}.pdf",
        plot_every
    )

def load_optimum(pr, save_path, exp_name):
    theta, theta_opt = load_state(save_path, exp_name)
    error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
    return error_lr

def initDir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

