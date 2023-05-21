########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################

## Used to pre-set networkx-class properties

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def monitor(name,current,total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )

def nx_options():
    options = {
     'node_color': 'skyblue',
     'node_size': 10,
     'edge_color': 'grey',
     'width': 0.5,
     'arrows': False,
     'node_shape': 'o',}
    return options

def save_npy(npys, root_path, exp_name):

    print("saving experiment results...")
    for i, array in enumerate(npys):
        np.save(
            f"{root_path}/{exp_name[i]}.npy", array
        )

def plot_figure(
    data, formats, legend, save_path, plot_every
):
    print("plotting the figure...")

    mark_every = 1
    font = FontProperties()
    font.set_size(18)
    font2 = FontProperties()
    font2.set_size(10)
    plt.figure(3)

    for i, line in enumerate(data):
        xaxis = np.linspace(0, len(line), num=len(line)+1, dtype=int)
        plt.plot(xaxis[::plot_every], line[::plot_every], formats[i], markevery = mark_every)

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale('log')
    plt.tick_params(labelsize='large', width=3)
    plt.title('MNIST', fontproperties=font)
    plt.xlabel('Epochs', fontproperties=font)
    plt.ylabel('Optimality Gap', fontproperties=font)
    plt.savefig(save_path, format = 'pdf', dpi = 4000, bbox_inches='tight')

def save_state():

    pass

def loadPathAndPlot(
    save_path, exp_name
):
    load_thetas = []
    for i, name in enumerate(exp_name):
        load_thetas.append(np.load(f"{save_path}/{name}"))

    # theta_CGD, theta_opt, F_opt = copt.CGD(
    #     logis_model, ground_truth_lr, CEPOCH_base, model_para_central
    # ) 
    # error_lr_0 = error(
    #     logis_model, theta_opt, F_opt
    # )

