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
from graph import *


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


def plot_figure_path(
    exp_save_path, exp_names, formats, legend, save_path, plot_every, plot_first
):
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
        xaxis = np.linspace(0, len(line) - 1, num=len(line), dtype=int)
        yaxis = [
            abs(point) for point in line[:plot_first:plot_every]
        ]  # the F_val could be negative
        if i >= len(formats):
            plt.plot(xaxis[:plot_first:plot_every], yaxis, markevery=mark_every)
        else:
            plt.plot(
                xaxis[:plot_first:plot_every], yaxis, formats[i], markevery=mark_every
            )

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
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
        xaxis = np.linspace(0, len(line) - 1, num=len(line), dtype=int)
        yaxis = [
            abs(point) for point in line[::plot_every]
        ]  # the F_val could be negative
        if i >= len(formats):
            plt.plot(xaxis[::plot_every], yaxis, markevery=mark_every)
        else:
            plt.plot(xaxis[::plot_every], yaxis, formats[i], markevery=mark_every)

    plt.legend(legend, prop=font2)
    plt.grid(True)
    plt.yscale("log")
    plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])
    plt.tick_params(labelsize="large", width=3)
    plt.title("MNIST", fontproperties=font)
    plt.xlabel("Epochs", fontproperties=font)
    plt.ylabel("Optimality Gap", fontproperties=font)
    plt.savefig(save_path, format="pdf", dpi=4000, bbox_inches="tight")
    print("figure plotted...")


def save_state(theta, save_path, exp_name):
    print("saving experiment results...")
    np.save(f"{save_path}/{exp_name}_theta", theta)


def load_state(save_path, exp_name, type="optimal"):
    print("loading experiment results...")
    if type == "path":
        theta = np.load(f"{save_path}/{exp_name}_theta_path.npy")
        return (
            theta,
            theta[-1],
        )  # return the state sequence and the last state (optimum)
    elif type == "optimal":
        theta = np.load(f"{save_path}/{exp_name}_theta_optimal.npy")
        return None, theta  # return the last state (optimum)
    elif type == "init":
        theta = np.load(save_path)
        return None, theta


def load_optimal(save_path, exp_name):
    return np.load(f"{save_path}/{exp_name}")


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
        plot_every,
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
        [
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
        ],
        legands,
        f"{save_path}/{fig_name}.pdf",
        plot_every,
    )


def load_optimum(pr, save_path, exp_name):
    theta, theta_opt = load_state(save_path, exp_name)
    error_lr = error(pr, theta[-1], pr.F_val(theta[-1]))
    return error_lr


def initDir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def convert_to_doubly_stochastic(matrix, max_iterations=100, tolerance=1e-6):
    """
    The Sinkhorn-Knopp algorithm.

    Converts the given matrix to a doubly stochastic matrix.
    The conversion is done in-place.

    @param matrix: The matrix to be converted.
    @param max_iterations: The maximum number of iterations to perform.
    @param tolerance: The tolerance to check convergence.
    @return: The given matrix, converted to doubly stochastic.
    """
    n = matrix.shape[0]  # Assuming matrix is a square matrix

    # Normalize rows and columns iteratively
    for _ in range(max_iterations):
        prev_matrix = matrix.copy()

        # Normalize rows
        row_sums = np.sum(matrix, axis=1)
        matrix /= row_sums[:, np.newaxis]

        # Normalize columns
        col_sums = np.sum(matrix, axis=0)
        matrix /= col_sums

        # Check convergence
        max_diff = np.max(np.abs(matrix - prev_matrix))
        if max_diff < tolerance:
            break

    return matrix


def spectral_norm(comm_matrix):
    # ones matrix
    ones = np.ones(comm_matrix.shape)
    matrix = comm_matrix - ones / comm_matrix.shape[0]

    # Compute the matrix A^T * A
    ata = np.matmul(matrix.T, matrix)

    # Calculate the eigenvalues of A^T * A
    eigenvalues, _ = np.linalg.eig(ata)

    # Find the maximum eigenvalue
    max_eigenvalue = np.max(eigenvalues)

    # Calculate the spectral norm as the square root of the maximum eigenvalue
    spectral_norm = np.sqrt(max_eigenvalue)

    return spectral_norm


def print_matrix(matrix, name):
    print(f"{name} Matrix:")
    for row in matrix:
        for element in row:
            if element < 0:
                print(f"{element:.3f}", end=" ")
            else:
                print(f" {element:.3f}", end=" ")
        print()
    print()


def is_primitive(matrix, max_iterations=10000):
    n = matrix.shape[0]  # Assuming matrix is a square matrix
    power_counter = 1

    while power_counter <= max_iterations:
        power_matrix = np.linalg.matrix_power(matrix, power_counter)
        if np.any(power_matrix <= 0):
            return False
        elif np.all(power_matrix > 0):
            return True
        power_counter += 1

    return False


def convergence_analysis():
    lr_constant = False
    epoch = 1000

    m = {
        "centralized": 12000,
        "distributed": 75,
    }
    n = {
        "centralized": 1,
        "distributed": 16,
    }

    net_lambda = {
        "grid": 0.8629819803720482,
        "exp": 0.5000000000000003,
        "geo": 0.43760846131515274,
    }

    algos = ["SGD", "CRR", "DSGD", "DRR", "DSGT", "GT-RR"]

    if lr_constant:
        # Constant learning rate
        lr = 1 / 8000

        for algo in algos:
            if algo == "SGD":
                error_floor = lr / n["centralized"]
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "CRR":
                error_floor = lr**2 * m["centralized"]
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "DSGD":
                for net, lambd in net_lambda.items():
                    error_floor = lr / n["distributed"] + lr**2 / (1 - lambd) ** 2
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DRR":
                for net, lambd in net_lambda.items():
                    error_floor = lr**2 * m["distributed"] / (1 - lambd) ** 3
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DSGT":
                for net, lambd in net_lambda.items():
                    error_floor = (
                        lr / n["distributed"]
                        + lr**2 / (1 - lambd)
                        + lr**4 / (n["distributed"] * (1 - lambd) ** 4)
                    )
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "GT-RR":
                for net, lambd in net_lambda.items():
                    error_floor = (
                        lr**2 * m["distributed"] / (1 - lambd)
                        + lr**4 * m["distributed"] ** 4 / (1 - lambd) ** 2
                    )
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

    else:
        # Decreasing learning rate
        lr = lambda t: 1 / (50 * t + 400)

        for algo in algos:
            if algo == "SGD":
                error_floor = 1 / (m["centralized"] * n["centralized"] * epoch)
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "CRR":
                error_floor = np.log(epoch) / (m["centralized"] * epoch**2)
                print(f"{algo:<5}: Error floor: {error_floor}")

            elif algo == "DSGD":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        m["distributed"] * n["distributed"] * epoch
                    ) + 1 / ((1 - lambd) ** 2 * m["distributed"] ** 2 * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DRR":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / ((1 - lambd) ** 3 * m["distributed"] * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "DSGT":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        m["distributed"] * n["distributed"] * epoch
                    ) + 1 / ((1 - lambd) ** 3 * m["distributed"] ** 2 * epoch**2)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")

            elif algo == "GT-RR":
                for net, lambd in net_lambda.items():
                    error_floor = 1 / (
                        (1 - lambd) * m["distributed"] * epoch**2
                    ) + 1 / ((1 - lambd) ** 2 * epoch**4)
                    print(f"{algo:<5}: Error floor: {error_floor} ({net})")


def try_geo(save_path):
    start = time.time()

    np.random.seed(0)
    nodes_list = [8, 16, 32]
    matrices_lambda = []
    search_space = int(1e5)

    print("generating matrices...", flush=True)
    for node_num in nodes_list:
        print(f"node_num = {node_num}", flush=True)
        mat = []
        for i in range(search_space):
            undir_graph = Geometric_graph(node_num).undirected(0.8)

            communication_matrix = Weight_matrix(undir_graph).row_stochastic()
            communication_matrix = convert_to_doubly_stochastic(
                communication_matrix, int(1e4), 1e-7
            )

            norm = spectral_norm(communication_matrix)
            mat.append((node_num, communication_matrix, norm))

            if i % (search_space / 10) == 0:
                print(f"{i / search_space * 100}% completed", flush=True)

        matrices_lambda.append(mat)
    print("time elapsed: ", time.time() - start, flush=True)

    print("searching for min distance... (1)", flush=True)
    min_dist = 1e10
    min_dist_tup = None
    min_dist_list = []
    for idx1, mat1 in enumerate(matrices_lambda[0]):
        for idx2, mat2 in enumerate(matrices_lambda[1]):
            dist = abs(mat1[2] - mat2[2])
            if dist < min_dist:
                min_dist = dist
                min_dist_tup = (mat1, mat2)
                min_dist_list.append((mat1, mat2))
        if idx1 % (search_space / 10) == 0:
            print(f"{idx1 / search_space * 100}% completed", flush=True)

    print(f"min_dist = {min_dist} (1)", flush=True)
    print(f"lambdas = {min_dist_tup[0][2]}, {min_dist_tup[1][2]}", flush=True)
    print("time elapsed: ", time.time() - start, flush=True)

    print("searching for min distance... (2)", flush=True)
    min_dist = 1e10
    min_dist_tup = None
    min_dist_list_2 = []
    for idx3, mat3 in enumerate(matrices_lambda[2]):
        dist1 = abs(mat3[2] - min_dist_list[-1][0][2])
        dist2 = abs(mat3[2] - min_dist_list[-1][1][2])
        dist = min(dist1, dist2)
        if dist < min_dist:
            min_dist = dist
            min_dist_tup = (*min_dist_list[-1], mat3)
            min_dist_list_2.append(min_dist_tup)
        if idx3 % (search_space / 10) == 0:
            print(f"{idx3 / search_space * 100}% completed", flush=True)

    print(f"min_dist = {min_dist} (2)", flush=True)
    print(
        f"lambdas = {min_dist_tup[0][2]}, {min_dist_tup[1][2]}, {min_dist_tup[2][2]}",
        flush=True,
    )
    print("time elapsed: ", time.time() - start, flush=True)

    # save the closest 100 matrices
    print("saving matrices...", flush=True)
    if not os.path.exists(f"{save_path}/geo"):
        os.mkdir(f"{save_path}/geo")

    for idx, tup in enumerate(min_dist_list_2[-min(100, len(min_dist_list_2)) :]):
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[0]}", tup[0][1])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[1]}", tup[1][1])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[2]}", tup[2][1])

    # save the closest 100 lambdas
    print("saving lambdas...", flush=True)
    for idx, tup in enumerate(min_dist_list_2[-min(100, len(min_dist_list_2)) :]):
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[0]}_lambda", tup[0][2])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[1]}_lambda", tup[1][2])
        np.save(f"{save_path}/geo/geo_{idx}_node{nodes_list[2]}_lambda", tup[2][2])

    print(f"Time elapsed: {time.time() - start}", flush=True)
    print("done", flush=True)


def init_comm_matrix(node_num, graph, load_path=None):
    if load_path is None:
        if graph == "exponential":
            undir_graph = Exponential_graph(
                node_num
            ).undirected()  # generate the undirected graph
        elif graph == "grid":
            undir_graph = Grid_graph(int(math.sqrt(node_num))).undirected()
        elif graph == "geometric":
            undir_graph = Geometric_graph(node_num).undirected(0.8)
        elif graph == "fully_connected":
            undir_graph = Fully_connected_graph(node_num).undirected()

        # communication_matrix = Weight_matrix(
        #     undir_graph
        # ).column_stochastic()  # generate the communication matrix
        communication_matrix = Weight_matrix(undir_graph).row_stochastic()
        communication_matrix = convert_to_doubly_stochastic(
            communication_matrix, int(1e4), 1e-7
        )
    else:
        communication_matrix = np.load(load_path)
        print(f"loaded communication matrix from {load_path}")

    return communication_matrix
