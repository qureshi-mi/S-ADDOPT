import numpy as np
from utilities import plot_figure, save_npy

root_path = "/Users/ultrali/Documents/Experiments/DRR/sanity-check/central_SGD"
batch_sizes = [100, 2000]
exp_res_paths = [
    f"{root_path}/bz{bz}.npy" for bz in batch_sizes
]


line_formats = [
    '-vb', '-^m', '-dy', '-sr', "-1k", "-2g", "-3C", "-4w"
]
plot_every = 1

all_res = []
for path in exp_res_paths:
    all_res.append(
        np.load(path)
    )

plot_figure(
    all_res, line_formats, 
    [f"bz = {bz}" for bz in batch_sizes],
    f"{root_path}/test.pdf",
    plot_every
)

# for errors in all_res:
#     last_n = 20
#     print(f"floor = {np.sum( errors[-last_n:] ) / last_n}")