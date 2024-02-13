import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import python_utils as pyutils

# get CLI
args_d = pyutils.get_cli_args()

# clear plots if necessary
if args_d.get("output"):
    plt.close("all")

# set standard matploylib style
pyutils.set_style()

# get root directory
root = pyutils.get_root()
output_folder = root / "assignment_1" / "output"
assert output_folder.exists()

# get data file list
outputFiles = sorted(list(output_folder.glob("*.dat")))

# extract arrays & metadata
phis = []
phisMeta = []
grid_sizes = []
for file in outputFiles:
    meta = pyutils.get_metadata(file)
    phisMeta.append(meta)

    # output
    n_x, n_y = meta["gs"].split("x")
    n_x, n_y = int(n_x), int(n_y)
    grid_sizes.append((n_x, n_y))
    phis.append(np.fromfile(file).reshape((n_x, n_y), order="C"))

# plot 3D surface
fig = plt.figure()
total_subplots = len(phis)
num_cols = 2
num_rows = total_subplots // num_cols
subplot_index = num_rows * 100 + num_cols * 10
for i, phi in enumerate(phis):
    subplot_index += 1
    ax = fig.add_subplot(subplot_index, projection="3d")
    n_x, n_y = grid_sizes[i]
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, phi, cmap="viridis")
    ax.set_title(f"$\phi$ for {n_x}x{n_y} grid (procg = {phisMeta[i]['procg']})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_zlabel("Phi")
plt.tight_layout()

# save plot
filename = f"poisson_surface.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# optimal omega and (2x2)
timeFolder = root / "assignment_1" / "ppoisson_times"
assert timeFolder.exists()

timeFiles = sorted(list(timeFolder.glob("*.dat")))
Omegas_22 = np.linspace(1.90, 1.99, 10)
Grids_22 = np.flip(np.linspace(100, 1000, 10))
W_G, G_W = np.meshgrid(Grids_22, Omegas_22)
number_of_proc = 4
number_of_omegas = 10
iters = []
times = []
cpu_util = []
for i, file in enumerate(timeFiles):
    meta = pyutils.get_metadata(file)
    if meta["procg"] == "2x2":
        if meta["type"] == "times":
            benchmark = np.fromfile(file, dtype=float).reshape(
                2, number_of_proc, number_of_omegas, order="C"
            )
            times.append(np.mean(benchmark[0], axis=0, dtype=float))  # wrt processes
            cpu_util.append(np.mean(benchmark[1], axis=0, dtype=float))

        if meta["type"] == "iters":
            iters.append(np.fromfile(file, dtype=int))

# 3D plot of iters and average time (omega vs grid size)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(G_W, W_G, np.array(iters), cmap="viridis")
ax.set_title(f"iterations")
ax.set_xlabel("$\omega$")
ax.set_ylabel("gridsize")

ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(G_W, W_G, np.array(times), cmap="viridis")
ax.set_title(f"times")

plt.tight_layout()

# save plot
filename = f"optimal_omega_22.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")


# optimal omega (4x1)
iters = []
times = []
cpu_util = []
for i, file in enumerate(timeFiles):
    meta = pyutils.get_metadata(file)
    if meta["procg"] == "4x1":
        if meta["type"] == "times":
            benchmark = np.fromfile(file, dtype=float).reshape(
                2, number_of_proc, number_of_omegas, order="C"
            )
            times.append(np.mean(benchmark[0], axis=0, dtype=float))  # wrt processes
            cpu_util.append(np.mean(benchmark[1], axis=0, dtype=float))

        if meta["type"] == "iters":
            iters.append(np.fromfile(file, dtype=int))

# 3D plot of iters and average time (omega vs grid size)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(G_W, W_G, np.array(iters), cmap="viridis")
ax.set_title(f"iterations")
ax.set_xlabel("$\omega$")
ax.set_ylabel("gridsize")

ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(G_W, W_G, np.array(times), cmap="viridis")
ax.set_title(f"times")


plt.tight_layout()

# save plot
filename = f"optimal_omega_41.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# time vs iters
timeFolder = root / "assignment_1" / "timeviters"
assert timeFolder.exists()

outputFiles = sorted(list(timeFolder.glob("*.dat")))

# extract arrays & metadata
data = ()
Niters = 400
for file in outputFiles:
    meta = pyutils.get_metadata(file)
    time = np.fromfile(file, dtype=float)[:Niters][::10]
    meta["time"] = np.cumsum(time)
    data += (meta,)

total_subplots = 4
num_cols = 2
num_rows = total_subplots // num_cols
fig, axs = plt.subplots(num_rows, num_cols, squeeze=True)
stride = 5
for i, dat in enumerate(data):
    idx = i % 4
    ax = axs[idx // 2][i % 2]
    time = dat["time"]
    iters = np.linspace(0, len(time) + 1, len(time))
    if dat["procg"] == "2x2":
        ax.plot(
            iters[::stride],
            time[::stride],
            marker=".",
            color="r",
            label="2x2",
            linestyle="None",
        )
        ax.set_title(dat["gs"])
        p = np.polyfit(iters[1:], time[1:], 1)
        y_fit = np.polyval(p, iters)
        ax.plot(iters, y_fit, linestyle="--", color="r")
    if dat["procg"] == "4x1":
        ax.plot(
            iters[::stride],
            time[::stride],
            marker="x",
            color="b",
            label="4x1",
            linestyle="None",
        )
        ax.set_title(dat["gs"])
        p = np.polyfit(iters[1:], time[1:], 1)
        y_fit = np.polyval(p, iters)
        ax.plot(iters, y_fit, linestyle="--", color="b")
axs[0][0].legend()
plt.tight_layout()

filename = f"timeviters.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# error plot of 800x800
plt.close("all")
errorFolder = root / "assignment_1" / "error_analysis"
assert errorFolder.exists()

outputFiles = sorted(list(timeFolder.glob("*.dat")))
data = []
Niters = 4300
for file in outputFiles:
    meta = pyutils.get_metadata(file)
    if meta["gs"] == "800x800" and meta["procg"] == "4x1":
        meta["error"] = np.fromfile(file, dtype=float)[Niters:]
        data.append(meta)

fig = plt.figure()
ax = fig.add_subplot(111)
for i, dat in enumerate(data):
    error = dat["error"][1:]
    ax.plot(error, label=dat["procg"])
ax.set_title("Residual error for 800x800 grid")
plt.tight_layout()

filename = f"error_800x800.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# sweep analysis
sweepFolder = root / "assignment_1" / "sweep_analysis"
assert timeFolder.exists()

outputFiles = sorted(list(sweepFolder.glob("*.dat")))
sweepData = {}
idx = 0
for i, file in enumerate(outputFiles):
    if i % 3 == 0:
        idx += 1
        sweepData[idx] = {}
    meta = pyutils.get_metadata(file)
    if meta["type"] == "iters":
        sweepData[idx]["meta"] = meta
        sweepData[idx]["iters"] = np.fromfile(file, dtype=int)
    if meta["type"] == "times":
        sweepData[idx]["times"] = np.fromfile(file, dtype=float)
    if meta["type"] == "sweeps":
        sweepData[idx]["sweeps"] = np.fromfile(file, dtype=int)

fig, axs = plt.subplots(1, 2, squeeze=True)
for i, dat in sweepData.items():
    ax = axs[0]
    ax.plot(dat["sweeps"], dat["iters"], label=dat["meta"]["procg"])
    ax.set_title("Iterations")
    ax.set_xlabel("Sweeps")
    ax = axs[1]
    ax.plot(dat["sweeps"], dat["times"] / dat["iters"], label=dat["meta"]["procg"])
    ax.set_title("time/iteration")

filename = f"sweep_analysis.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# latency analysis
latencyFolder = root / "assignment_1" / "latency_analysis"
assert latencyFolder.exists()

outputFiles = sorted(list(latencyFolder.glob("*.dat")))
latencyData = {}
pgrids = []
for i, file in enumerate(outputFiles):
    latencyData[i] = {}
    meta = pyutils.get_metadata(file)
    latencyData[i]["meta"] = meta
    pgrid = meta["procg"]
    if pgrid not in pgrids:
        pgrids.append(pgrid)
    p_x = int(pgrid.split("x")[0])
    p_y = int(pgrid.split("x")[1])
    number_of_proc = p_x * p_y
    latency = np.fromfile(file, dtype=float)
    total_length = len(latency)
    number_of_iters = total_length // (2 * number_of_proc)
    latency = latency.reshape(2, number_of_proc, number_of_iters, order="C")
    overhead_avg = np.mean(latency[0], axis=0)
    bytes_avg = np.mean(latency[1], axis=0)
    latencyData[i]["overhead"] = overhead_avg
    latencyData[i]["bytes"] = bytes_avg

# plot moving avarage bandwidth for several grid sizes
gridsizes = ["100x100", "200x200", "400x400", "800x800"]
fig, axs = plt.subplots(2, 2, squeeze=True, sharey=True)
number_of_pgrids = len(pgrids)
number_of_grids = len(gridsizes)
# window_size = 20
for i, dat in latencyData.items():
    gridsize = dat["meta"]["gs"]
    if gridsize in gridsizes:
        depth = i // number_of_grids
        I = gridsizes.index(gridsize)
        row_idx = I // 2
        col_idx = I % 2
        ax = axs[row_idx][col_idx]
        number_of_iters = len(dat["overhead"])
        window_size = number_of_iters // 10  # ~10% of the iterations
        bandwith = pyutils.moving_average(dat["bytes"] / dat["overhead"], window_size)
        ax.plot(bandwith * 1e-9, label=dat["meta"]["procg"])
        ax.set_title(gridsize)

axs[0][0].set_xlabel("Iterations")
axs[0][0].set_ylabel("Bandwidth (GB/s)")
axs[0][0].legend(fontsize=10)
fig.suptitle("Moving average of bandwith")
plt.tight_layout()

# save plot
filename = f"latency_analysis.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# clear plots if necessary
if args_d.get("output"):
    plt.show()
