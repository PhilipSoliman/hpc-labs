import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import python_utils as pyutils
from matplotlib import cm

# get CLI
args_d = pyutils.get_cli_args()

# clear plots if necessary
if args_d.get("output"):
    plt.close("all")

# set standard matploylib style
pyutils.set_style()

# get root directory
root = pyutils.get_root()

# get data
outputFolder = root / "assignment_2" / "output"
assert outputFolder.exists()

# output folder
outputFiles = sorted(list(outputFolder.glob("*.dat")))
outputData = {}
for i, file in enumerate(outputFiles):
    meta = pyutils.get_metadata(file)
    outputData[i] = {"meta": meta}
    number_of_vertices = int(meta["nvert"])
    outputData[i]["phi"] = np.fromfile(file, dtype=np.float64).reshape(
        number_of_vertices, 3
    )

# 3d surface plot
print("\tPlotting FEM poisson solution...", end="")
fig = plt.figure()
for i, data in outputData.items():
    grid = data["meta"]["grid"]
    adapt = int(data["meta"]["adapt"])
    if grid == "100x100" and adapt == 0:
        procg = data["meta"]["procg"]
        if procg == "2x2":
            ax = fig.add_subplot(121, projection="3d")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("$\phi$")
            ax.set_title(f"{procg}")
        elif procg == "4x1":
            ax = fig.add_subplot(122, projection="3d")
            ax.set_title(f"{procg}")
        phi = data["phi"]
        ax.plot_trisurf(
            phi[:, 0],
            phi[:, 1],
            phi[:, 2],
            antialiased=False,
            cmap=cm.coolwarm,
            shade=True,
            linewidth=0.01,
            edgecolor="black",
        )
plt.tight_layout()

# save plot
filename = f"fempoisson_surface.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=500, pad_inches=0.5, bbox_inches="tight")
print("Done!")

# benchmark folder
benchmarkFolder = root / "assignment_2" / "benchmark"
assert benchmarkFolder.exists()

# benchmark files
benchmarkFiles = sorted(list(benchmarkFolder.glob("*.dat")))
benchmarkData = {}
cols = ["computation", "exchange", "communication", "idle", "I/O"]
for i, file in enumerate(benchmarkFiles):
    meta = pyutils.get_metadata(file)
    benchmarkData[i] = {"meta": meta}
    number_of_processors = int(meta["nproc"])
    if meta["type"] == "times":
        benchmarkData[i]["times"] = np.fromfile(file, dtype=float).reshape(
            number_of_processors, len(cols)
        )
    elif meta["type"] == "error":
        benchmarkData[i]["error"] = np.fromfile(file, dtype=float)

# stack bar plot for benchmark times
print("\tPlotting FEM poisson benchmark...", end="")
grids = ["100x100", "200x200", "400x400"]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15), sharex=True)
width = 0.5

for i, data in benchmarkData.items():
    number_of_processors = int(data["meta"]["nproc"])
    x = np.arange(number_of_processors)
    bottom = np.zeros(number_of_processors)
    meta = data["meta"]
    grid = meta["grid"]
    adapt = int(meta["adapt"])
    procg = meta["procg"]
    if meta["type"] == "times" and adapt == 0:
        times = data["times"]
        if grid in grids:
            rowidx = grids.index(grid)
            if procg == "2x2":
                ax = axs[rowidx, 0]
                ax.set_title(f"{grid}", loc="left")
            elif procg == "4x1":
                ax = axs[rowidx, 1]
            for j, col in enumerate(cols):
                ax.bar(
                    x,
                    times[:, j],
                    width,
                    label=col,
                    bottom=bottom,
                )
                bottom += times[:, j]
            if rowidx == 0:
                ax.set_title(f"{procg}")

# place legend in the
axs[0, 0].legend(fontsize=10)
axs[0, 0].set_xlabel("Process rank")
axs[0, 0].set_ylabel("Time (s)")
plt.tight_layout()

# save plot
filename = f"fempoisson_benchmark.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, pad_inches=0.5, bbox_inches="tight")
print("Done!")

# computation and communication overlap estimate (fixed number of processors)
grids = {0: {}, 1: {}}
overlaps = {0: {}, 1: {}}
for i, data in benchmarkData.items():
    meta = data["meta"]
    procg = meta["procg"]
    gridsize = int(meta["grid"].split("x")[0])
    if gridsize > 100:
        continue
    adapt = int(meta["adapt"])
    if meta["type"] == "times":
        if grids[adapt].get(procg) is None:
            grids[adapt][procg] = [gridsize]
        else:
            grids[adapt][procg].append(gridsize)
        times = data["times"]
        computation = times[:, 0]
        communication = times[:, 2] + times[:, 1]
        overlap = np.sum(computation) / np.sum(communication)  # sum over all processors
        if overlaps[adapt].get(procg) is None:
            overlaps[adapt][procg] = [overlap]
        else:
            overlaps[adapt][procg].append(overlap)

print("\tPlotting FEM poisson comp./comm. overlap vs grid size (fixed #proc.)...", end="")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
for adapt, data in grids.items():
    for procg, grid in data.items():
        ax = axs[adapt]
        if adapt == 0:
            marker = "x"
            ax.set_title("Non-adaptive")
        else:
            marker = "o"
            ax.set_title("Adaptive")
        ls = ax.plot(
            grid, overlaps[adapt][procg], label=procg, marker=marker, linestyle="None"
        )

        if adapt == 0:
            # linear fit
            x = np.array(grid)
            y = np.array(overlaps[adapt][procg])
            p = np.polyfit(x, y, 1)
            y_fit = np.polyval(p, np.sort(x))
            ax.plot(np.sort(x), y_fit, linestyle="-", color=ls[0].get_color())

# labels
axs[0].plot(x, np.ones(len(x)), "--r")
axs[1].plot(x, np.ones(len(x)), "--r")
axs[0].set_xlabel("Grid size")
axs[0].set_ylabel("Overlap")
axs[0].legend()

plt.tight_layout()

# save plot
filename = f"fempoisson_overlap_vs_gridsize.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, pad_inches=0.5, bbox_inches="tight")
print("Done!")

# communication overlap estimate (fixed grid size)
pgrids = {0: [], 1: []}
overlaps = {0: [], 1: []}
for i, data in benchmarkData.items():
    meta = data["meta"]
    procg = meta["procg"]
    pgridsize = int(meta["procg"].split("x")[0])
    gridsize = int(meta["grid"].split("x")[0])
    if gridsize != 1000:
        continue
    adapt = int(meta["adapt"])
    if meta["type"] == "times":
        pgrids[adapt].append(pgridsize)
        times = data["times"]
        computation = times[:, 0]
        communication = times[:, 2] + times[:, 1]
        overlap = np.sum(computation) / np.sum(communication)  # sum over all processors
        overlaps[adapt].append(overlap)

print("\tPlotting FEM poisson comp./comm. overlap vs #proc. (fixed grid size)...", end="")
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
for adapt, pgrid in pgrids.items():
    ax = axs[adapt]
    if adapt == 0:
        marker = "x"
        ax.set_title("Non-adaptive")
    else:
        marker = "o"
        ax.set_title("Adaptive")
    ls = ax.plot(pgrid, overlaps[adapt], label=procg, marker=marker, linestyle="None")

    if adapt == 0:
        # linear fit
        x = np.array(pgrid)
        y = np.array(overlaps[adapt])
        p = np.polyfit(x, y, 1)
        y_fit = np.polyval(p, np.sort(x))
        ax.plot(np.sort(x), y_fit, linestyle="-", color=ls[0].get_color())

# labels
axs[0].plot(x, np.ones(len(x)), "--r")
axs[1].plot(x, np.ones(len(x)), "--r")
axs[0].set_xlabel("number of processes")
axs[0].set_ylabel("Overlap")

plt.tight_layout()

# save plot
filename = f"fempoisson_overlap_vs_processes.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, pad_inches=0.5, bbox_inches="tight")
print("Done!")

# error evolution plot
grids = ["100x100", "200x200", "400x400"]
pgrids = ["2x2", "4x1"]
adapt_times = np.zeros((len(grids), len(pgrids)))
nadapt_times = np.zeros((len(grids), len(pgrids)))
total_times = np.zeros((len(grids), len(pgrids)))
print("\tPlotting adapative grid refinement performance...", end="")
fig, axs = plt.subplots(
    nrows=len(grids), ncols=len(pgrids), figsize=(10, 15), sharey=True
)
for i, data in benchmarkData.items():
    meta = data["meta"]
    grid = meta["grid"]
    pgrid = meta["procg"]
    if pgrid not in pgrids:
        continue
    if grid not in grids:
        continue
    ridx = grids.index(grid)
    cidx = pgrids.index(pgrid)
    if meta["type"] == "times":
        adapt = int(meta["adapt"])
        times = data["times"]
        if adapt == 0: nadapt_times[ridx][cidx] = np.sum(times) / 4
        if adapt == 1: adapt_times[ridx][cidx] = np.sum(times) / 4
        continue
    if meta["type"] == "error":
        ridx = grids.index(grid)
        cidx = pgrids.index(pgrid)
        ax = axs[ridx][cidx]
        error = data["error"]
        if meta["adapt"] == "0":
            if cidx == 0:
                ax.plot(error, label="non-adaptive")
            else:
                ax.plot(error)
        if meta["adapt"] == "1":
            if cidx == 0:
                ax.plot(error, label="adaptive")
            else:
                ax.plot(error)
        ax.set_yscale("log")
        if cidx == 0 and ridx == 0:
            ax.set_ylabel("Error")
            ax.set_xlabel("Iterations")
            ax.legend()
        if cidx == 0:
            ax.set_title(f"{grid}", loc="left")
        if ridx == 0:
            ax.set_title(f"{pgrid}", loc="center")

# add delta time
speedup = (adapt_times - nadapt_times)/nadapt_times * 100
for i in range(len(grids)):
    for j in range(len(pgrids)):
        axs[i][j].set_title(
            r"$\frac{\Delta t}{t} = " + f"{speedup[i][j]:.2f} \%$",
            loc="right",
            fontsize=14,
        )

# save plot
filename = f"fempoisson_error_evolution.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, pad_inches=0.5, bbox_inches="tight")
print("Done!")

plt.tight_layout()
if args_d.get("output"):
    plt.show()
