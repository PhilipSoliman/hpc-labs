import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import python_utils.python_utils as pyutils

# get CLI
args_d = pyutils.get_cli_args()

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
pyutils.set_style()
fig = plt.figure()
total_subplots = len(phis)
num_cols = 2
num_rows = total_subplots // num_cols
subplot_index = num_rows * 100 + num_cols * 10 + 1
for i, phi in enumerate(phis):
    subplot_index += i
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

# show plot
if args_d.get("output") or __name__ == "__main__":
    # plt.show()
    pass

# save plot
filename = f"poisson_surface.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# optimal omega
timeFolder = root / "assignment_1" / "ppoisson_times"
assert timeFolder.exists()

timeFiles = sorted(list(timeFolder.glob("*.dat")))
omegasMeta = []
omegas = []
iters = []
times = []
grid_sizes = []
pgrid_sizes = []
for file in timeFiles:
    meta = pyutils.get_metadata(file)
    omegasMeta.append(meta)
    if meta["type"] == "omegas":
        n_x, n_y = meta["gs"].split("x")
        n_x, n_y = int(n_x), int(n_y)
        grid_sizes.append((n_x, n_y))
        omegas.append(np.fromfile(file))

    if meta["type"] == "times":
        iters.append(np.fromfile(file))
        p_x, p_y = meta["procg"].split("x")
        p_x, p_y = int(p_x), int(p_y)
        pgrid_sizes.append((p_x, p_y))

        number_of_proc = p_x * p_y
        number_of_omegas = int(meta["nomega"])

        times.append(np.fromfile(file).reshape(2, number_of_proc, number_of_omegas, order="C"))
 
    if meta["type"] == "iters":
        iters.append(np.fromfile(file))

print("Omegas:")
pprint(omegas)
print("Iters:")
pprint(iters)
print("Times:")
pprint(times)
print("Grid Sizes:")
pprint(grid_sizes)
print("PGrid Sizes:")
pprint(pgrid_sizes)
