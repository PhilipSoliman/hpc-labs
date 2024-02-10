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
outputFiles = list(output_folder.glob("*.dat"))

# extract arrays & metadata
phis = []
phisMeta = []
sizes = []
for file in outputFiles:
    meta = pyutils.get_metadata(file)
    phisMeta.append(meta)
    n_x, n_y = meta["gs"].split("x")
    n_x, n_y = int(n_x), int(n_y)
    sizes.append((n_x, n_y))
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
    n_x, n_y = sizes[i]
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
    plt.show()

# save plot
filename = f"poisson_surface.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")
