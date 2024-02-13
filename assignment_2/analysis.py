import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import python_utils.python_utils as pyutils
from matplotlib import cm

# get CLI
args_d = pyutils.get_cli_args()

# clear plots if necessary
if args_d.get("output") or __name__ == "__main__":
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

pprint(outputData)
# 3d surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
data = outputData[6]
phi = data["phi"]
ax.plot_trisurf(
    phi[:, 0],
    phi[:, 1],
    phi[:, 2],
    label=f"{i}",
    antialiased=False,
    cmap=cm.coolwarm,
    shade=True,
    linewidth=0.01,
    edgecolor="black",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("$\phi$")

plt.tight_layout()

# save plot
filename = f"fempoisson_surface.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, pad_inches=0.5, bbox_inches="tight")


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
    elif meta["type"] == "errors":
        benchmarkData[i]["errors"] = np.fromfile(file, dtype=float)

if args_d.get("output") or __name__ == "__main__":
    plt.show()
