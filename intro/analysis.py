import numpy as np
import matplotlib.pyplot as plt
import utils.python_utils as pyutils
from pprint import pprint
from tabulate import tabulate
from texttable import Texttable
import latextable

# get root directory
root = pyutils.get_root()
pingPongFolder = root / "intro" / "pingPong_times"
MMFolder = root / "intro" / "MM_times"

# check folder existence
assert pingPongFolder.exists()
assert MMFolder.exists()

# get files
pingPongFiles = list(pingPongFolder.glob("*.dat"))
MMFiles = list(MMFolder.glob("*.dat"))

# extract arrays & metadata
pingPongTimes = []
pingPongMeta = []
for file in pingPongFiles:
    pingPongTimes.append(np.fromfile(file).reshape((21, 2), order="C"))
    pingPongMeta.append(pyutils.get_metadata(file))
pingPongTimes = np.array(pingPongTimes)

MMTimes = []
MMMeta = []
for file in MMFiles:
    MMTimes.append(np.fromfile(file).reshape((4, 4), order="C"))
    MMMeta.append(pyutils.get_metadata(file))
MMTimes = np.array(MMTimes)

# plot of pingPongTimes
fig, ax = plt.subplots()
ax.set_title("Ping Pong Times")
ax.set_xlabel("Message Size (bytes)")
ax.set_ylabel("Time (s)")
ax.grid(True)

for i in range(len(pingPongTimes)):
    ax.plot(
        pingPongTimes[i, :, 0],
        pingPongTimes[i, :, 1],
        label=f"#nodes = {pingPongMeta[i]['nnodes']}",
    )

ax.legend()
# plt.show()
plt.tight_layout()
filename = f"pingPong_times.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")

# print latex table
header = [
    "matrix size",
    "processes",
    "nodes",
    "execution time (s)",
    r"\textbf{speedup}",
]
caption = "Execution time and speedup for matrix multiplication"
MMTimes_T = MMTimes.transpose(2, 0, 1)  # matrix size, processes, time
Nsizes, Nprocs, _ = MMTimes_T.shape
rows = [[]]*(Nsizes*(Nprocs+1)+1)
rows[0] = header
procs = sorted([int(MMMeta[i]["nproc"]) for i in range(len(MMMeta))])

def scientific_fmt(s: float, prec: int = 2) -> str:
    specifier = f"{{:.{prec}e}}"
    scientific_str = specifier.format(s)
    mantissa, exponent = scientific_str.split("e")
    out = mantissa + r"$\times 10^{" + exponent + "}$"
    return out

for i in range(len(MMTimes_T)):
    measurements = MMTimes_T[i]
    for j, [seq, par, speedup, size] in enumerate(measurements):
        nnodes, nproc = str(MMMeta[j]["nnodes"]), str(MMMeta[j]["nproc"])
        proc_idx = procs.index(int(nproc))+1
        if j==0:
            size_str = r"\hline" + f"{size:.0f}" + r"$\times$" + f"{size:.0f}"
            # size_str = f"{size:.0f}" + r"$\times$" + f"{size:.0f}"
            rows[i*(Nprocs+1)+1] = [size_str, 1, nnodes, seq, ""]
        row_index = i*Nprocs+proc_idx+1
        rows[i*(Nprocs+1)+proc_idx+1] = ["", nproc, nnodes, par, speedup]
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.add_rows(rows)
table.set_cols_dtype(["i", "t", "t", scientific_fmt, scientific_fmt])
label = f"tab:{header[0]}"


filename = "MM_times_table.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(latextable.draw_latex(table, caption=caption, label=label))
