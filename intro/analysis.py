import sys
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import python_utils as pyutils
from tabulate import tabulate
from texttable import Texttable
import latextable


# get CLI
args_d = pyutils.get_cli_args()
if args_d.get("output"):
    plt.close("all")

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

MMTimes = []
MMMeta = []
for file in MMFiles:
    MMTimes.append(np.fromfile(file).reshape((4, 4), order="C"))
    MMMeta.append(pyutils.get_metadata(file))
MMTimes = np.array(MMTimes)

# plot of pingPongTimes
fig, ax = plt.subplots()
ax.set_title("Ping Pong Times")
ax.set_xlabel("Message Size (MB)")
ax.set_ylabel("Time (s)")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(True)

print("\tMaking plot of ping pong times...", end="")
for i in range(len(pingPongTimes)):
    x = pingPongTimes[i][7:, 0] * 1e-6
    y = pingPongTimes[i][7:, 1]

    lines = ax.plot(
        x, y, label=f"#nodes = {pingPongMeta[i]['nnodes']}", marker="x", linestyle=""
    )

    # numpy polyfit
    p = np.polyfit(x, y, 1)
    y_fit = np.polyval(p, x)
    ax.plot(x, y_fit, label="fit", linestyle="--", color=lines[0].get_color())

    msg = r"$\alpha$ = " + f"{p[1]:.2e}\n" + r"$\beta$ = " + f"{p[0]*1e6:.2e}"
    ax.text(x[-2], y[-2], msg, fontsize=13, ha="right")

ax.legend()
plt.tight_layout()

filename = f"pingPong_times.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")
print("Done!")

# plot of pingPongTimes (small message sizes)
print("\tMaking plot of ping pong times (small message sizes)...", end="")
fig, ax = plt.subplots()
ax.set_title("Ping Pong Times")
ax.set_xlabel("Message Size (B)")
ax.set_ylabel("Time (s)")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(True)

for i in range(len(pingPongTimes)):
    x = pingPongTimes[i][1:7, 0]
    y = pingPongTimes[i][1:7, 1]

    lines = ax.plot(
        x, y, label=f"#nodes = {pingPongMeta[i]['nnodes']}", marker="x", linestyle=""
    )

    # numpy polyfit
    p = np.polyfit(x, y, 1)
    y_fit = np.polyval(p, x)
    ax.plot(x, y_fit, label="fit", linestyle="--", color=lines[0].get_color())

    msg = r"$\alpha$ = " + f"{p[1]:.2e}\n" + r"$\beta$ = " + f"{p[0]:.2e}"
    ax.text(x[-2], y[-2], msg, fontsize=13, ha="right")


ax.legend()
plt.tight_layout()
filename = f"pingPong_times_small.png"
filepath = root / "report" / "figures" / filename
fig.savefig(filepath, dpi=300, bbox_inches="tight")
print("Done!")

# print latex table
print("\tMaking table of matrix-matrix multiplication times...", end="")
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
rows = [[]] * (Nsizes * (Nprocs + 1) + 1)
rows[0] = header
procs = sorted([int(MMMeta[i]["nproc"]) for i in range(len(MMMeta))])


for i in range(len(MMTimes_T)):
    measurements = MMTimes_T[i]
    for j, [seq, par, speedup, size] in enumerate(measurements):
        nnodes, nproc = MMMeta[j]["nnodes"].split(".")[0], MMMeta[j]["nproc"]
        proc_idx = procs.index(int(nproc)) + 1
        if j == 0:
            size_str = r"\hline" + f"{size:.0f}" + r"$\times$" + f"{size:.0f}"
            # size_str = f"{size:.0f}" + r"$\times$" + f"{size:.0f}"
            rows[i * (Nprocs + 1) + 1] = [size_str, 1, nnodes, seq, 1]
        row_index = i * Nprocs + proc_idx + 1
        rows[i * (Nprocs + 1) + proc_idx + 1] = ["", nproc, nnodes, par, speedup]
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.set_cols_dtype(["i", "t", "t", pyutils.scientific_fmt, pyutils.scientific_fmt])
table.add_rows(rows)
label = f"tab:{header[0]}"

# add position specifier
table_str = latextable.draw_latex(table, caption=caption, label=label)
table_str = table_str.replace(r"\begin{table}", r"\begin{table}[H]")


filename = "MM_times_table.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(table_str)
print("Done!")

if args_d.get("output"):
    plt.show()
