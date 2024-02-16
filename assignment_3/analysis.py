import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import python_utils as pyutils
from matplotlib import cm
from tabulate import tabulate
from texttable import Texttable
import latextable

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
benchmarkFolder = root / "assignment_3" / "benchmark"
benchmarkFiles = sorted(benchmarkFolder.glob("*.txt"))
benchmarkData = {}
for i, file in enumerate(benchmarkFiles):
    benchmarkData[i] = {}
    benchmarkData[i]["meta"] = pyutils.get_metadata(file)
    with open(file, "r") as f:
        data = f.readlines()
        # cpu time
        cpu_time = float(data[0].split(":")[1].strip())

        # gpu time (global)
        gpu_g_times = data[1].split(":")[1].strip()
        gpu_g_time, gpu_g_mem_time = gpu_g_times.split(",")
        gpu_g_time = float(gpu_g_time.strip())
        gpu_g_mem_time = float(gpu_g_mem_time.strip())

        # gpu time (shared)
        gpu_s_times = data[2].split(":")[1].strip()
        gpu_s_time, gpu_s_mem_time = gpu_s_times.split(",")
        gpu_s_time = float(gpu_s_time.strip())
        gpu_s_mem_time = float(gpu_s_mem_time.strip())

        # gpu shared memory size (KB)
        gpu_s_mem_size = float(data[3].split(":")[1].strip())

        benchmarkData[i]["cpu_time"] = cpu_time
        benchmarkData[i]["gpu_g_time"] = [gpu_g_time, gpu_g_mem_time]
        benchmarkData[i]["gpu_s_time"] = [gpu_s_time, gpu_s_mem_time]
        benchmarkData[i]["gpu_s_mem_size"] = gpu_s_mem_size

# table of execution times
print("\tMaking table of execution times...", end="")
header = [
    "matrix size",
    "cpu time",
    "32 threads",
    "64 threads",
    "100 threads",
]
sizes = [50, 500, 2000, 4000, 5000]
threads = [32, 64, 100]
caption = "Execution time for different matrix sizes and number of threads."
thread_idx = 0
rows = [[0] * len(header) for _ in range(len(sizes) + 1)]
rows[0] = header
separator = r" $\left.\right|$ "
for i, data in benchmarkData.items():
    nthreads = data["meta"]["nthreads"].split(".")[0] + " threads"
    try:
        colidx = header.index(nthreads)
    except ValueError:
        continue
    n = int(data["meta"]["n"])
    rowidx = sizes.index(n)
    cpu_time = data["cpu_time"]
    gpu_g_time = pyutils.scientific_fmt(data["gpu_g_time"][0])
    if nthreads != "100 threads":
        gpu_s_time = pyutils.scientific_fmt(data["gpu_s_time"][0])
    else:
        gpu_s_time = "N.A."

    rows[rowidx + 1][0] = n
    rows[rowidx + 1][1] = cpu_time
    rows[rowidx + 1][colidx] = gpu_g_time + separator + gpu_s_time

# sort rows by matrix size
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.set_cols_dtype(["i", pyutils.scientific_fmt, "t", "t", "t"])
table.add_rows(rows)
label = f"tab:power_method_exectimes"

# add position specifier & change font size
table_str = latextable.draw_latex(table, caption=caption, label=label)
table_str = table_str.replace(r"\begin{table}", r"\begin{table}[H]\footnotesize")

filename = "power_method_exectimes.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(table_str)
print("Done!")

# table of speedups without memory transfer
print("\tMaking table of speedups (without mem. transfer)...", end="")
header = [
    "matrix size",
    "32 threads",
    "64 threads",
    "100 threads",
]
caption = "speedups without memory transfer."
thread_idx = 0
rows = [[0] * len(header) for _ in range(len(sizes) + 1)]
rows[0] = header
for i, data in benchmarkData.items():
    nthreads = data["meta"]["nthreads"].split(".")[0] + " threads"
    try:
        colidx = header.index(nthreads)
    except ValueError:
        continue
    n = int(data["meta"]["n"])
    rowidx = sizes.index(n)
    cpu_time = data["cpu_time"]
    gpu_g_speedup = "{0:4.2f}".format(cpu_time / data["gpu_g_time"][0])
    if nthreads != "100 threads":
        gpu_s_speedup = "{0:4.2f}".format(cpu_time / data["gpu_s_time"][0])
    else:
        gpu_s_speedup = "N.A."

    rows[rowidx + 1][0] = n
    rows[rowidx + 1][colidx] = gpu_g_speedup + separator + gpu_s_speedup

# sort rows by matrix size
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.set_cols_dtype(["i", "t", "t", "t"])
table.add_rows(rows)
label = f"tab:power_method_speedup_no_mem"

# add position specifier
table_str = latextable.draw_latex(table, caption=caption, label=label)
table_str = table_str.replace(r"\begin{table}", r"\begin{table}[H]\footnotesize")

filename = "power_method_speedup_no_mem.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(table_str)
print("Done!")

# table of speedups with memory transfer
print("\tMaking table of speedups (with mem. transfer)...", end="")
caption = "speedups with memory transfer."
thread_idx = 0
rows = [[0] * len(header) for _ in range(len(sizes) + 1)]
rows[0] = header
for i, data in benchmarkData.items():
    nthreads = data["meta"]["nthreads"].split(".")[0] + " threads"
    try:
        colidx = header.index(nthreads)
    except ValueError:
        continue
    n = int(data["meta"]["n"])
    rowidx = sizes.index(n)
    cpu_time = data["cpu_time"]
    gpu_g_speedup = "{0:5.2f}".format(cpu_time / np.sum(data["gpu_g_time"]))
    if nthreads != "100 threads":
        gpu_s_speedup = "{0:5.2f}".format(cpu_time / np.sum(data["gpu_s_time"]))
    else:
        gpu_s_speedup = "N.A."

    rows[rowidx + 1][0] = n
    rows[rowidx + 1][colidx] = gpu_g_speedup + separator + gpu_s_speedup

# sort rows by matrix size
table = Texttable()
table.set_cols_align(["c"] * len(header))
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.set_precision(2)
table.set_cols_dtype(["i", "t", "t", "t"])
table.add_rows(rows)
label = f"tab:power_method_speedup_with_mem"

# add position specifier
table_str = latextable.draw_latex(table, caption=caption, label=label)
table_str = table_str.replace(r"\begin{table}", r"\begin{table}[H]\footnotesize")

filename = "power_method_speedup_with_mem.tex"
filepath = root / "report" / "tables" / filename
with open(filepath, "w") as f:
    f.write(table_str)
print("Done!")
