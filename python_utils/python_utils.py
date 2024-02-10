from os.path import abspath, dirname
from pathlib import Path
from sys import path, argv
from pathlib import Path
import matplotlib.pyplot as plt

def get_root() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path).parent
    # return Path(getcwd()).resolve()

# get CLI
def get_cli_args() -> dict:
    args_d = {}
    if "--show-output=" in argv:
        output = argv[argv.index("--show-output=") + 1]
        if output.lower == "True":
            output = True
        elif output == "False":
            output = False
        else:
            raise ValueError("No or invalid output flag found")
        argv.remove("--show-output=")
        args_d["output"] = output
    return args_d

def get_metadata(file: Path) -> dict:
    metadata = {}
    metas = file.stem.split("_")
    for meta in metas:
        try: key, value = meta.split("=")
        except ValueError: continue
        metadata[key] = value
    return metadata

def scientific_fmt(s: float, prec: int = 2) -> str:
    specifier = f"{{:.{prec}e}}"
    scientific_str = specifier.format(s)
    mantissa, exponent = scientific_str.split("e")
    if exponent[0] == "+":
        sign = ""
    elif exponent[0] == "-":
        sign = "-"
    if exponent[1] == "0":
        exponent = exponent[2:]
    if exponent == "0":
        out = mantissa
    else:
        out = mantissa + r"$\times 10^{" + sign + exponent + "}$"
    return out

# set standard matploylib style
def set_style():
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 12
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.minor.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.shadow"] = True
    plt.rcParams["legend.borderpad"] = 1
    plt.rcParams["legend.borderaxespad"] = 1
    plt.rcParams["legend.handletextpad"] = 1
    plt.rcParams["legend.handlelength"] = 1.5
    plt.rcParams["legend.labelspacing"] = 1
    plt.rcParams["legend.columnspacing"] = 2
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["savefig.orientation"] = "landscape"
    # plt.rcParams["savefig.frameon"] = False