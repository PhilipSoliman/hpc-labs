from os.path import abspath, dirname
from pathlib import Path
from sys import path
from pathlib import Path

def get_root() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path).parent
    # return Path(getcwd()).resolve()

def get_metadata(file: Path) -> dict:
    metadata = {}
    if "nnodes" in file.stem:
            metadata["nnodes"] = file.stem.split("nnodes=")[1].split("_")[0]
    if "nproc" in file.stem:
        metadata["nproc"] = file.stem.split("nproc=")[1].split("_")[0]
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

