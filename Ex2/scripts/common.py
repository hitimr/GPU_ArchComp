from os import W_OK
from pathlib import Path
import sys
import pandas as pd
import subprocess
import json


WORKSPACE_DIR = proj_root_dir = Path(__file__).parent.parent
INPUT_DATA_DIR = WORKSPACE_DIR / "input_data"
SCRIPT_DIR = WORKSPACE_DIR / "scripts"
OUT_DIR = WORKSPACE_DIR / "out"
EXECUTABLE = WORKSPACE_DIR / "build/src/ex2"


def load_df(filename):
    df = pd.read_csv(filename, sep=";")
    return df


def run(args="", print_output=True):
    cmd = str(f"{EXECUTABLE} {args}")

    if(print_output):
        subprocess.run(cmd.split(sep=" "), check=True,
                       stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd.split(sep=" "), check=True,
                       stdout=subprocess.DEVNULL)


def run_benchmark(outfile, args="", print_output=True, verbose=True):
    cmd = str(f"{EXECUTABLE} --ouputfile_timings {outfile} {args}")

    if(verbose):
        print(f"Running benchmark with .{cmd}")

    if(print_output):
        subprocess.run(cmd.split(sep=" "), check=True,
                       stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd.split(sep=" "), check=True,
                       stdout=subprocess.DEVNULL)

    return load_df(outfile)


if __name__ == "__main__":
    run_benchmark(OUT_DIR / "timing_results.csv")
