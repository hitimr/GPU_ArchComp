from os import W_OK
from pathlib import Path
import sys
import pandas as pd
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

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




# benchmark stuff
def out_file_name(single_run):
    return single_run['prefix'] + "_" + str(single_run['n_nodes']) + "_" + str(single_run['density']) + ".csv"

def do_benchmark_runs(list_of_runs):
    for single_run in list_of_runs:
        args = f"--inputfile benchmark_data/barabasi_{single_run['n_nodes']}_{single_run['density']}.csv {single_run['flags']}"
        out_file = out_file_name(single_run)
        #common.run_benchmark(common.OUT_DIR / out_file, args=args, print_output=False)
        run_benchmark(OUT_DIR / out_file, args=args, print_output=False)


# collect data for stacked bars
def collect_for_stack(list_of_runs, stack_items=['Initialize', 'filter()', 'grow MST', 'sort()', 'partition()']):
    labels = []
    results = {}
    for si in stack_items:
        results[si] = []

    for single_run in list_of_runs:
        labels.append(single_run['n_nodes'])
        file_name = out_file_name(single_run)
        #df = pd.read_csv(common.OUT_DIR / file_name, sep=';')
        df = pd.read_csv(OUT_DIR / file_name, sep=';')
        df.set_index('tag',inplace=True)
        for si in stack_items:
            results[si].append(df.loc[si]['total'])
    return labels, results

# collect run times for plotting
def collect_run_times(list_of_runs):
    results = {}
    row_labels = []

    for single_run in list_of_runs:
        file_name = out_file_name(single_run)
        #df = pd.read_csv(common.OUT_DIR / file_name, sep=';')
        df = pd.read_csv(OUT_DIR / file_name, sep=';')
        df.set_index('tag',inplace=True)
        run_time = df.loc['total']['median']
        row_label = single_run['prefix'] + "_" + single_run['density']
        if not(row_label in row_labels):
            row_labels.append(row_label)
            results[row_label+"_x"] = []
            results[row_label+"_y"] = []
        results[row_label+"_x"].append(single_run['n_nodes'])
        results[row_label+"_y"].append(run_time)
    return row_labels, results


def plot_stacked_lines(labels, results):
    for key in results:
        plt.bar(labels, results[key])
    plt.legend()
    plt.grid()
    plt.xlabel('number of nodes')
    plt.ylabel('execution time in s')



#def plot_lines(row_labels, results):
def plot_lines(list_of_runs):
    labels, results = collect_run_times(list_of_runs)

    for row_label in labels:
        plt.loglog(results[row_label+'_x'], results[row_label+'_y'], label=row_label, marker='o')
    plt.legend()
    plt.grid()
    plt.xlabel('number of nodes')
    plt.ylabel('execution time in s')


def stack_percent(stack, top_at_100=True):
    top = 1
    if top_at_100:
        top = 100

    stack_pct = {}
    totals = []
    keys = list(stack.keys())
    resolutions = len(stack[keys[0]])

    for i in range(resolutions):
        total = 0
        for key in stack:
            total += stack[key][i]
        totals.append(total)
    
    for key in keys:
        stack_pct[key] = [stack[key][i] / totals[i] * top for i in range(resolutions)]
        
    return stack_pct, totals


def stacked_bars(list_of_runs):
    labels, stack = collect_for_stack(list_of_runs)
    str_labels = []
    for label in labels:
        str_labels.append(str(label))
    key_list = list(stack.keys())
    stack_pct, totals = stack_percent(stack, top_at_100=True)
    #print(totals)
    bottom = np.zeros(len(labels))
    for i in range(len(key_list)):
        #plt.bar(str_labels, stack_pct[key_list[i]], bottom=stack_pct[key_list[i-1]], label=key_list[i])
        plt.bar(str_labels, stack_pct[key_list[i]], bottom=bottom, label=key_list[i])
        bottom += np.array(stack_pct[key_list[i]])

    plt.legend()
    plt.xlabel('number of nodes')
    plt.ylabel('portion of run time in %')



if __name__ == "__main__":
    run_benchmark(OUT_DIR / "timing_results.csv")
