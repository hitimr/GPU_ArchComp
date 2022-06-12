import numpy as np
import pandas as pd
import igraph as ig
import time
import math
import random

import common




# Generate a random, connected graph using Barabasi algorithm
# Edges are bidirectional with random weights

def generate_graph(n_nodes: int, connectivity: int, w_min=0, w_max=100):
    """_summary_

    Args:
        n_nodes (int): Number nodes/vertices
        connectivity (int): Number of nodes to be drawn with every new vertex. Total number 
        w_min (int, optional): Minimum edge weight. Defaults to 0
        w_max (int, optional): Maximum Edge weight. Defaults to 100

    Returns:
        g (pd.DataFrame): Adjecency Matrix as DataFrame in COO-Format (coo1, coo2, weight) as returned by generate_graph()
    """

    # generate bidirectional connected graph
    # Barabasi does not assign weights though
    g = ig.Graph.Barabasi(n_nodes, connectivity)

    # retrieve edgelist ans assign a new column with random weihts
    df = g.get_edge_dataframe()
    df["weight"] = np.random.randint(low=w_min, high=w_max, size=len(df))

    # create new graph from DataFrame and return it
    return ig.Graph.DataFrame(df)


def convert_to_edgelist(g: ig.Graph):
    # retrieve edges including weights
    df = g.get_edge_dataframe()
    df = df.rename(columns={
        "source": "u",
        "target": "v"
    })

    # sort edges by coordinates to ensure correct order
    df = df.sort_values(by=["u", "v"])

    return df


def export_graph(g: pd.DataFrame, filename: str):
    """Write graph data to [filename] according to the format specified in TUWEL

    Args:
        g (pd.DataFrame): Adjecency Matrix as DataFrame in COO-Format (coo1, coo2, weight) as returned by generate_graph()
        filename (str): output filename (absolute path recommended)
    """
    df = convert_to_edgelist(g)

    # add Tags to leftmost columns
    df["tag"] = "E"
    df = df[["tag", "u", "v", "weight"]]  # reorder columns

    # create header by renaming columns names
    df = df.rename(columns={
        "tag": "H",
        "u": str(g.vcount()),
        "v": str(len(df)),
        "weight": "1"}
    )
    # write to file
    df.to_csv(filename, sep=";", index=False)


def calculate_mst(g: ig.Graph):
    """Calculate the Minimum spanning Tree of g

    Args:
        g (ig.Graph): Connected graph with weighted edges

    Returns:
        ig.Graph: Minimum Spanning Tree-Graph 
    """
    return g.spanning_tree(weights=g.get_edge_dataframe()["weight"])

def graph_stats(num_edges, density):
    num_nodes = int(math.sqrt(num_edges/density))
    connectivity = int(num_nodes * density)
    return num_nodes, connectivity

def create_graphs(graph_list, output_dir=common.INPUT_DATA_DIR, create_mst=True):
    # graph list is a list of dictionaries:
    # [{n_edges: int, density: float, w_min: int, w_max: int}]
    for graph in graph_list:
#        print(graph)
#        print(graph["n_nodes"])

        base_filename = output_dir / \
        f"barabasi_{graph['n_nodes']}_{graph['connectivity']}" 

        # generate Connected, weithed graph
        print("generating graph... n_nodes: {:.1e}, connectivity: {:.3e}".format(graph['n_nodes'], graph['connectivity']))
        start_time = time.time()
        #n_nodes, connectivity = graph_stats(graph['n_edges'], graph['density'])
        #print("settling for n_nodes: {}, connectivity: {}, n_edges: {:.3e}".format(n_nodes, connectivity, n_nodes*connectivity))
        g = generate_graph(graph['n_nodes'], graph['connectivity'], w_min=graph['w_min'], w_max=graph['w_max'])
        print("graph generated in {:.2f} seconds".format(time.time() - start_time))

        # save graph and mst to file
        print("exporting...")
        export_graph(g, f"{base_filename}.csv")
        print("export done after {:.2f} seconds".format(time.time() - start_time))

        # calculate reference solution using igraph
        if create_mst:
            start_time = time.time()
            print("calculating mst...")
            g_mst_gt = calculate_mst(g)
            export_graph(g_mst_gt, f"{base_filename}_mst_gt.csv")
            print("mst done after {:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":

    #densities = [.01, .1, .25, .5, .75, .9, .99]
    # densities = [.5]
    #c_facs = [.052,.054,.056,.058]
    #c_facs = [.292,.294,.296,.298]
    #connectivities = [60,70,80,90,100]  
    connectivities = [1]  

    #c_facs = [.053]
    c_facs = [.293]

    for f in c_facs: 
        graph_list = [
            {"n_nodes": 100, "connectivity": int(f*100), "w_min": 1, "w_max": 1000},
            {"n_nodes": 200, "connectivity": int(f*200), "w_min": 1, "w_max": 1000},
            {"n_nodes": 400, "connectivity": int(f*400), "w_min": 1, "w_max": 1000},
            {"n_nodes": 800, "connectivity": int(f*800), "w_min": 1, "w_max": 1000}
#            {"n_nodes": 1600, "connectivity": int(f*1600), "w_min": 1, "w_max": 1000},
#            {"n_nodes": 3200, "connectivity": int(f*3200), "w_min": 1, "w_max": 1000},
#            {"n_nodes": 6400, "connectivity": int(f*6400), "w_min": 1, "w_max": 1000}
#            {"n_nodes": 3200, "connectivity": int(c_fac*3200), "w_min": 1, "w_max": 1000},
#            {"n_edges": int(3e4), "density": d, "w_min": 1, "w_max": 1000}
#            {"n_edges": int(1e5), "density": d, "w_min": 1, "w_max": 1000},
#            {"n_edges": int(3e5), "density": d, "w_min": 1, "w_max": 1000},
#            {"n_edges": int(1e6), "density": d, "w_min": 1, "w_max": 1000},
#            {"n_edges": int(3e6), "density": d, "w_min": 1, "w_max": 1000},
#            {"n_edges": int(1e7), "density": d, "w_min": 1, "w_max": 1000},
#            {"n_edges": int(3e7), "density": d, "w_min": 1, "w_max": 1000}
        ]
#        out_dir = common.INPUT_DATA_DIR / "tester"
        out_dir = common.WORKSPACE_DIR / "benchmark_data"
        create_graphs(graph_list, output_dir=out_dir, create_mst=False)



