import numpy as np
import pandas as pd
import igraph as ig

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
    df = df.rename(columns= {
        "source" : "u",
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
