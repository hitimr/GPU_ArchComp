import numpy as np
import pandas as pd
import igraph as ig
import time
import math

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
    for graph in graph_list:

        base_filename = output_dir / \
        f"barabasi_{graph['n_nodes']}_{graph['density']}" 

        # generate Connected, weighted graph
        print("generating graph... n_nodes: {:.1e}, connectivity: {:.3e}".format(graph['n_nodes'], graph['connectivity']))
        start_time = time.time()
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

    d_fac = {"10pct": .053, "50pct": .293, "90pct": .684} # these factors will create the desired densities

    ranges = {"10pct": [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600],
              "50pct": [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800],
              "90pct": [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]    
             }

    #for key in d_fac:
    for key in ['90pct']:
        print("density: ", key)
        graph_list = []
        for r in ranges[key]:
            graph_list.append({"n_nodes": r, "connectivity": int(d_fac[key]*r), "w_min": 1, "w_max": 1000, 'density': key})
        print('list of graphs to be created:')
        for graph in graph_list:
            print(graph)
        print('')

        out_dir = common.WORKSPACE_DIR / "benchmark_data"
        print('creating graphs: ')
        create_graphs(graph_list, output_dir=out_dir, create_mst=True)


