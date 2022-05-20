import numpy as np
import igraph as ig
import os
NODES = 20
DENSITY = 0.2

max_edges = NODES * (NODES - 1)
min_edges = NODES - 1


weight_min = 1
weight_max = 100

num_edges = int(max_edges * DENSITY)

print(f"NUMER of EDGES = {num_edges} - MIN: {min_edges}, MAX: {max_edges}")
assert num_edges > min_edges

current_edges_num = 0
adjacency_matrix = np.zeros((NODES, NODES))
vis = np.zeros((NODES, NODES))

current_density = 0

while True:
    current_edges_num = 0
    adjacency_matrix = np.zeros((NODES, NODES))
    vis = np.zeros((NODES, NODES))

    current_density = 0
    while current_edges_num < num_edges:
        starting_node = np.random.randint(NODES)
        end_node = np.random.randint(NODES)
        if starting_node == end_node:
            continue
        if adjacency_matrix[starting_node, end_node] == 0 and adjacency_matrix[end_node, starting_node] == 0:
            adjacency_matrix[starting_node, end_node] = np.random.randint(weight_min, weight_max)
            adjacency_matrix[end_node, starting_node] = np.random.randint(weight_min, weight_max)
        
            current_edges_num += 1

    g = ig.Graph.Weighted_Adjacency(adjacency_matrix, mode = 'undirected')

    if g.is_connected():
        mst = g.spanning_tree(weights=g.es["weight"],return_tree = False)
        mst_adj = g.spanning_tree(weights=g.es["weight"],return_tree = True)
        g.es["color"] = "lightgray"
        g.es[mst]["color"] = "midnightblue"   
        g.es["width"] = g.es["weight"]
        g.es["width"] = [10*x/weight_max for x in g.es["width"]]
        ig.plot(g)
        filename = f"{NODES}_{DENSITY}_{weight_min}_{weight_max}.csv"
        with open('../input_data/'+filename, 'w') as f:
            f.write(f"H;{NODES};{num_edges};1")
            for i in range(NODES):
                for j in range(i+1,NODES):
                    if(adjacency_matrix[i,j] > 0):
                        f.write("\n")
                        f.write(f"E;{i};{j};{int(adjacency_matrix[i,j])}")
        gt_filename = f"mst_gt_{NODES}_{DENSITY}_{weight_min}_{weight_max}.csv"

        assert mst_adj.is_connected()
        mst_adj_mat = mst_adj.get_adjacency()
        mst_adj_mat = np.array(mst_adj_mat.data)

        print("Minimum edge weight sum:", sum(g.es[mst]["weight"]))

        mst_adj_mat = np.multiply(mst_adj_mat, adjacency_matrix)

        with open('../input_data/'+gt_filename, 'w') as f:
            f.write(f"H;{NODES};{mst_adj.ecount()};1")
            for i in range(NODES):
                for j in range(i+1,NODES):
                    if(mst_adj[i,j] > 0):
                        f.write("\n")
                        f.write(f"E;{i};{j};{int(mst_adj_mat[i,j])}")
        break
    else:
        pass

