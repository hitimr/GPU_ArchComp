import numpy as np
import pandas as pd
from pathlib import Path

proj_root_dir = Path(__file__).parent.parent

# Union-FInd Data structure for evaluating which Node belongs to what tree
# https://en.wikipedia.org/wiki/Disjoint-set_data_structure
class UnionFind:
    def __init__(self, n):
        self.parent = np.linspace(0, n-1, n, dtype=int)

    def find(self, i):
        # check to what tree a certain edge belongs
        # for it just uses a regular recursive algorithm which is ineficcient
        # optimizations such as branch-compression are possible
        
        i = int(i) # for some reason ints appear as double here some times. dont know why yet
        if self.parent[i] == i: return i
        else: return self.find(self.parent[i])

    def link(self, i, j):
        assert self.find(i) != self.find(j) 
        self.parent[i] = j

    def union(self, i, j):
        # merge 2 disconnected trees to 1
        if self.find(i) != self.find(j):
            self.link(self.find(i), self.find(j))

# regular kruskal
def kruskal(E, T, P):
    # Sort all edges by weight
    sorted = E.sort_values(by="weight", ascending=True)

    # grow MST
    for index, edge in sorted.iterrows():
        if P.find(edge["u"]) != P.find(edge["v"]):
            T = T.append(edge)
            P.union(edge["u"], edge["v"])
    return E, T.sort_values(by=["u"]), P 


# FIlter Kruskal from paper
# uses regular kruskal as a subroutine
def filter_kruskal(E, T, P):

    # call regular kruskal once the subtree is small enough
    # For now some thresholds dont seem to work. My guess is that sometimes the right pivot-set is empty whicht breaks the code
    # havent had time to debug it yet
    if len(E) < 16: # KruskalTheshold. Arbitrary for now
        return kruskal(E, T, P)

    else:
        E_big = pd.DataFrame()
        E_leq = pd.DataFrame()
        p = E.sample(1, random_state=42)["weight"].values[0] # Pick random weight as pivot element
        E_leq = E.loc[E["weight"] <= p]
        E_big = E.loc[E["weight"] > p]
        _, T, P = filter_kruskal(E_leq, T, P)
        E_big = filter(E_big, P)
        _, T, P = filter_kruskal(E_big, T, P)

    return E,T,P

def filter(E, P):
    filtered = pd.DataFrame()
    for index, edge in E.iterrows():
        if P.find(edge["u"]) != P.find(edge["v"]):
            filtered = filtered.append(edge, ignore_index=True)   
    return filtered


if __name__ == "__main__":
    graph = pd.read_csv(proj_root_dir / "input_data/example.txt", sep=";", header=None)[1:].drop(columns=0)
    graph = graph.rename(columns={1: "u", 2: "v", 3: "weight"})
    graph = pd.DataFrame(graph, dtype=int)

    E = graph

    print("Regular Kruskal results")
    T = pd.DataFrame(dtype=int)
    P = UnionFind(len(graph))
    E,T,P = kruskal(E, T, P)
    print(T)

    print("\nfilter kruskal reuslts")
    T = pd.DataFrame(dtype=int)
    P = UnionFind(len(graph))
    E,T,P = filter_kruskal(E, T, P)
    print(T)