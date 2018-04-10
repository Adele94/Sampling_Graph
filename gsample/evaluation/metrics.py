import pandas as pd
from scipy import stats
from gsample.sampling.Algorithms import *
from collections import defaultdict
import numpy as np

def degree_cdf(graph):  # Degree distribution
    nodes = graph.nodes()
    num_nodes = float(graph.number_of_nodes())

    dist = defaultdict(int)

    for n in nodes:
        num_neighbors = len(list(graph.neighbors(n)))
        if num_neighbors > 0:
            dist[num_neighbors] += 1

    cdf = []
    s = 0.0
    na = 1.0
    for k in sorted(dist.keys()):
        s += float(dist[k]) / num_nodes
        na -= float(dist[k]) / num_nodes
        cdf.append(s)

    return np.asarray(cdf)

def clus_coeff_cdf(graph):   # Clustering coefficient distribution
    nodes = graph.nodes()
    clus_c = nx.clustering(graph)

    dist = defaultdict(int)
    num_node_ge_one = 0
    for n in nodes:
        num_neighbors = len(list(graph.neighbors(n)))
        if num_neighbors > 1:
            cc = np.round(clus_c[n], 2)
            dist[cc] += 1
            num_node_ge_one += 1

    cdf = []
    num_node_ge_one = float(num_node_ge_one)
    s = 0.0
    for k in sorted(dist.keys()):
        s += float(dist[k]) / num_node_ge_one
        cdf.append(s)

    return np.asarray(cdf)

def hop_cdf(graph):   # Path length distribution or hop disribution
    nodes = graph.number_of_nodes()
    paths = nx.shortest_path_length(graph)

    dist = defaultdict(int)

    for u in paths.keys():
        for v in paths[u].keys():
            dist[paths[u][v]] += 1

    cdf = []
    for k in sorted(dist.keys()):
        cdf.append(float(dist[k]) / (nodes * nodes))

    return np.asarray(cdf)

def eigenvalues(graph,size):
    try:
        import numpy.linalg as linal
        eigenvalues = linal.eigvals
    except ImportError:
        raise ImportError("numpy can not be imported.")

    L = nx.normalized_laplacian_matrix(graph)
    eigen_values = eigenvalues(L.A)

    return sorted(eigen_values, reverse=True)[:size]

def normalized_L1(p, q):  #normalized L1 distance
    size = len(p)
    return sum([float(np.abs(p[i] - q[i])) / float(p[i]) for i in range(size)]) / float(size)