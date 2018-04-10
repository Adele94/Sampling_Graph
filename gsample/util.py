import matplotlib.pyplot as plt
import math
import numpy as np
from gsample.sampling.Algorithms import *
import itertools
import collections
from pandas import DataFrame, read_csv
import pandas as pd


def complete_graph(n) -> nx.Graph:
    G = nx.Graph()
    if n > 1:
        if G.is_directed():
            edges = itertools.permutations(range(n), 2)
        else:
            edges = itertools.combinations(range(n), 2)
        G.add_edges_from(edges)
    return G


def complete_graph_show(n):
    G = complete_graph(n)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos)
    plt.show()


def cluster_coefficient_node(graph, v):
    return nx.clustering(graph, v)


def cluster_coefficient_average(graph):
    return nx.average_clustering(graph)


def average_degree(graph):
    values = graph.degree().values()
    return sum(values) / len(values)


def bfs(graph, start, end):
    graph_successors = nx.bfs_successors(graph, start)

    queue = [start]
    visited = []

    while len(queue) > 0:
        v = queue.pop(0)

        if v == end:
            visited.append(v)
            return visited

        if v not in visited:
            visited.append(v)
            queue += [x for x in graph_successors.get(v, []) if x not in visited]

    return visited


def normalized_mean_square_error(n, true_value, results):
    result_list = [(true_value - result) ** 2 for result in results]
    return math.sqrt(sum(result_list) / n) / true_value


def page_rank(graph):
    pr = sorted(nx.pagerank_numpy(graph, alpha=0.9).items(), key=lambda x: x[1])
    return pr[1:10]


def degree_distribution(graph):
    M = nx.to_scipy_sparse_matrix(graph)

    indegrees = M.sum(0).A[0]
    outdegrees = M.sum(1).T.A[0]
    indegree_distribution = np.bincount(indegrees)
    outdegree_distribution = np.bincount(outdegrees)
    return indegree_distribution, outdegree_distribution


def degree_histogram(G,k,name):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    ax= plt.subplot(2,2,k)
    plt.bar(deg, cnt,tick_label = None, width=0.80, color='b')
    plt.title(name)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([])
    ax.set_xticklabels(deg)
    plt.subplots_adjust(top = 0.8, hspace=0.6,wspace=0.4)

def show_random_walk_graphs(Graph_Main, size_fraction, dataset):
    plt.suptitle("Random walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main, 1, "Main Graph")
    degree_histogram(RWsampling(Graph_Main, size_fraction), 2, "Random walk 1")
    degree_histogram(RWsampling(Graph_Main, size_fraction), 3, "Random walk 2")
    degree_histogram(RWsampling(Graph_Main, size_fraction), 4, "Random walk 3")
    plt.savefig("gsample/data/output/RW/" + "Random walk " + dataset + ".png")
    plt.show()


def show_PR_walk_graphs(Graph_Main, size_fraction, dataset):
    plt.suptitle("Page Rank walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main, 1, "Main Graph")
    degree_histogram(PRsampling(Graph_Main, size_fraction), 2, "Page Rank walk 1")
    degree_histogram(PRsampling(Graph_Main, size_fraction), 3, "Page Rank walk 2")
    degree_histogram(PRsampling(Graph_Main, size_fraction), 4, "Page Rank walk 3")
    plt.savefig("gsample/data/output/PRW/" + "Page Rank walk " + dataset + ".png")
    plt.show()


def show_FF_graphs(Graph_Main, size_fraction, dataset):
    plt.suptitle("Forest Fire division \n dataset: " + dataset)
    degree_histogram(Graph_Main, 1, "Main Graph")
    degree_histogram(FFsampling(Graph_Main, size_fraction), 2, "Forest Fire 1")
    degree_histogram(FFsampling(Graph_Main, size_fraction), 3, "Forest Fire 2")
    degree_histogram(FFsampling(Graph_Main, size_fraction), 4, "Forest Fire 3")
    plt.savefig("gsample/data/output/FF/" + "Forest Fire " + dataset + ".png")
    plt.show()


def show_ESi_graphs(Graph_Main, size_fraction, dataset):
    plt.suptitle("Induced edges division \n dataset: " + dataset)
    degree_histogram(Graph_Main, 1, "Main Graph")
    degree_histogram(ESisampling(Graph_Main, size_fraction), 2, "Induced edges 1")
    degree_histogram(ESisampling(Graph_Main, size_fraction), 3, "Induced edges 2")
    degree_histogram(ESisampling(Graph_Main, size_fraction), 4, "Induced edges 3")
    plt.savefig("gsample/data/output/ESi/" + "Induced edges " + dataset + ".png")
    plt.show()


def error_graph(loaded_csv, title):
    loaded_csv.head()
    table = loaded_csv
    x = table.values[0 ,1:]
    y1 = table.values[1 ,1:]
    y2 = table.values[2 ,1:]
    y3 = table.values[3 ,1:]
    y4 = table.values[4 ,1:]
    fig, ax = plt.subplots()
    ax.plot(x,y1, color="blue", label="Forest Fire")
    ax.plot(x,y2, color="red", label="Induced Edges")
    ax.plot(x,y3, color="green", label="Random walk")
    ax.plot(x,y4, color="black", label="Page Rank walk")
    ax.set_xlabel("fraction")
    ax.set_ylabel("error")
    ax.legend()
    plt.suptitle("Error graph \n " + title)
    plt.savefig("gsample/data/output/Errors/"+title+".png")
    plt.show()

def separate_file(filename, Ns):
        SepList = []
        df = pd.read_csv(filename, delimiter = '\t' , dtype = int, comment = '#', header = None )
        n = int((df.__len__() + 1) / Ns)
        for i in range(Ns - 1):
            a = []
            dfm = df[i * n:(n + i * n)]
            SepList.append(dfm)
        dfm = df[(Ns - 1) * n:]
        SepList.append(dfm)
        return SepList

