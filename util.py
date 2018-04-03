# coding=utf-8

import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from Algorithms import *

import collections
from collections import defaultdict


def complete_graph(n) -> nx.Graph:
    """
    完全グラフを計算する

    :param n: ノードの数
    :return: グラフ
    """
    G = nx.Graph()
    if n > 1:
        if G.is_directed():
            edges = itertools.permutations(range(n), 2)
        else:
            edges = itertools.combinations(range(n), 2)
        G.add_edges_from(edges)
    return G


def complete_graph_show(n):
    """
    完全グラフを表示する

    :param n: ノードの数
    :return:
    """
    G = complete_graph(n)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos)
    plt.show()


def cluster_coefficient_node(graph, v):
    """
    graphのあるノードvのクラスタ係数を求める

    :param graph: グラフ
    :param v: ノード
    :return:
    """
    return nx.clustering(graph, v)


def cluster_coefficient_average(graph):
    """
    graphの平均クラスタ係数を求める
    :param graph: グラフ
    :return: 平均クラスタ係数
    """
    return nx.average_clustering(graph)


def average_degree(graph):
    """
    graphの平均次数を求める
    :param graph: グラフ
    :return: 平均次数
    """
    values = graph.degree().values()
    return sum(values) / len(values)


def bfs(graph, start, end):
    """
    graphを幅優先探索する

    :param graph: グラフ
    :param start: 始点ノード
    :param end: 終点ノード
    :return: 訪れたノード列
    """
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
    """
    正規化された平均自乗誤差を計算する。
    :param n: 試行回数
    :param true_value: 真値
    :param results: 結果を格納したリスト
    :return: 正規化された平均自乗誤差
    """
    result_list = [(true_value - result) ** 2 for result in results]
    return math.sqrt(sum(result_list) / n) / true_value


def page_rank(graph):
    """
    page rank アルゴリズムで計算したノード上位10個を返す
    :param graph: nx.Graph
    :return: [(id, value)]
    """
    pr = sorted(nx.pagerank_numpy(graph, alpha=0.9).items(), key=lambda x: x[1])
    return pr[1:10]


def degree_distribution(graph):
    """
    次数分布の indegree, outdegree を計算する。

    :param graph: nx.Graph
    :return: indegree_distribution, outdegree_distribution
    """
    M = nx.to_scipy_sparse_matrix(graph)

    indegrees = M.sum(0).A[0]
    outdegrees = M.sum(1).T.A[0]
    # ノードに入ってくる辺数
    indegree_distribution = np.bincount(indegrees)
    # ノードから出ていく辺数
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


def show_random_walk_graphs(dataset,size_fraction):
    Graph_Main = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    plt.suptitle("Random walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(create_random_walk_graph(dataset,size_fraction),2,"Random walk 1")
    degree_histogram(create_random_walk_graph(dataset,size_fraction),3,"Random walk 2")
    degree_histogram(create_random_walk_graph(dataset,size_fraction),4,"Random walk 3")
    plt.savefig("data/output/RW/"+"Random walk "+dataset + ".png")
    plt.show()

def show_PR_walk_graphs(dataset,size_fraction):
    Graph_Main = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    plt.suptitle("Page Rank walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(create_PR_walk_graph(dataset,size_fraction),2,"Page Rank walk 1")
    degree_histogram(create_PR_walk_graph(dataset,size_fraction),3,"Page Rank walk 2")
    degree_histogram(create_PR_walk_graph(dataset,size_fraction),4,"Page Rank walk 3")
    plt.savefig("data/output/PRW/"+"Page Rank walk "+ dataset + ".png")
    plt.show()

def show_FF_graphs(dataset,size_fraction):
    Graph_Main = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    plt.suptitle("Forest Fire division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(FFsampling(dataset,size_fraction),2,"Forest Fire 1")
    degree_histogram(FFsampling(dataset,size_fraction),3,"Forest Fire 2")
    degree_histogram(FFsampling(dataset,size_fraction),4,"Forest Fire 3")
    plt.savefig("data/output/FF/"+"Forest Fire "+ dataset + ".png")
    plt.show()

def show_ESi_graphs(dataset,size_fraction):
    Graph_Main = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    plt.suptitle("Induced edges division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(ESisampling(dataset,size_fraction),2,"Induced edges 1")
    degree_histogram(ESisampling(dataset,size_fraction),3,"Induced edges 2")
    degree_histogram(ESisampling(dataset,size_fraction),4,"Induced edges 3")
    plt.savefig("data/output/ESi/"+"Induced edges "+ dataset + ".png")
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
    plt.savefig("data/output/Errors/"+title+".png")
    plt.show()