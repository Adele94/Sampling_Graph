# coding=utf-8

import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math


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


