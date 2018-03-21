# -*- coding=utf-8 -*-

import networkx as nx
import numpy as np
import itertools
import random

from util import normalized_mean_square_error


def random_walk(graph, start_node=None, size=-1, metropolized=False):
    """
    RWでサンプリングしたノード列を返す

    :param graph: グラフ
    :param start_node: 先頭ノード
    :param size: ノード列のサイズ
    :param metropolized: metropolis hasting random walk フラグ
    :return: サンプリングしたノード列
    """
    if start_node is None:
        start_node = random.choice(graph.nodes())

    v = start_node
    for c in itertools.count():
        if c == size:
            return
        if metropolized:
            candidate = random.choice(graph.neighbors(v))
            v = candidate if (random.random() < float(graph.degree(v)) / graph.degree(candidate)) else v
        else:
            v = random.choice(graph.neighbors(v))

        yield v


def random_walk_cca(graph, start_node=None, size=-1, metropolized=False):
    """
    RWでサンプリングしたグラフの平均クラスタ係数を返す

    :param graph: グラフ
    :param start_node: 先頭ノード
    :param size: サイズ
    :param metropolized: metropolis hasting random walk フラグ
    :return: 平均クラスタ係数
    """
    nodes = list(random_walk(graph=graph, start_node=start_node, size=size, metropolized=metropolized))
    data = list()
    for node in nodes:
        data.append(nx.clustering(graph, node))
    return sum(data) / len(data)


def random_walk_aggregation(graph, start_node=None, size=-1, metropolized=False, tv=-1, n=100):
    """
    RW, MHRWでサンプリングしたノード列について、クラスタ係数を100回計算し、その平均と分散を返す

    :param graph: グラフ
    :param start_node: 先頭ノード
    :param size: サイズ
    :param metropolized: metropolis hasting random walk フラグ
    "param tv: 真値
    :return: {average: 平均, var: 分散}
    """
    if start_node is None:
        start_node = random.choice(graph.nodes())

    result = []
    for i in range(1, 100):
        cca = random_walk_cca(graph, start_node, size, metropolized)
        result.append(cca)

    data = np.array(result)
    average = np.average(data)
    var = np.var(data)
    nmse = normalized_mean_square_error(100, tv, result)
    return {"average": average, "var": var, "nmse": nmse}