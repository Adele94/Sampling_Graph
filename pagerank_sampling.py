# coding=utf-8

import networkx as nx


def page_rank_sampling(graph, size=2000):
    """
    ページランクアルゴリズムでサンプリングする。
    :param graph:
    :param size:
    :return:
    """
    pg = sorted(nx.pagerank(graph).items(), key=lambda x: x[1])
    pg.reverse()
    nodes = list(map(lambda x: x[0], pg))[0:size]
    return nodes


def page_rank_sampling_cca(graph, size=2000):
    """
    ページランクアルゴリズムでサンプリングしたノードのクラスタ係数を計算し、
    その平均を出力する。
    :param graph:
    :param size:
    :return:
    """
    nodes = page_rank_sampling(graph, size)
    data = list()
    for node in nodes:
        print("cluster coefficient of node: {0}".format(nx.clustering(graph, node)))
        data.append(nx.clustering(graph, node))
    return sum(data) / len(data)

