# coding=utf-8

import networkx as nx
import numpy as np
import random
import itertools
from util import normalized_mean_square_error
from random_walk import random_walk_aggregation

def random_walk_proposal(graph, start_node=None, size=-1):
    """
    提案手法のランダムウォーク
    :param graph: nx.Graph
    :param start_node: ノード
    :param size: サイズ
    :return: [node]
    """
    if start_node is None:
        start_node = random.choice(graph.nodes())
    v = start_node
    for c in itertools.count():
        if c == size:
            return
        candidate = random.choice(graph.neighbors(v))
        neighbors_degree = np.array(list(map(lambda x: graph.degree(x), graph.neighbors(v))))
        v = candidate if (random.random() < float(graph.degree(v) ** 2 / np.sum(neighbors_degree))) else v
        yield v


def random_walk_proposal_aggregation(graph, start_node=None, size=-1, tv=1.0, n=100):
    """
    提案手法のランダムウォークを用いてCluster Coefficientを計算する
    :param graph: nx.Graph
    :param start_node: ノード
    :param size: サイズ
    :param tv: 真値
    :param n: 試行回数
    :return:
    """
    result = []
    for i in range(1, n):
        nodes = list(random_walk_proposal(graph, start_node, size))
        data = []
        for node in nodes:
            data.append(nx.clustering(graph, node))
        average = sum(data) / len(data)
        result.append(average)
    ndata = np.array(result)
    naverage = np.average(ndata)
    var = np.var(ndata)
    nmse = normalized_mean_square_error(n, tv, result)
    return {"average": naverage, "var": var, "nmse": nmse}


def cc_average_plot(graph, name, tv=None):
    import matplotlib.pyplot as plt
    num = [100, 200, 300, 500, 1000, 2000, 3000, 5000]
    if tv is None:
        raise RuntimeError("true value is None.")
    tvs = [tv] * len(num)
    metropolis = []
    proposal = []
    for n in num:
        proposal_average = random_walk_proposal_aggregation(
            graph=graph,
            size=n,
            tv=tv,
            n=100
        ).get("average")
        metropolis_average = random_walk_aggregation(
            graph=graph,
            size=n,
            tv=tv,
            metropolized=True,
            n=100
        ).get("average")
        proposal.append(proposal_average)
        metropolis.append(metropolis_average)
    plt.title("Average of Cluster Coefficient Plotting - " + name)
    plt.xlabel("Sampled Size of the Graph")
    plt.ylabel("Average of Cluster Coefficient")
    plt.plot(num, proposal, label="proposal")
    plt.plot(num, metropolis, label="metropolis")
    plt.plot(num, tvs, label="true value")
    leg = plt.legend(loc="lower right")
    leg.get_frame().set_alpha(0.5)
    plt.savefig("data/output/cc_" + name + "2.png")
    plt.show()


if __name__ == '__main__':
    amazon = nx.read_edgelist('data/input/com-amazon.ungraph.txt')
    amazon_tv = 0.3967
    enron = nx.read_edgelist('data/input/email-Enron.txt')
    enron_tv = 0.4970
    youtube = nx.read_edgelist('data/input/com-youtube.ungraph.txt')
    youtube_tv = 0.0808
    facebook = nx.read_edgelist('data/input/facebook_combined.txt')
    facebook_tv = 0.6055
    cc_average_plot(facebook, 'Facebook', facebook_tv)
