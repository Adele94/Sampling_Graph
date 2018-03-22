import networkx as nx
import matplotlib.pyplot as plt
import random
import collections
from collections import Counter
from networkx.readwrite import json_graph
from random_walk import *
from util import *

"""
#simple create
def create_main_graph():
    Graph_Main= nx.dense_gnm_random_graph(20, 100)
    nx.write_edgelist(Graph_Main, "edgelistMain")
    Graph_Main=nx.read_edgelist("edgelistMain")
    return Graph_Main
"""

def create_main_graph():
    g_partition = nx.Graph()
    g_partition = nx.random_partition_graph([10, 10, 10, 1], 0.25, 0.5, False)
    nx.write_edgelist(g_partition, "edgelistMain",data = False)
    g_partition=nx.read_edgelist("edgelistMain",nodetype= int)
    return g_partition

def divide_graph(Graph_Main):
    for i in Graph_Main.edges():
        k = random.randrange(0,3)
        Random_Graph[k].append(i)
    return Random_Graph

def create_and_save_subgraph(G1, k):
    g1 = nx.Graph()
    g1.add_edges_from(G1)
    nx.write_edgelist(g1, "edgelist %s" % k)
    return g1

def sampling():
    G = nx.Graph()
    G = nx.read_edgelist("edgelistMain", nodetype=int)
    #G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (4, 7), (4, 8), (5, 9), (5, 10), (7, 11), (7, 12)])
    print(cluster_coefficient_node(G, 2))
    print(average_degree(G))
    print(list(random_walk(graph=G, size=10)))

#sampling()

def twitter_sampling():
    G = nx.read_edgelist("edgelistMain", nodetype=int)
    print(cluster_coefficient_average(G))
    print(random_walk_cca(graph=G, size=100000, metropolized=True))
    print(random_walk_aggregation(graph=G, size=10000, metropolized=True))

#twitter_sampling()

def youtube_sampling():
    G = nx.read_edgelist("edgelistMain", nodetype=int)
    print(random_walk_cca(graph=G, size=10000))
    print(random_walk_aggregation(graph=G, size=10000, metropolized=False))

#youtube_sampling()

def youtube_sampling_plot(size):
    G = nx.read_edgelist("edgelistMain", nodetype=int)
    nodes = list(random_walk(graph=G, size=size))
    graph = nx.Graph()
    graph.add_path(nodes)
    nx.draw_networkx(G=graph)
    #nx.draw_random(G=graph)
    plt.savefig("edgelistMain.ungraph.rw." + str(size) + ".png")
    plt.show()

#youtube_sampling_plot(10)


def degree_histogram(G):
    Gdeg = []
    for key in G.degree().keys():
        Gdeg.append(G.degree()[key])

    degree_sequence = sorted(Gdeg, reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    pos = nx.spring_layout(G)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()

G1 = []
G2 = []
G3 = []
Random_Graph = [G1, G2, G3]
Graph_Main = create_main_graph()

Random_Graph = divide_graph(Graph_Main)
g1 = create_and_save_subgraph(G1, 1)
g2 = create_and_save_subgraph(G2, 2)
g3 = create_and_save_subgraph(G3, 3)

degree_histogram(Graph_Main)
degree_histogram(g1)
degree_histogram(g2)
degree_histogram(g3)
