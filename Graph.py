import networkx as nx
import matplotlib.pyplot as plt
import random
import collections
from collections import Counter
from networkx.readwrite import json_graph
from random_walk import *
from pagerank_sampling import *
from util import *
import argparse
from collections import defaultdict
from scipy import stats

"""
#simple create
def create_main_graph():
    Graph_Main= nx.dense_gnm_random_graph(20, 100)
    nx.write_edgelist(Graph_Main,"data/input/"+ "edgelistMain")
    Graph_Main=nx.read_edgelist("data/input/"+"edgelistMain")
    return Graph_Main
"""

def create_main_graph(dataset):
    g_partition = nx.Graph()
    g_partition = nx.random_partition_graph([10, 10, 10, 1], 0.25, 0.5, False)
    nx.write_edgelist(g_partition,"data/input/"+dataset ,data = False)
    g_partition=nx.read_edgelist("data/input/"+dataset,nodetype= int)
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

def sampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    return list(random_walk(graph=G, size=1000))

def PRsampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    return list(page_rank_sampling(G,size = 1000))

def FFsampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    for j, fraction in enumerate(range(1, 10)):
        fraction = float(fraction) / 10.0
    return create_ff_graph(G,fraction)

def ESisampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    for j, fraction in enumerate(range(1, 10)):
        fraction = float(fraction) / 10.0
    return create_ESi_graph(G,fraction)

def create_ff_graph(graph, sampling_fraction, geometric_dist_param=0.7):
    sampled_graph = nx.Graph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_nodes = graph.nodes()
    np.random.shuffle(shuffled_graph_nodes)
    already_visited = dict()

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        burn_seed_node = shuffled_graph_nodes[0]
        shuffled_graph_nodes = shuffled_graph_nodes[1:]

        if burn_seed_node in already_visited:
            continue

        already_visited[burn_seed_node] = 1

        num_edges_to_burn = np.random.geometric(p=geometric_dist_param)
        neighbors_to_burn = graph.neighbors(burn_seed_node)[:num_edges_to_burn]
        np.random.shuffle(neighbors_to_burn)
        burn_queue = []

        for n in neighbors_to_burn:
            if burn_seed_node != n:
                sampled_graph.add_edge(burn_seed_node, n)
                burn_queue.append(n)

        while len(burn_queue) > 0:
            burn_seed_node = burn_queue[0]
            burn_queue = burn_queue[1:]

            if burn_seed_node in already_visited:
                continue

            already_visited[burn_seed_node] = 1

            num_edges_to_burn = np.random.geometric(p=geometric_dist_param)

            neighbors_to_burn = graph.neighbors(burn_seed_node)[:num_edges_to_burn]
            np.random.shuffle(neighbors_to_burn)

            for n in neighbors_to_burn:
                if burn_seed_node != n:
                    sampled_graph.add_edge(burn_seed_node, n)
                    burn_queue.append(n)

    return sampled_graph


def create_ESi_graph(graph, sampling_fraction):
    sampled_graph = nx.Graph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_edges = graph.edges()
    np.random.shuffle(shuffled_graph_edges)

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        u, v = shuffled_graph_edges[0]
        shuffled_graph_edges = shuffled_graph_edges[1:]

        if u != v:
            sampled_graph.add_edge(u, v)

    for u, v in graph.edges():
        if sampled_graph.has_node(u) and sampled_graph.has_node(v) and (not sampled_graph.has_edge(u, v)):
            if u != v:
                sampled_graph.add_edge(u, v)

    return sampled_graph
    


def degree_histogram(G,k,name):
    Gdeg = []
    for key in G.degree().keys():
        Gdeg.append(G.degree()[key])

    degree_sequence = sorted(Gdeg, reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    ax= plt.subplot(2,2,k)
    #fig, ax = plt.subplots()
    plt.bar(deg, cnt,tick_label = None, width=0.80, color='b')
    plt.title(name)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([])
    ax.set_xticklabels(deg)
    """
    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
    pos = nx.spring_layout(G)
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    """
    plt.subplots_adjust(top = 0.8, hspace=0.6,wspace=0.4)




def degree_histogram_Sampling(G,k,name):
    degree_sequence = sorted(G, reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    ax= plt.subplot(2,2,k)
    #fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title(name)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([])
    ax.set_xticklabels(deg)
    plt.subplots_adjust(top = 0.8,hspace=0.6,wspace=0.4)

def create_random_walk_graph(dataset):
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    my_graph= []
    samp = sampling(dataset)
    for a in samp:
        my_graph.append(G.degree()[a])
    return my_graph

def create_PR_walk_graph(dataset):
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    my_graph= []
    samp = PRsampling(dataset)
    for a in samp:
        my_graph.append(G.degree()[a])
    return my_graph

def show_random_graphs(Graph_Main,dataset):
    plt.suptitle("Random division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(g1,2,"First Graph")
    degree_histogram(g2,3,"Second Graph")
    degree_histogram(g3,4,"Third Graph")
    plt.savefig("data/output/Random/"+"Random "+dataset + ".png")
    plt.show()

def show_random_walk_graphs(Graph_Main,dataset):
    plt.suptitle("Random walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram_Sampling(create_random_walk_graph(dataset),2,"Random walk 1")
    degree_histogram_Sampling(create_random_walk_graph(dataset),3,"Random walk 2")
    degree_histogram_Sampling(create_random_walk_graph(dataset),4,"Random walk 3")
    plt.savefig("data/output/RW/"+"Random walk "+dataset + ".png")
    plt.show()

def show_PR_walk_graphs(Graph_Main,dataset):
    plt.suptitle("Page Rank walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram_Sampling(create_PR_walk_graph(dataset),2,"Page Rank walk 1")
    degree_histogram_Sampling(create_PR_walk_graph(dataset),3,"Page Rank walk 2")
    degree_histogram_Sampling(create_PR_walk_graph(dataset),4,"Page Rank walk 3")
    plt.savefig("data/output/PRW/"+"Page Rank walk "+dataset + ".png")
    plt.show()

def show_FF_graphs(Graph_Main,dataset):
    plt.suptitle("Forest Fire division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(FFsampling(dataset),2,"Forest Fire 1")
    degree_histogram(FFsampling(dataset),3,"Forest Fire 2")
    degree_histogram(FFsampling(dataset),4,"Forest Fire 3")
    plt.savefig("data/output/FF/"+"Forest Fire "+ dataset + ".png")
    plt.show()

def show_ESi_graphs(Graph_Main,dataset):
    plt.suptitle("Induced edges division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(ESisampling(dataset),2,"Induced edges 1")
    degree_histogram(ESisampling(dataset),3,"Induced edges 2")
    degree_histogram(ESisampling(dataset),4,"Induced edges 3")
    plt.savefig("data/output/ESi/"+"Induced edges "+ dataset + ".png")
    plt.show()


G1 = []
G2 = []
G3 = []
Random_Graph = [G1, G2, G3]

#dataset = "edgelistMain"
#Graph_Main = create_main_graph(dataset)


#dataset = "email-Eu-core.txt"
#dataset = "p2p-Gnutella04.txt"
#dataset = "p2p-Gnutella08.txt"
#dataset = "ca-HepTh.txt"
#dataset = "ca-GrQc.txt"
dataset = "facebook_combined.txt"
#dataset = "p2p-Gnutella09.txt"

Graph_Main = nx.read_edgelist("data/input/"+dataset, nodetype=int)

Random_Graph = divide_graph(Graph_Main)
g1 = create_and_save_subgraph(G1, 1)
g2 = create_and_save_subgraph(G2, 2)
g3 = create_and_save_subgraph(G3, 3)

show_FF_graphs(Graph_Main,dataset)
#show_ESi_graphs(Graph_Main,dataset)
#show_random_graphs(Graph_Main,dataset)
#show_random_walk_graphs(Graph_Main,dataset)
#show_PR_walk_graphs(Graph_Main,dataset)
