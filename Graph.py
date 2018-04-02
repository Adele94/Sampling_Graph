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
#simple create (random graph)
def sampling(dataset):
    g_partition = nx.Graph()
    g_partition = nx.random_partition_graph([10, 10, 10, 1], 0.25, 0.5, False)
    nx.write_edgelist(g_partition,"data/input/"+dataset ,data = False)
    g_partition=nx.read_edgelist("data/input/"+dataset,nodetype= int)
    return g_partition

"""


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


def RWsampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    return random_walk(graph=G, size=1000)

def PRWsampling(dataset):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    return page_rank_sampling(G,size = 1000)

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

    shuffled_graph_nodes = list(graph.nodes())
    np.random.shuffle(shuffled_graph_nodes)
    already_visited = dict()

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        burn_seed_node = shuffled_graph_nodes[0]
        shuffled_graph_nodes = shuffled_graph_nodes[1:]

        if burn_seed_node in already_visited:
            continue

        already_visited[burn_seed_node] = 1

        num_edges_to_burn = np.random.geometric(p=geometric_dist_param)
        neighbors_to_burn = list(graph.neighbors(burn_seed_node))[:num_edges_to_burn]
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

            neighbors_to_burn = list(graph.neighbors(burn_seed_node))[:num_edges_to_burn]
            np.random.shuffle(neighbors_to_burn)

            for n in neighbors_to_burn:
                if burn_seed_node != n:
                    sampled_graph.add_edge(burn_seed_node, n)
                    burn_queue.append(n)

    return sampled_graph

def create_ESi_graph(graph, sampling_fraction):
    sampled_graph = nx.Graph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_edges = list(graph.edges())
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

def create_random_walk_graph(dataset):
    G = nx.read_edgelist("data/input/" +dataset, nodetype=int)
    my_graph = nx.Graph()
    samp = RWsampling(dataset)
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph

def create_PR_walk_graph(dataset):
    G = nx.read_edgelist("data/input/" +dataset, nodetype=int)
    my_graph = nx.Graph()
    samp = PRWsampling(dataset)
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph


def degree_histogram(G,k,name):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
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
    print (G)
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

def eigenvalues(graph):
    try:
        import numpy.linalg as linal
        eigenvalues = linal.eigvals
    except ImportError:
        raise ImportError("numpy can not be imported.")

    L = nx.normalized_laplacian_matrix(graph)
    eigen_values = eigenvalues(L.A)

    return sorted(eigen_values, reverse=True)[:25]

def normalized_L1(p, q):  #normalized L1 distance
    size = len(p)
    return sum([float(np.abs(p[i] - q[i])) / float(p[i]) for i in range(size)]) / float(size)




def show_NS_graphs(Graph_Main, dataset):
    plt.suptitle("Random division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(g1,2,"First Graph")
    degree_histogram(g2,3,"Second Graph")
    degree_histogram(g3,4,"Third Graph")
    plt.savefig("data/output/NS/"+"Random "+dataset + ".png")
    plt.show()

def show_random_walk_graphs(Graph_Main,dataset):
    plt.suptitle("Random walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(create_random_walk_graph(dataset),2,"Random walk 1")
    degree_histogram(create_random_walk_graph(dataset),3,"Random walk 2")
    degree_histogram(create_random_walk_graph(dataset),4,"Random walk 3")
    plt.savefig("data/output/RW/"+"Random walk "+dataset + ".png")
    plt.show()

def show_PR_walk_graphs(Graph_Main,dataset):
    plt.suptitle("Page Rank walk division \n dataset: " + dataset)
    degree_histogram(Graph_Main,1,"Main Graph")
    degree_histogram(create_PR_walk_graph(dataset),2,"Page Rank walk 1")
    degree_histogram(create_PR_walk_graph(dataset),3,"Page Rank walk 2")
    degree_histogram(create_PR_walk_graph(dataset),4,"Page Rank walk 3")
    plt.savefig("data/output/PRW/"+"Page Rank walk "+ dataset + ".png")
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
#Graph_Main = Nsampling(dataset)


dataset = "email-Eu-core.txt"
#dataset = "p2p-Gnutella04.txt"
#dataset = "p2p-Gnutella08.txt"
#dataset = "ca-HepTh.txt"
#dataset = "ca-GrQc.txt"
#dataset = "p2p-Gnutella25.txt"
#dataset = "p2p-Gnutella09.txt"

Graph_Main = nx.read_edgelist("data/input/"+dataset, nodetype=int)


Random_Graph = divide_graph(Graph_Main)
g1 = create_and_save_subgraph(G1, 1)
g2 = create_and_save_subgraph(G2, 2)
g3 = create_and_save_subgraph(G3, 3)

degree_cdf_graph = degree_cdf(Graph_Main)
cc_cdf_graph = clus_coeff_cdf(Graph_Main)
ev_graph = eigenvalues(Graph_Main)


print ('Eigen Values Computed')

deg_mean = np.zeros((5, 9))
cc_mean = np.zeros((5, 9))
ev_mean = np.zeros((5, 9))  # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D


print ('Forest Fire')
for j, fraction in enumerate(range(1, 10)):
    fraction = float(fraction) / 10.0
    print ('Fraction:', fraction)
    deg = []
    cc = []
    ev = []

    for iter_no in range(5):
        ff_sampled_graph = FFsampling(dataset)
        degree_cdf_ff = degree_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)  #ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
        deg.append(D)

        cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
        cc.append(D)

        ev_ff = eigenvalues(ff_sampled_graph)
        l1 = normalized_L1(ev_graph, ev_ff)
        ev.append(l1)

    deg_mean[0][j] = np.mean(deg)
    cc_mean[0][j] = np.mean(cc)
    ev_mean[0][j] = np.mean(ev)

print ('Induced Edges')
for j, fraction in enumerate(range(1, 10)):
        fraction = float(fraction) / 10.0
        print ('Fraction:', fraction)
        deg = []
        cc = []
        ev = []

        for iter_no in range(5):
            ff_sampled_graph = ESisampling(dataset)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[1][j] = np.mean(deg)
        cc_mean[1][j] = np.mean(cc)
        ev_mean[1][j] = np.mean(ev)


print ('Random walk')
for j, fraction in enumerate(range(1, 10)):
    fraction = float(fraction) / 10.0
    print ('Fraction:', fraction)
    deg = []
    cc = []
    ev = []

    for iter_no in range(5):
        ff_sampled_graph = create_random_walk_graph(dataset)
        degree_cdf_ff = degree_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)  #ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
        deg.append(D)

        cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
        cc.append(D)

        ev_ff = eigenvalues(ff_sampled_graph)
        l1 = normalized_L1(ev_graph, ev_ff)
        ev.append(l1)

    deg_mean[2][j] = np.mean(deg)
    cc_mean[2][j] = np.mean(cc)
    ev_mean[2][j] = np.mean(ev)

print ('Page Rank walk')
for j, fraction in enumerate(range(1, 10)):
    fraction = float(fraction) / 10.0
    print ('Fraction:', fraction)
    deg = []
    cc = []
    ev = []

    for iter_no in range(5):

        ff_sampled_graph = create_PR_walk_graph(dataset)
        degree_cdf_ff = degree_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)  #ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
        deg.append(D)

        cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
        cc.append(D)

        ev_ff = eigenvalues(ff_sampled_graph)
        l1 = normalized_L1(ev_graph, ev_ff)
        ev.append(l1)

    deg_mean[3][j] = np.mean(deg)
    cc_mean[3][j] = np.mean(cc)
    ev_mean[3][j] = np.mean(ev)


print ('Node')


for j, fraction in enumerate(range(1, 10)):
    fraction = float(fraction) / 10.0
    print ('Fraction:', fraction)
    deg = []
    cc = []
    ev = []

    for iter_no in range(5):
        ff_sampled_graph = g1
        degree_cdf_ff = degree_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)  #ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
        deg.append(D)

        cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
        D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
        cc.append(D)

        ev_ff = eigenvalues(ff_sampled_graph)
        l1 = normalized_L1(ev_graph, ev_ff)
        ev.append(l1)

    deg_mean[4][j] = np.mean(deg)
    cc_mean[4][j] = np.mean(cc)
    ev_mean[4][j] = np.mean(ev)


np.savetxt("data/metrics/"+'KS-Degree.txt', deg_mean) # Degree distribution
np.savetxt("data/metrics/"+'KS-CC.txt', cc_mean)  # Clustering coefficient distribution
np.savetxt("data/metrics/"+'KS-EV.txt', ev_mean)  # Eigenvalue

show_FF_graphs(Graph_Main,dataset)
show_ESi_graphs(Graph_Main,dataset)
show_NS_graphs(Graph_Main,dataset)
show_random_walk_graphs(Graph_Main,dataset)
show_PR_walk_graphs(Graph_Main,dataset)

