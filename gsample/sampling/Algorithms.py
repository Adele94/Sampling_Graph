from gsample.sampling.FF import FF
from gsample.sampling.ESi import ESi
from gsample.sampling.PR import PR
from gsample.sampling.RW import RW
import networkx as nx


def RWsampling(G,sampling_fraction):
    my_graph = nx.Graph()
    samp = RW(graph=G, size=int(G.number_of_nodes() * sampling_fraction))
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph

def PRsampling(G,sampling_fraction):
    my_graph = nx.Graph()
    samp = PR(graph=G, size=int(G.number_of_nodes() * sampling_fraction))
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph

def FFsampling(G,size_fraction):
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
    return FF(G,fraction)

def ESisampling(G,size_fraction):
    for j, fraction in enumerate(range(1,size_fraction)):
        fraction = float(fraction) / size_fraction
    return ESi(G,fraction)

