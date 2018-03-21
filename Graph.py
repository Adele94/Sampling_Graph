import networkx as nx
import matplotlib.pyplot as plt
import random
import collections

def create_main_graph():
    Graph_Main= nx.dense_gnm_random_graph(20, 100)
    nx.write_edgelist(Graph_Main, "edgelistMain")
    Graph_Main=nx.read_edgelist("edgelistMain")
    return Graph_Main

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

def degree_histogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
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

"""
#print graphs
nx.draw(Graph_Main)
plt.savefig("graph_main.png")
plt.show()
nx.draw(g1)
plt.savefig("graph_first.png")
plt.show()
nx.draw(g2)
plt.savefig("graph_second.png")
plt.show()
nx.draw(g3)
plt.savefig("graph_third.png")
plt.show()
"""

degree_histogram(Graph_Main)
degree_histogram(g1)
degree_histogram(g2)
degree_histogram(g3)