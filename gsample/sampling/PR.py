import networkx as nx


# Page Rank Sampling
def PR(graph, size=2000):
    pg = sorted(nx.pagerank(graph).items(), key=lambda x: x[1])
    pg.reverse()
    nodes = list(map(lambda x: x[0], pg))[0:size]
    return nodes