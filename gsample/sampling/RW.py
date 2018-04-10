import random
import itertools

# Random Walk sampling
def RW(graph, start_node=None, size=-1, metropolized=False):
    if start_node is None:
        start_node = random.choice(list(graph.nodes()))

    v = start_node
    for c in itertools.count():
        if c == size:
            return
        if metropolized:
            candidate = random.choice(list(graph.neighbors(v)))
            v = candidate if (random.random() < float(graph.degree(v)) / graph.degree(candidate)) else v
        else:
            v = random.choice(list(graph.neighbors(v)))

        yield v


