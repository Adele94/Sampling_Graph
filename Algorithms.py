from random_walk import *
from pagerank_sampling import *




def RWsampling(dataset,sampling_fraction):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    return random_walk(graph=G, size=sampling_fraction)

def PRWsampling(dataset,sampling_fraction):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    return page_rank_sampling(G,size = sampling_fraction)

def FFsampling(dataset,size_fraction):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
    return create_ff_graph(G,fraction)

def ESisampling(dataset,size_fraction):
    G = nx.Graph()
    G = nx.read_edgelist("data/input/"+dataset, nodetype=int)
    for j, fraction in enumerate(range(1,size_fraction)):
        fraction = float(fraction) / size_fraction
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

def create_random_walk_graph(dataset,sampling_fraction):
    G = nx.read_edgelist("data/input/" +dataset, nodetype=int)
    my_graph = nx.Graph()
    samp = RWsampling(dataset,int(G.number_of_nodes() * sampling_fraction))
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph

def create_PR_walk_graph(dataset,sampling_fraction):
    G = nx.read_edgelist("data/input/" +dataset, nodetype=int)
    my_graph = nx.Graph()
    samp = PRWsampling(dataset,int(G.number_of_nodes() * sampling_fraction))
    for a in samp:
        my_graph.add_path(G.adj[a])
    return my_graph
