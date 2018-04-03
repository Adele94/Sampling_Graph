import pandas as pd
from scipy import stats
from Algorithms import *
from collections import defaultdict

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

def eigenvalues(graph,size):
    try:
        import numpy.linalg as linal
        eigenvalues = linal.eigvals
    except ImportError:
        raise ImportError("numpy can not be imported.")

    L = nx.normalized_laplacian_matrix(graph)
    eigen_values = eigenvalues(L.A)

    return sorted(eigen_values, reverse=True)[:size]

def normalized_L1(p, q):  #normalized L1 distance
    size = len(p)
    return sum([float(np.abs(p[i] - q[i])) / float(p[i]) for i in range(size)]) / float(size)


def metrics(dataset, size_fraction):
    Graph_Main = nx.read_edgelist("data/input/" + dataset, nodetype=int)
    size_eigenvalue = int(Graph_Main.number_of_nodes() / size_fraction)
    degree_cdf_graph = degree_cdf(Graph_Main)
    cc_cdf_graph = clus_coeff_cdf(Graph_Main)
    ev_graph = eigenvalues(Graph_Main, size_eigenvalue)

    print('Eigen Values Computed')
    deg_mean = np.zeros((5, size_fraction - 1))
    cc_mean = np.zeros((5, size_fraction - 1))
    ev_mean = np.zeros((5, size_fraction - 1))  # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D
    for j in range(0, size_fraction - 1):
        deg_mean[0][j] = ((j + 1) / size_fraction)
        cc_mean[0][j] = deg_mean[0][j]
        ev_mean[0][j] = deg_mean[0][j]


    print('Random walk')
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []
        for iter_no in range(5):
            ff_sampled_graph = create_random_walk_graph(dataset, fraction)
            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff,
                                  degree_cdf_graph)  # ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph, size_eigenvalue)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[3][j] = np.mean(deg)
        cc_mean[3][j] = np.mean(cc)
        ev_mean[3][j] = np.mean(ev)

    print('Forest Fire')
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []

        for iter_no in range(5):
            ff_sampled_graph = FFsampling(dataset, size_fraction)
            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff,
                                  degree_cdf_graph)  # ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph, size_eigenvalue)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[1][j] = np.mean(deg)
        cc_mean[1][j] = np.mean(cc)
        ev_mean[1][j] = np.mean(ev)

    print('Induced Edges')
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []

        for iter_no in range(5):
            ff_sampled_graph = ESisampling(dataset, size_fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph, size_eigenvalue)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[2][j] = np.mean(deg)
        cc_mean[2][j] = np.mean(cc)
        ev_mean[2][j] = np.mean(ev)

    print('Page Rank walk')
    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []

        for iter_no in range(5):
            ff_sampled_graph = create_PR_walk_graph(dataset, fraction)
            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff,
                                  degree_cdf_graph)  # ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph, size_eigenvalue)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[4][j] = np.mean(deg)
        cc_mean[4][j] = np.mean(cc)
        ev_mean[4][j] = np.mean(ev)

    df = pd.DataFrame(deg_mean)
    dc = pd.DataFrame(cc_mean)
    de = pd.DataFrame(ev_mean)
    df.to_csv("Degree.csv", header="Errors")
    dc.to_csv("CC.csv", header="Errors")
    de.to_csv("Ev.csv", header="Errors")

