from gsample.evaluation.metrics import *
from gsample.util import *


def metrics(Graph_Main, size_fraction):
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
            ff_sampled_graph = RWsampling(Graph_Main, fraction)
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
            ff_sampled_graph = FFsampling(Graph_Main, size_fraction)
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
            ff_sampled_graph = ESisampling(Graph_Main, size_fraction)

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
            ff_sampled_graph = PRsampling(Graph_Main, fraction)
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


if __name__ == '__main__':
    dataset = "ca-GrQc.txt"
    # dataset = "p2p-Gnutella04.txt"
    # dataset = "p2p-Gnutella08.txt"
    # dataset = "ca-HepTh.txt"
    # dataset = "p2p-Gnutella25.txt"
    # dataset = "p2p-Gnutella09.txt"

    size_fraction = 10

    Graph_Main = nx.read_edgelist("gsample/data/input/" + dataset, nodetype=int)
    """
    show_FF_graphs(Graph_Main, size_fraction, dataset)
    show_ESi_graphs(Graph_Main, size_fraction, dataset)
    show_random_walk_graphs(Graph_Main, size_fraction, dataset)
    show_PR_walk_graphs(Graph_Main, size_fraction, dataset)

    metrics(Graph_Main, size_fraction)

    loaded_csv1 = pd.read_csv("Degree.csv")
    loaded_csv2 = pd.read_csv("CC.csv")
    loaded_csv3 = pd.read_csv("Ev.csv")

    error_graph(loaded_csv1, "Degree")
    error_graph(loaded_csv2, "CC")
    error_graph(loaded_csv3, "Ev")
    """
    SepList = separate_file("gsample/data/input/ca-GrQc.txt", 5)
    d = SepList[0].values.tolist()
    print(d)
