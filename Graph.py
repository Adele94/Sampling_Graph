import csv

from gsample.evaluation.metrics import *
from gsample.util import *


def metrics(Graph_Main, size_fraction):
    size_eigenvalue = int(Graph_Main.number_of_nodes() / size_fraction)
    degree_cdf_graph = degree_cdf(Graph_Main)
    cc_cdf_graph = clus_coeff_cdf(Graph_Main)
    ev_graph = eigenvalues(Graph_Main, size_eigenvalue)

    print('Eigen Values Computed')
    deg_mean_RW = np.zeros((1, size_fraction - 1))
    deg_mean_ESi = np.zeros((1, size_fraction - 1))
    deg_mean_PRW = np.zeros((1, size_fraction - 1))
    deg_mean_FF = np.zeros((1, size_fraction - 1))
    cc_mean_RW = np.zeros((1, size_fraction - 1))
    cc_mean_ESi = np.zeros((1, size_fraction - 1))
    cc_mean_PRW = np.zeros((1, size_fraction - 1))
    cc_mean_FF = np.zeros((1, size_fraction - 1))
    ev_mean_RW = np.zeros((1, size_fraction - 1))
    ev_mean_ESi = np.zeros((1, size_fraction - 1))
    ev_mean_PRW = np.zeros((1, size_fraction - 1))
    ev_mean_FF = np.zeros((1, size_fraction - 1))


    # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D
    for j in range(0, size_fraction - 1):
        deg_mean_RW[0][j] = ((j + 1) / size_fraction)
        deg_mean_ESi[0][j] = deg_mean_RW[0][j]
        deg_mean_FF[0][j] = deg_mean_RW[0][j]
        deg_mean_PRW[0][j] = deg_mean_RW[0][j]
        cc_mean_RW[0][j] = deg_mean_RW[0][j]
        cc_mean_PRW[0][j] = deg_mean_RW[0][j]
        cc_mean_FF[0][j] = deg_mean_RW[0][j]
        cc_mean_ESi[0][j] = deg_mean_RW[0][j]
        ev_mean_RW[0][j] = deg_mean_RW[0][j]
        ev_mean_PRW[0][j] = deg_mean_RW[0][j]
        ev_mean_FF[0][j] = deg_mean_RW[0][j]
        ev_mean_ESi[0][j] = deg_mean_RW[0][j]

    print('Random walk')

    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []
        """
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

        deg_mean_RW[j] = np.mean(deg)
        cc_mean_RW[j] = np.mean(cc)
        ev_mean_RW[j] = np.mean(ev)
        """
        deg_mean_RW[0][j] = 0
        cc_mean_RW[0][j] = 0
        ev_mean_RW[0][j] = 0

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

        deg_mean_FF[0][j] = np.mean(deg)
        cc_mean_FF[0][j] = np.mean(cc)
        ev_mean_FF[0][j] = np.mean(ev)

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

        deg_mean_ESi[0][j] = np.mean(deg)
        cc_mean_ESi[0][j] = np.mean(cc)
        ev_mean_ESi[0][j] = np.mean(ev)

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

        deg_mean_PRW[0][j] = np.mean(deg)
        cc_mean_PRW[0][j] = np.mean(cc)
        ev_mean_PRW[0][j] = np.mean(ev)

    df_RW = pd.DataFrame(deg_mean_RW)
    dc_RW = pd.DataFrame(cc_mean_RW)
    de_RW = pd.DataFrame(ev_mean_RW)
    df_FF = pd.DataFrame(deg_mean_FF)
    dc_FF = pd.DataFrame(cc_mean_FF)
    de_FF = pd.DataFrame(ev_mean_FF)
    df_ESi = pd.DataFrame(deg_mean_ESi)
    dc_ESi = pd.DataFrame(cc_mean_ESi)
    de_ESi = pd.DataFrame(ev_mean_ESi)
    df_PRW = pd.DataFrame(deg_mean_PRW)
    dc_PRW = pd.DataFrame(cc_mean_PRW)
    de_PRW = pd.DataFrame(ev_mean_PRW)


    with open('Degree_RW.csv', 'a') as f:
        df_RW.to_csv(f, header=False)
    with open('CC_RW.csv', 'a') as f:
        dc_RW.to_csv(f, header=False)
    with open('Ev_RW.csv', 'a') as f:
        de_RW.to_csv(f, header=False)
    with open('Degree_PRW.csv', 'a') as f:
        df_PRW.to_csv(f, header=False)
    with open('CC_PRW.csv', 'a') as f:
        dc_PRW.to_csv(f, header=False)
    with open('Ev_PRW.csv', 'a') as f:
        de_PRW.to_csv(f, header=False)
    with open('Degree_FF.csv', 'a') as f:
        df_FF.to_csv(f, header=False)
    with open('CC_FF.csv', 'a') as f:
        dc_FF.to_csv(f, header=False)
    with open('Ev_FF.csv', 'a') as f:
        de_FF.to_csv(f, header=False)
    with open('Degree_ESi.csv', 'a') as f:
        df_ESi.to_csv(f, header=False)
    with open('CC_ESi.csv', 'a') as f:
        dc_ESi.to_csv(f, header=False)
    with open('Ev_ESi.csv', 'a') as f:
        de_ESi.to_csv(f, header=False)

def metrics_Bitcoin(Graph_Main, size_fraction):

    SepList = separate_file("gsample/data/input/ca-GrQc.txt", size_fraction)
    di = []
    d = nx.Graph(di)

    size_eigenvalue = int(Graph_Main.number_of_nodes() / size_fraction)
    degree_cdf_graph = degree_cdf(Graph_Main)
    cc_cdf_graph = clus_coeff_cdf(Graph_Main)
    ev_graph = eigenvalues(Graph_Main, size_eigenvalue)

    print('Eigen Values Computed')
    deg_mean = np.zeros((2, size_fraction - 1))
    cc_mean = np.zeros((2, size_fraction - 1))
    ev_mean = np.zeros((2, size_fraction - 1))  # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D
    for j in range(0, size_fraction - 1):
        deg_mean[0][j] = ((j + 1) / size_fraction)
        cc_mean[0][j] = deg_mean[0][j]
        ev_mean[0][j] = deg_mean[0][j]
    print('Shuffle')

    for j, fraction in enumerate(range(1, size_fraction)):
        fraction = float(fraction) / size_fraction
        print('Fraction:', fraction)
        deg = []
        cc = []
        ev = []

        d_next = nx.from_edgelist(SepList[j-1].values.tolist())
        sh_sampled_graph = nx.disjoint_union(d,d_next)
        d = sh_sampled_graph
        degree_cdf_ff = degree_cdf(sh_sampled_graph)
        D, p = stats.ks_2samp(degree_cdf_ff,
                                  degree_cdf_graph)  # ks_2samp - Computes the Kolmogorov-Smirnov statistic on 2 samples.
        deg.append(D)

        cc_cdf_ff = clus_coeff_cdf(sh_sampled_graph)
        D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
        cc.append(D)

        ev_ff = eigenvalues(sh_sampled_graph, size_eigenvalue)
        l1 = normalized_L1(ev_graph, ev_ff)
        ev.append(l1)

        print (np.mean(deg))
        print (np.mean(cc))
        print (np.mean(ev))

        deg_mean[1][j] = np.mean(deg)
        cc_mean[1][j] = np.mean(cc)
        ev_mean[1][j] = np.mean(ev)

    df = pd.DataFrame(deg_mean)
    dc = pd.DataFrame(cc_mean)
    de = pd.DataFrame(ev_mean)
    df.to_csv("Degree.csv", header="Errors")
    dc.to_csv("CC.csv", header="Errors")
    de.to_csv("Ev.csv", header="Errors")


if __name__ == '__main__':
    #dataset = "ca-GrQc.txt"
    # dataset = "p2p-Gnutella04.txt"
    # dataset = "p2p-Gnutella08.txt"
    # dataset = "ca-HepTh.txt"
    # dataset = "p2p-Gnutella25.txt"
    # dataset = "p2p-Gnutella09.txt"
    dataset = ["ca-GrQc.txt","p2p-Gnutella04.txt","p2p-Gnutella08.txt","ca-HepTh.txt","p2p-Gnutella25.txt","p2p-Gnutella09.txt"]

    size_fraction = 10

    deg_mean_RW = np.zeros((1,size_fraction - 1))
    # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D
    for j in range(0, size_fraction - 1):
        deg_mean_RW[0][j] = ((j + 1) / size_fraction)

    df = pd.DataFrame(deg_mean_RW)
    df.to_csv("Degree_RW.csv", header="Errors")
    df.to_csv("CC_RW.csv", header="Errors")
    df.to_csv("Ev_RW.csv", header="Errors")
    df.to_csv("Degree_PRW.csv", header="Errors")
    df.to_csv("CC_PRW.csv", header="Errors")
    df.to_csv("Ev_PRW.csv", header="Errors")
    df.to_csv("Degree_FF.csv", header="Errors")
    df.to_csv("CC_FF.csv", header="Errors")
    df.to_csv("Ev_FF.csv", header="Errors")
    df.to_csv("Degree_ESi.csv", header="Errors")
    df.to_csv("CC_ESi.csv", header="Errors")
    df.to_csv("Ev_ESi.csv", header="Errors")

    count_datasets = 2
    for i in range(count_datasets):
        Graph_Main = nx.read_edgelist("gsample/data/input/" + dataset[i], nodetype=int)

        metrics(Graph_Main, size_fraction)

    loaded_csv_DR = pd.read_csv("Degree_RW.csv")
    loaded_csv_CR = pd.read_csv("CC_RW.csv")
    loaded_csv_ER = pd.read_csv("Ev_RW.csv")
    loaded_csv_DP = pd.read_csv("Degree_PRW.csv")
    loaded_csv_CP = pd.read_csv("CC_PRW.csv")
    loaded_csv_EP = pd.read_csv("Ev_PRW.csv")
    loaded_csv_DF = pd.read_csv("Degree_FF.csv")
    loaded_csv_CF = pd.read_csv("CC_FF.csv")
    loaded_csv_EF = pd.read_csv("Ev_FF.csv")
    loaded_csv_DE = pd.read_csv("Degree_ESi.csv")
    loaded_csv_CE = pd.read_csv("CC_ESi.csv")
    loaded_csv_EE = pd.read_csv("Ev_ESi.csv")

    error_graph2(loaded_csv_DR, "Degree RW", count_datasets, dataset)
    error_graph2(loaded_csv_CR, "CC RW", count_datasets, dataset)
    error_graph2(loaded_csv_ER, "Ev RW", count_datasets, dataset)
    error_graph2(loaded_csv_DP, "Degree PRW", count_datasets, dataset)
    error_graph2(loaded_csv_CP, "CC PRW", count_datasets, dataset)
    error_graph2(loaded_csv_EP, "Ev PRW", count_datasets, dataset)
    error_graph2(loaded_csv_DF, "Degree FF", count_datasets, dataset)
    error_graph2(loaded_csv_CF, "CC FF", count_datasets, dataset)
    error_graph2(loaded_csv_EF, "Ev FF", count_datasets, dataset)
    error_graph2(loaded_csv_DE, "Degree ESi", count_datasets, dataset)
    error_graph2(loaded_csv_CE, "CC ESi", count_datasets, dataset)
    error_graph2(loaded_csv_EE, "Ev ESi", count_datasets, dataset)


    """
    Graph_Main = nx.read_edgelist("gsample/data/input/" + dataset, nodetype=int)
    
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

    #metrics_Bitcoin(Graph_Main, size_fraction)
    """
    SepList = separate_file("gsample/data/input/ca-GrQc.txt", size_fraction)
    d = SepList[0].values.tolist()

    print(d)
    """


