import pandas as pd
import collections

from Metrics import metrics
from util import *

dataset = "email-Eu-core.txt"
#dataset = "p2p-Gnutella04.txt"
#dataset = "p2p-Gnutella08.txt"
#dataset = "ca-HepTh.txt"
#dataset = "ca-GrQc.txt"
#dataset = "p2p-Gnutella25.txt"
#dataset = "p2p-Gnutella09.txt"

size_fraction = 10

show_FF_graphs(dataset,size_fraction)
show_ESi_graphs(dataset,size_fraction)
show_random_walk_graphs(dataset,size_fraction)
show_PR_walk_graphs(dataset,size_fraction)

metrics(dataset, size_fraction)

loaded_csv1 = pd.read_csv("Degree.csv")
loaded_csv2 = pd.read_csv("CC.csv")
loaded_csv3 = pd.read_csv("Ev.csv")

error_graph(loaded_csv1, "Degree")
error_graph(loaded_csv2, "CC")
error_graph(loaded_csv3, "Ev")


