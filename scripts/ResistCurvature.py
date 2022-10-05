import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


G = nx.karate_club_graph()

def ResistanceCurvature(graph):
    G = graph.copy()
    if not nx.get_edge_attributes(G, "weight"):
        for (v1, v2) in G.edges():
            G[v1][v2]["weight"] = 1.0
            
    n = G.number_of_nodes()
    Q = nx.laplacian_matrix(G)
    Qdag = np.linalg.pinv(Q.toarray())

    eye = np.identity(n)
    for (i,j) in G.edges:
        ei = eye[i]
        ej = eye[j]
        wij =(ei-ej)@Qdag@(ei-ej)
        G[i][j]['effectiveRes'] = wij
        G[i][j]['relativeRes'] = wij*G[i][j]["weight"]
        

    for i in G.nodes():
        sums = np.sum([G[i][j]['relativeRes'] for j in G.neighbors(i)])
        G.nodes[i]['resist'] = 1-.5*sums
        

    for (u,v) in G.edges():
        G[u][v]['resistCurvature'] = 2*(G.nodes[u]['resist'] + G.nodes[v]['resist'])/G[u][v]['effectiveRes']
    return G



G_res = ResistanceCurvature(G)

# nx.get_node_attributes(G_res,'resist')
