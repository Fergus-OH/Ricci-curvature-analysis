import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


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


def NodeCurvature(G,attrstring):
    n = G.number_of_nodes()
    edgelist1 = [e for e in G.edges]
    edgelist2 = [tuple(reversed(e)) for e in G.edges]
    edgelist = edgelist1+edgelist2
    
    edgedic = {}
    for i in edgelist:  
        edgedic.setdefault(i[0],[]).append(i[1])
     
    nodecurv = np.zeros(n)
    for i in edgedic.keys():
        sumval = 0
        for j in edgedic[i]:
            sumval += G.get_edge_data(i,j)[attrstring]
        nodecurv[i] = sumval
    return nodecurv




def WeightScheme(graph):
    G = graph.copy()
    n = nx.number_of_nodes(G)
    degs = G.degree()

    # node scheme
    w  = np.zeros(n)
    for i in G.nodes():
        sumval = np.sum(np.array([G.degree(j) for j in G.neighbors(i) ]))
        if degs[i] != 0:
            w[i] = 1/degs[i]*sumval
        
    w = np.divide(w,np.max(w))
    
    for i in G.nodes():
        G.nodes[i]["weight"] = w[i]
    
        
    # edge scheme
    gamma = np.zeros((n,n))
    for (u, v) in G.edges():
        gamma[u,v] = np.sqrt(w[u]**2 + w[v]**2)
        
    gamma = np.divide(gamma,np.max(gamma))
    
    for (u, v) in G.edges():
        G.edges[u,v]["weight"] = gamma[u,v]
        
    return G


Graphtype = nx.Graph()

### REAL NETWORKS
# UNWEIGHTED
# G = nx.karate_club_graph()
# modelstring = 'Zachary karate club'
# df = pd.read_csv('datasets/euroroad.txt', sep=' ')
# modelstring = 'Euro Road'
# df = pd.read_csv('datasets/powergrid.txt', sep=' ')
# modelstring = 'US Power Grid'
# df = pd.read_csv('datasets/contiguous-usa.txt', sep=' ')
# modelstring = 'Contiguous USA States'
# df = pd.read_csv('datasets/dolphins.txt', sep='\t')
# modelstring = 'Dolphins'
# df = pd.read_csv('datasets/arenas-jazz.txt', sep='\t')
# modelstring = 'Jazz Musicians'
# df = pd.read_csv('datasets/zebra.txt', sep=' ')
# modelstring = 'Zebra'
# df = pd.read_csv('datasets/twin.txt', sep='\t')
# modelstring = 'Sister cities'

# G = nx.from_pandas_edgelist(df, edge_attr=None, create_using=Graphtype)

# WEIGHTED
# df = pd.read_csv('datasets/lesmis.txt', sep=' ')
# modelstring = 'Les miserables'
# df = pd.read_csv('datasets/beach.txt', sep=' ')
# modelstring = 'Windsurfers'

# G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=Graphtype)



# G = nx.convert_node_labels_to_integers(G)


# G = nx.erdos_renyi_graph(1000, 0.01)
G = nx.watts_strogatz_graph(1000, 100, .5)
# G = nx._graph(1000, 100)

G = WeightScheme(G)



n = G.number_of_nodes()

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[0,0].set_box_aspect(1)
ax[0,1].set_box_aspect(1)
ax[1,0].set_box_aspect(1)
ax[1,1].set_box_aspect(1)
cmap = sns.cm.rocket_r

orc = OllivierRicci(G)
orc.compute_ricci_curvature()
G_orc = orc.G.copy() 

curvvals = nx.get_edge_attributes(G_orc, "ricciCurvature")
curvmat_orc = np.zeros((n,n))
for i in curvvals.keys():
    curvmat_orc[i[0],i[1]] = curvvals[i]
    curvmat_orc[i[1],i[0]] = curvvals[i]

g1 = sns.heatmap(curvmat_orc, yticklabels=False, xticklabels=False, cmap = cmap, center=0, ax=ax[0,0])
g1.tick_params(left=False, bottom=False)
g1.invert_yaxis()
g1.title.set_text("Ollivier")



G_res = ResistanceCurvature(G)

curvvals = nx.get_edge_attributes(G_res, "resistCurvature")
curvmat_res = np.zeros((n,n))
for i in curvvals.keys():
    curvmat_res[i[0],i[1]] = curvvals[i]
    curvmat_res[i[1],i[0]] = curvvals[i]
    
g2 = sns.heatmap(curvmat_res, yticklabels=False, xticklabels=False, cmap = cmap, center=0, ax=ax[0,1])
g2.invert_yaxis()
g2.title.set_text("Link resistance")


frc = FormanRicci(G,method="1d")
frc.compute_ricci_curvature()
G_frc = frc.G.copy()

curvvals = nx.get_edge_attributes(G_frc, "formanCurvature")
curvmat_frc = np.zeros((n,n))
for i in curvvals.keys():
    curvmat_frc[i[0],i[1]] = curvvals[i]
    curvmat_frc[i[1],i[0]] = curvvals[i]
    
g3 = sns.heatmap(curvmat_frc, yticklabels=False, xticklabels=False, cmap = cmap, center=0, ax=ax[1,0])
g3.invert_yaxis()
g3.title.set_text("Forman")


afrc = FormanRicci(G)
afrc.compute_ricci_curvature()
G_afrc = afrc.G.copy()

curvvals = nx.get_edge_attributes(G_afrc, "formanCurvature")
curvmat_afrc = np.zeros((n,n))
for i in curvvals.keys():
    curvmat_afrc[i[0],i[1]] = curvvals[i]
    curvmat_afrc[i[1],i[0]] = curvvals[i]
    
g4 = sns.heatmap(curvmat_afrc, yticklabels=False, xticklabels=False, cmap = cmap, center=0, ax=ax[1,1])
g4.tick_params(left=False, bottom=False)
g4.invert_yaxis()
g4.title.set_text("Augmented Forman")
    
    
    



plt.show()
fig.savefig("output.png")







