import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
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
df = pd.read_csv('datasets/twin.txt', sep='\t')
modelstring = 'Sister cities'

G = nx.from_pandas_edgelist(df, edge_attr=None, create_using=Graphtype)

# WEIGHTED
# df = pd.read_csv('datasets/lesmis.txt', sep=' ')
# modelstring = 'Les miserables'
# df = pd.read_csv('datasets/beach.txt', sep=' ')
# modelstring = 'Windsurfers'

# G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=Graphtype)



G = nx.convert_node_labels_to_integers(G)








sampsize = 1

# SPEARMAN
# Edge curvature
spear_orc_frc = np.zeros(sampsize)
spear_orc_afrc = np.zeros(sampsize)
spear_orc_res = np.zeros(sampsize)
spear_afrc_res = np.zeros(sampsize)
spear_frc_res = np.zeros(sampsize)

# Node curvature
spear_node_orc_frc = np.zeros(sampsize)
spear_node_orc_afrc = np.zeros(sampsize)
spear_node_orc_res = np.zeros(sampsize)
spear_node_afrc_res = np.zeros(sampsize)
spear_node_frc_res = np.zeros(sampsize)


# Other measures
spear_ebc_res = np.zeros(sampsize)
spear_disp_res = np.zeros(sampsize)

spear_degs_res = np.zeros(sampsize)
spear_bc_res = np.zeros(sampsize)
spear_clust_res = np.zeros(sampsize)



# PEARSON
# Edge curvature
pears_orc_frc = np.zeros(sampsize)
pears_orc_afrc = np.zeros(sampsize)
pears_orc_res = np.zeros(sampsize)
pears_afrc_res = np.zeros(sampsize)
pears_frc_res = np.zeros(sampsize)

# Node curvature
pears_node_orc_frc = np.zeros(sampsize)
pears_node_orc_afrc = np.zeros(sampsize)
pears_node_orc_res = np.zeros(sampsize)
pears_node_afrc_res = np.zeros(sampsize)
pears_node_frc_res = np.zeros(sampsize)


# Other measures
pears_ebc_res = np.zeros(sampsize)
pears_disp_res = np.zeros(sampsize)

pears_degs_res = np.zeros(sampsize)
pears_bc_res = np.zeros(sampsize)
pears_clust_res = np.zeros(sampsize)


for i in range(sampsize):
    # MODEL NETWORKS
    # G = nx.erdos_renyi_graph(1000, 0.003)
    # modelstring = 'ER n=1000 p=0.003'
    # G = nx.erdos_renyi_graph(1000, 0.007)
    # modelstring = 'ER n=1000 p=0.007'
    # G = nx.erdos_renyi_graph(1000, 0.01)
    # modelstring = 'ER n=1000 p=0.01'
    
    # G = nx.watts_strogatz_graph(1000, 2, .5)
    # modelstring = 'WS n=1000 k=2 p=0.5'
    # G = nx.watts_strogatz_graph(1000, 8, .5)
    # modelstring = 'WS n=1000 k=8 p=0.5'
    # G = nx.watts_strogatz_graph(1000, 10, .5)
    # modelstring = 'WS n=1000 k=10 p=0.5'
    
    # G = nx.barabasi_albert_graph(1000, 2)
    # modelstring = 'BA n=1000 m=2'
    # G = nx.barabasi_albert_graph(1000, 4)
    # modelstring = 'BA n=1000 m=4'
    # G = nx.barabasi_albert_graph(1000, 5)
    # modelstring = 'BA n=1000 m=5'
    
    # Uncomment the following to apply weighting scheme
    # G = WeightScheme(G)
    
    # Ollivier curvature
    orc = OllivierRicci(G)
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()
    curvvals_orc = nx.get_edge_attributes(G_orc, "ricciCurvature")
    edge_curvvals_orc = np.array(list(curvvals_orc.values()))
    node_curvvals_orc = NodeCurvature(G_orc,"ricciCurvature")
    
    # Forman curvature
    afrc = FormanRicci(G)
    afrc.compute_ricci_curvature()
    G_afrc = afrc.G.copy()
    curvvals_afrc = nx.get_edge_attributes(G_afrc, "formanCurvature")
    edge_curvvals_afrc = np.array(list(curvvals_afrc.values()))
    node_curvvals_afrc = NodeCurvature(G_afrc,"formanCurvature")
    
    # Augmented Forman
    frc = FormanRicci(G,method="1d")
    frc.compute_ricci_curvature()
    G_frc = frc.G.copy()
    curvvals_frc = nx.get_edge_attributes(G_frc, "formanCurvature")
    edge_curvvals_frc = np.array(list(curvvals_frc.values()))
    node_curvvals_frc = NodeCurvature(G_frc,"formanCurvature")
    
    # Resistance and link resistance
    G_res = ResistanceCurvature(G)
    curvvals_res = nx.get_edge_attributes(G_res, "resistCurvature")
    edge_curvvals_res = np.array(list(curvvals_res.values()))
    curvvals_res = nx.get_node_attributes(G_res,"resist")
    node_curvvals_res = np.array(list(curvvals_res.values()))


    # Other measures
    # Edge based measures of centrality
    ebc = np.array(list(nx.edge_betweenness_centrality(G,weight="weight").values()))
    disp = np.array([nx.dispersion(G,u,v) for (u,v) in G.edges])
    
    # Node based measures of centrality
    degs = np.array(list(dict(nx.degree(G,weight="weight")).values()))
    bc = np.array(list(nx.betweenness_centrality(G,weight="weight").values()))
    clust = np.array(list(nx.clustering(G,weight="weight").values()))
    
    
    
    # Calculating correlations:
    # Spearman
    spear_orc_frc[i] = spearmanr(edge_curvvals_orc,edge_curvvals_frc)[0]
    spear_orc_afrc[i] = spearmanr(edge_curvvals_orc,edge_curvvals_afrc)[0]
    spear_orc_res[i] = spearmanr(edge_curvvals_orc,edge_curvvals_res)[0]
    spear_afrc_res[i] = spearmanr(edge_curvvals_afrc,edge_curvvals_res)[0]
    spear_frc_res[i] = spearmanr(edge_curvvals_frc,edge_curvvals_res)[0]
    
    spear_node_orc_frc[i] = spearmanr(node_curvvals_orc,node_curvvals_frc)[0]
    spear_node_orc_afrc[i] = spearmanr(node_curvvals_orc,node_curvvals_afrc)[0]
    spear_node_orc_res[i] = spearmanr(node_curvvals_orc,node_curvvals_res)[0]
    spear_node_afrc_res[i] = spearmanr(node_curvvals_afrc,node_curvvals_res)[0]
    spear_node_frc_res[i] = spearmanr(node_curvvals_frc,node_curvvals_res)[0]
    
    
    # Other measures
    spear_ebc_res[i] = spearmanr(ebc,edge_curvvals_res)[0]
    spear_disp_res[i] = spearmanr(disp,edge_curvvals_res)[0] 
    
    spear_degs_res[i] = spearmanr(degs,node_curvvals_res)[0]
    spear_bc_res[i] = spearmanr(bc,node_curvvals_res)[0]
    spear_clust_res[i] = spearmanr(clust,node_curvvals_res)[0]
    
    

    
    
    # Pearson
    pears_orc_frc[i] = pearsonr(edge_curvvals_orc,edge_curvvals_frc)[0]
    pears_orc_afrc[i] = pearsonr(edge_curvvals_orc,edge_curvvals_afrc)[0]
    pears_orc_res[i] = pearsonr(edge_curvvals_orc,edge_curvvals_res)[0]
    pears_afrc_res[i] = pearsonr(edge_curvvals_afrc,edge_curvvals_res)[0]
    pears_frc_res[i] = pearsonr(edge_curvvals_frc,edge_curvvals_res)[0]
    
    pears_node_orc_frc[i] = pearsonr(node_curvvals_orc,node_curvvals_frc)[0]
    pears_node_orc_afrc[i] = pearsonr(node_curvvals_orc,node_curvvals_afrc)[0]
    pears_node_orc_res[i] = pearsonr(node_curvvals_orc,node_curvvals_res)[0]
    pears_node_afrc_res[i] = pearsonr(node_curvvals_afrc,node_curvvals_res)[0]
    pears_node_frc_res[i] = pearsonr(node_curvvals_frc,node_curvvals_res)[0]
    

    # Other measures
    pears_ebc_res[i] = pearsonr(ebc,edge_curvvals_res)[0]
    pears_disp_res[i] = pearsonr(disp,edge_curvvals_res)[0] 
    
    pears_degs_res[i] = pearsonr(degs,node_curvvals_res)[0]
    pears_bc_res[i] = pearsonr(bc,node_curvvals_res)[0]
    pears_clust_res[i] = pearsonr(clust,node_curvvals_res)[0]
    
    print("Sample " + str(i) + " completed")


# Averaging over the samples
# Spearman correlation
# Edge curvature
avg_spear_orc_frc = np.average(spear_orc_frc)
avg_spear_orc_afrc = np.average(spear_orc_afrc)
avg_spear_orc_res = np.average(spear_orc_res)
avg_spear_afrc_res = np.average(spear_afrc_res)
avg_spear_frc_res = np.average(spear_frc_res)

# Node curvature
avg_spear_node_orc_frc = np.average(spear_node_orc_frc)
avg_spear_node_orc_afrc = np.average(spear_node_orc_afrc)
avg_spear_node_orc_res = np.average(spear_node_orc_res)
avg_spear_node_afrc_res = np.average(spear_node_afrc_res)
avg_spear_node_frc_res = np.average(spear_node_frc_res)


# Other measures
avg_spear_ebc_res = np.average(spear_ebc_res)
avg_spear_disp_res = np.average(spear_disp_res)

avg_spear_degs_res = np.average(spear_degs_res)
avg_spear_bc_res = np.average(spear_bc_res)
avg_spear_clust_res = np.average(spear_clust_res)


# Pearson correlation
# Edge curvature
avg_pears_orc_frc = np.average(pears_orc_frc)
avg_pears_orc_afrc = np.average(pears_orc_afrc)
avg_pears_orc_res = np.average(pears_orc_res)
avg_pears_afrc_res = np.average(pears_afrc_res)
avg_pears_frc_res = np.average(pears_frc_res)

# Node curvature
avg_pears_node_orc_frc = np.average(pears_node_orc_frc)
avg_pears_node_orc_afrc = np.average(pears_node_orc_afrc)
avg_pears_node_orc_res = np.average(pears_node_orc_res)
avg_pears_node_afrc_res = np.average(pears_node_afrc_res)
avg_pears_node_frc_res = np.average(pears_node_frc_res)

# Other measures
avg_pears_ebc_res = np.average(pears_ebc_res)
avg_pears_disp_res = np.average(pears_disp_res)

avg_pears_degs_res = np.average(pears_degs_res)
avg_pears_bc_res = np.average(pears_bc_res)
avg_pears_clust_res = np.average(pears_clust_res)






## SAVING RESULTS
with open('results.csv','a') as f:
    f.write("\n")
    f.write(modelstring + ',' +
        str(round(avg_spear_orc_frc,2)) + ',' +
        str(round(avg_spear_orc_afrc,2)) + ',' +
        str(round(avg_spear_orc_res,2)) + ',' +
        str(round(avg_spear_frc_res,2)) + ',' +
        str(round(avg_spear_afrc_res,2)) + ',' +
        'edges:GAP:nodes,'+
        str(round(avg_spear_node_orc_frc,2)) + ',' +
        str(round(avg_spear_node_orc_afrc,2)) + ',' +
        str(round(avg_spear_node_orc_res,2)) + ',' +
        str(round(avg_spear_node_frc_res,2)) + ',' +
        str(round(avg_spear_node_afrc_res,2)) + ',' +
        'nodes:GAP:others,'+
        str(round(avg_spear_ebc_res,2)) + ',' +
        str(round(avg_spear_disp_res,2)) + ',' +
        str(round(avg_spear_degs_res,2)) + ',' +
        str(round(avg_spear_bc_res,2)) + ',' +
        str(round(avg_spear_clust_res,2)) + ',' +
        
        'spearman:GAP:pearson,' +
        
        str(round(avg_pears_orc_frc,2)) + ',' +
        str(round(avg_pears_orc_afrc,2)) + ',' +
        str(round(avg_pears_orc_res,2)) + ',' +
        str(round(avg_pears_frc_res,2)) + ',' +
        str(round(avg_pears_afrc_res,2)) + ',' +
        'edges:GAP:nodes,'+
        str(round(avg_pears_node_orc_frc,2)) + ',' +
        str(round(avg_pears_node_orc_afrc,2)) + ',' +
        str(round(avg_pears_node_orc_res,2)) + ',' +
        str(round(avg_pears_node_frc_res,2)) + ',' +
        str(round(avg_pears_node_afrc_res,2)) + ',' +
        'nodes:GAP:others,'+
        str(round(avg_pears_ebc_res,2)) + ',' +
        str(round(avg_pears_disp_res,2)) + ',' +
        str(round(avg_pears_degs_res,2)) + ',' +
        str(round(avg_pears_bc_res,2)) + ',' +
        str(round(avg_pears_clust_res,2)))
        
    
print("Computations complete, results saved in results.csv")



