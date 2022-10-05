import networkx as nx
import numpy as np
import pandas as pd



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





def ResistanceFlow(graph,iterations=30,step=1,delta=1e-4,verbose=False):
    G = graph.copy()
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    
    if nx.get_edge_attributes(G, "original_RC"):
        print("original_RC detected, continue to refine the ricci flow.")
    else:    
        G = ResistanceCurvature(G)
        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["resistCurvature"]
    
    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2]["effectiveRes"] -= step * 2*(G.nodes[v1]["resist"] + G.nodes[v2]["resist"])
            G[v1][v2]['relativeRes'] = G[v1][v2]["effectiveRes"]*G[v1][v2]["weight"]
    
        print("Round " + str(i))
        
        # Update resistance curvatures:
        for u in G.nodes():
            sums = np.sum([G[u][v]['relativeRes'] for v in G.neighbors(u)])
            G.nodes[u]['resist'] = 1-.5*sums
            
        for (u,v) in G.edges():
            G[u][v]['resistCurvature'] = 2*(G.nodes[u]['resist'] + G.nodes[v]['resist'])/G[u][v]['effectiveRes']
    
    
        rc = nx.get_edge_attributes(G, "resistCurvature")
    
        diff = max(rc.values()) - min(rc.values())
        print("Resist curvature difference:", diff)
        if diff < delta:
            print("Resist Curvature converged, process terminated.")
            break
    
        if verbose:
            for n1, n2 in G.edges():
                print(n1, n2, G[n1][n2])
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
df = pd.read_csv('datasets/lesmis.txt', sep=' ')
modelstring = 'Les miserables'
# df = pd.read_csv('datasets/beach.txt', sep=' ')
# modelstring = 'Windsurfers'

G = nx.from_pandas_edgelist(df, edge_attr=True, create_using=Graphtype)

G = nx.convert_node_labels_to_integers(G)




res = ResistanceCurvature(G)
G_flow = ResistanceFlow(G,100)





