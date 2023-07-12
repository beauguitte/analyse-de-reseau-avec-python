#transformer les mesures en attributs
deg = dict(G.degree)
degnor = nx.degree_centrality(G)
degwei = dict(G.degree(weight = 'weight'))

# ajout comme attributs
nx.set_node_attributes(G, deg2, "degré")
nx.set_node_attributes(G, degnor, "degré normalisé")
nx.set_node_attributes(G, degwei, "degré pondéré")

# contrôle
list(list(G.nodes(data=True))[0][-1].keys())

#pour étudier les relations entre attributs
#créer un dataframe avec sommets et attributs
SOM = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
SOM.dtypes
SOM.head(5)
SOM.describe()
plt.scatter(SOM["P19_POP"], SOM["degré"], s=50, alpha=0.6, edgecolor='black', linewidth=1)

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_degree_connectivity.html
#nx.average_degree_connectivity(G0)
#nx.average_degree_connectivity(G0, weight="weight")

# mesures locales - éloignement (réseau connexe)
nx.eccentricity(G0)

# longueur liens entre chaque sommet et tous les autres
nx.all_pairs_node_connectivity(G0)

#cycles
#sorted(nx.simple_cycles(G0u))

# simplifier le graphe ?
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.summarization.snap_aggregation.html#networkx.algorithms.summarization.snap_aggregation
#•GA = nx.snap_aggregation(G0u, node_attributes="MARS")

#détection de communautés - https://networkx.org/documentation/stable/reference/algorithms/community.html
louv = nx.community.louvain_communities(G0, seed=123)
louv[1]

#https://iut-info.univ-reims.fr/users/blanchard/ISN20181218/utilisation-de-la-bibliotheque-networkx.html

#mesure de la modularité
nx.community.modularity(G0, louv)
#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.partition_quality.html#networkx.algorithms.community.quality.partition_quality
#nx.partition_quality(G0, louv)# error

louv = nx.community.louvain_communities(G0u, seed=123)
len(louv)

# https://python-louvain.readthedocs.io/en/latest/api.html
# import community as community_louvain
# import matplotlib.cm as cm

# partition = community_louvain.best_partition(GU)

# visualisation de la partition proposée
pos = nx.spring_layout(GD)
cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(GD, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(GD, pos, alpha=0.5)
# community_louvain.modularity(GU, partition)

# visualisation de greed
partition = nx.community.louvain_communities(GD, seed=123)
couleurs_num = [0] * GD.number_of_nodes()
for i in range(len(partition)):
    for j in partition[i]:
        couleurs_num[j] = i      # erreur
 
pos = nx.spring_layout(GD)
cmap = plt.get_cmap('viridis')
nx.draw_networkx_nodes(GD, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=couleurs_num)
nx.draw_networkx_edges(GD, pos, alpha=0.5)
couleurs_num = [0] * g.number_of_nodes()
for i in range(len(partition)):
  for j in partition[i]:
    couleurs_num[j] = i
options = {
      'cmap'       : plt.get_cmap('jet'), 
      'node_color' : couleurs_num,
      'node_size'  : 550,
      'edge_color' : 'tab:grey',
      'with_labels': True
    }
plt.figure()
nx.draw(g,**options)

###############
# visualisation
# taille des sommets et couleur des liens

d = dict(G0u.degree)
#♦ création 
edges,weights = zip(*nx.get_edge_attributes(G0u,'weight').items())

nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_cmap=plt.cm.Blues,
               edge_color=weights, 
               width=4)

# autre syntaxe possible - plus simple
weights = [G0u[u][v]['weight'] for u,v in edges]

nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_cmap=plt.cm.Blues,
               edge_color=weights, 
               width=3)

# faire varier l'épaisseur
nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_color="blue", 
               width=weights)

# modifier les valeurs des intensités
weights = [G0u[u][v]['weight'] for u,v in edges]

weigh2 = [i/50 for i in weights]

nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_color="blue", 
               width=weigh2)

# https://python.doctor/page-apprendre-listes-list-tableaux-tableaux-liste-array-python-cours-debutant
#,             style = 'dashed', 
#              label = "Spring"),
#              edge_color = 'blue',

# communautés = sommets de couleurs différentes (G0, louv)
#https://graphsandnetworks.com/community-detection-using-networkx/
# créer l'objet par taille décroissante
communities = sorted(louv, key=len, reverse=True)

# fonction qui crée un attribut communauté pour les sommets
def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G0.nodes[v]['community'] = c + 1

# fonction qui génère une couleur pour chaque communauté
def get_color(i, r_off=1, g_off=1, b_off=1): 
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

set_node_community(G0, communities)
node_color = [get_color(G0.nodes[v]['community']) for v in G0]

# layout
G0_pos = nx.spring_layout(G0)

# Draw nodes
nx.draw_networkx(
    G0,
    pos=G0_pos,
    node_size = 20,
    with_labels=False,
    node_color=node_color,
    edge_color="silver")

# colorer les liens : si intra communautés et si extra
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0

# Set node and edge communities
set_edge_community(G0)

# Set community color for edges between members of the same community (internal) and intra-community edges (external)
external = [(v, w) for v, w in G0.edges if G0.edges[v, w]['community'] == 0] #too many values to unpack (expected 2)
internal = [(v, w) for v, w in G0.edges if G0.edges[v, w]['community'] > 0]
internal_color = ['black' for e in internal]

# Draw external edges
plt.rcParams.update({'figure.figsize': (15, 10)})
nx.draw_networkx(
    G0,
    pos=G0_pos,
    node_size=0,
    with_labels=False,
    edgelist=external,
    edge_color="silver")
# Draw nodes and internal edges
nx.draw_networkx(
    G0,
    pos=G0_pos,
    node_size = 20,
    with_labels=False,
    node_color=node_color,
    edgelist=internal,
    edge_color=internal_color)

# autres options de visualisation

# intensité notée sur lien https://stackoverflow.com/questions/57421372/display-edge-weights-on-networkx-graph
pos = nx.spring_layout(G0, k=5)
nx.draw(G0, pos, with_labels=False)
labels = {e: G0.edges[e]['weight'] for e in G0.edges}
nx.draw_networkx_edge_labels(G0, pos, edge_labels=labels)
plt.show()

#https://trenton3983.github.io/files/projects/2020-05-21_intro_to_network_analysis_in_python/2020-05-21_intro_to_network_analysis_in_python.html

#matrice
import nxviz as nv
h = nv.MatrixPlot(G0)

# Draw the MatrixPlot to the screen
h.draw()
plt.show()

nv.ArcPlot(G0)
nv.CircosPlot(G0)
