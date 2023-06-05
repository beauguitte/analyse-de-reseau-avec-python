import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sommets = pd.read_csv("C:/Users/Laurent/Documents/python/test_site/data/som_d13.csv", sep = ";")
liens = pd.read_csv("C:/Users/Laurent/Documents/python/test_site/data/liens_d13.csv", sep = ";")
sommets.head(5)
liens.head(5)
print(sommets.dtypes)
print(liens.dtypes)

#♣ recodage des codes insee en type str
sommets[['CODGEO']] = sommets[['CODGEO']].astype('string')
sommets[['SUPERF']] = sommets[['SUPERF']].astype('float')
sommets[['MARS']] = sommets[['MARS']].astype('bool')

liens[['Origine']] = liens[['Origine']].astype('string')
liens[['Arrivee']] = liens[['Arrivee']].astype('string')

print(sommets.dtypes)
print(liens.dtypes)

G = nx.from_pandas_edgelist(liens,source = "Origine",
                            target = "Arrivee",
                            edge_attr="weight",
                            create_using=nx.DiGraph())
nx.is_directed(G)

#ajout des attributs des sommets
#https://www.roelpeters.be/python-networkx-set-node-attributes-from-pandas-dataframe/

nodes_attr = sommets.set_index('CODGEO').to_dict(orient = 'index')
nx.set_node_attributes(G, nodes_attr)

#contrôle
G.nodes("MARS")

# afficher liste des attributs des sommets et des liens
list(list(G.edges(data=True))[0][-1].keys())
list(list(G.nodes(data=True))[0][-1].keys())

#diamètre 
#nx.diameter(G) # message d'erreur car graphe non fortement connexe

# si transformation en réseau non orienté
Gu = nx.to_undirected(G)

# les intensités des liens ne sont pas prises en compte
# controle
print("lien ij :", G["13001"]["13201"]['weight'])
print("lien ji :", G["13201"]["13001"]['weight'])
print("lien ij non orienté : ", Gu["13001"]["13201"]['weight'])

# pour créer des liens avec ij + ji
 # as_indirected  : ne permet pas agrégation des intensités
 # pour avoir ij + ji
 # https://stackoverflow.com/questions/25778137/networkx-digraph-to-graph-with-edge-weights-not-summed-how-to-sum-weights

# créer une copie sans aucun lien
GU = nx.create_empty_copy(G, with_data=True)
GU = nx.to_undirected(GU)

# contrôle
list(list(GU.nodes(data=True))[0][-1].keys())
nx.is_directed(GU)

# éviter message "Frozen graph can't be modified"
GU = nx.Graph(GU)

# récupérer liens avec intensité nulle
GU.add_edges_from(G.edges(), weight=0)

# pour chaque lien ij + ji
for u, v, d in G.edges(data=True):
    GU[u][v]['weight'] += d['weight']

# contrôle

#a attributs liens et sommets
list(list(GU.edges(data=True))[0][-1].keys())
list(list(GU.nodes(data=True))[0][-1].keys())

# propriétés du réseau
nx.is_directed(GU)
nx.is_connected(GU)

print("lien ij :", G["13001"]["13201"]['weight'])
print("lien ji :", G["13201"]["13001"]['weight'])
print("lien ij non orienté : ", GU["13001"]["13201"]['weight'])

# extraire la plus grande composante connexe
Gcc = sorted(nx.connected_components(GU), key=len, reverse=True)

# version non orientée
G0 = GU.subgraph(Gcc[0])
nx.is_directed(G0)

# attribut conservés dans la version non orientée 
list(list(G0.edges(data=True))[0][-1].keys())
list(list(G0.nodes(data=True))[0][-1].keys())

##################
# mesures globales
##################

print("densité (orienté) : ", round(nx.density(G), 2))
print("densité (non orienté) :", round(nx.density(GU), 2))
print("Diamètre (non orienté et connexe) : ", nx.diameter(G0))
nx.diameter(G0, weight = "weight")

# indice de wiener (somme des pcc)
nx.wiener_index(G0)              # distance topologique
nx.wiener_index(G0, 'weight')    # somme des intensités

# rayon
nx.radius(G0)

# barycentre
nx.barycenter(G0)

nx.barycenter(G0, weight='weight')

# centre
list(nx.center(G0))

# périphérie
nx.periphery(G0)

# isthmes et points d'articulation (réseau connexe)
# nombre d'isthmes
print("nb isthmes : ", len(list(nx.bridges(G0))))

# liste des isthmes
print("isthmes : ",list(nx.bridges(G0)))

# nombre de points d'articulation
print("nb points d'articulation : ", len(list(nx.articulation_points(G0))))

# liste des points d'articulation
print("points d'articulation : ", list(nx.articulation_points(G0)))

# assortativité (degré par défaut)
# orienté
print("assortativité in-in : ", 
      round(nx.degree_assortativity_coefficient(G, x="in", y='in'),3))

print("assortativité in-out : ", 
      round(nx.degree_assortativity_coefficient(G, x="in", y='out'),3))

print("assortativité out-in : ", 
      round(nx.degree_assortativity_coefficient(G, x="out", y='in'),3))

print("assortativité out-out : ", 
      round(nx.degree_assortativity_coefficient(G, x="out", y='out'),3))

# non orienté
print("assortativité globale (non orienté) : ", 
      round(nx.degree_assortativity_coefficient(GU),3))

# selon un critère autre que le degré
print(round(nx.numeric_assortativity_coefficient(G, "MARS"),3))

#################
# Mesures locales
#################

# degré et distribution des degrés

#degré topologique total
nx.degree(G)

# degré normalisé
nx.degree_centrality(G)

# degré sortant et entrant
G.out_degree()
G.in_degree()

# prise en compte de la valuation des liens
G.degree(weight = 'weight')
G.out_degree(weight = 'weight')

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

#m degré moyen des voisins
nx.average_neighbor_degree(G)

# distribution des degrés
nx.degree_histogram(G)

#distribution des degrés
degree_freq = nx.degree_histogram(G)
degrees = range(len(degree_freq))
plt.loglog(degrees, degree_freq,'go') 
plt.xlabel('Degré')
plt.ylabel('Fréquence')

# autre option qui permet de différencier in et out
degree_sequence = sorted((d for n, d in G.in_degree()), reverse=True)
plt.bar(*np.unique(degree_sequence, return_counts=True))
plt.title("Degrés entrants")
plt.xlabel("Degré")
plt.ylabel("Fréquence")

degree_sequence = sorted((d for n, d in G.out_degree()), reverse=True)
plt.bar(*np.unique(degree_sequence, return_counts=True))
plt.title("Degrés sortants")
plt.xlabel("Degré")
plt.ylabel("Fréquence")

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.average_degree_connectivity.html
#nx.average_degree_connectivity(G0)
#nx.average_degree_connectivity(G0, weight="weight")

# mesures locales - éloignement (réseau connexe)
nx.eccentricity(G0)

# longueur liens entre chaque sommet et tous les autres
nx.all_pairs_node_connectivity(G0)

# proximité (orienté ou non orneité)
nx.closeness_centrality(G)

# intermédiarité sommets et liens (orienté et non orienté)
nx.betweenness_centrality(G)
nx.edge_betweenness_centrality(G)

#degré moyen des sommets voisins (simple et pondéré)
nx.average_neighbor_degree(G0)
nx.average_neighbor_degree(G, weight="weight")

#transitivity globale
#round(nx.transitivity(G0), 2)

#locale (orienté et non orienté)
nx.clustering(G)

# ontrainte de Burt

#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.structuralholes.constraint.html#networkx.algorithms.structuralholes.constraint
nx.constraint(G0) # orienté ou non orienté

# non orienté seulement
nx.rich_club_coefficient(GU)

# transition vers sous-groupes et partitions

nx.triangles(GU) # non orienté only
nx.triadic_census(G) # orienté only

#recherche des cliques (non orienté)
cl = list(nx.find_cliques(GU))
cl#print("cliques %s" % cl)

# nombre de cliques
sum(1 for c in nx.find_cliques(GU))

# ordre de la plus grande 
max(len(c) for c in nx.find_cliques(GU))

# composition de la plus grande
max(nx.find_cliques(GU), key=len)



#cycles
#sorted(nx.simple_cycles(G0u))

#k-components - réseau non orienté
list(nx.k_components(GU))
list(nx.k_core(GU, k = 8))
list(nx.k_core(GU, k = 7))

# simplifier le graphe ?
# https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.summarization.snap_aggregation.html#networkx.algorithms.summarization.snap_aggregation
#•GA = nx.snap_aggregation(G0u, node_attributes="MARS")

#détection de communautés - https://networkx.org/documentation/stable/reference/algorithms/community.html
louv = nx.community.louvain_communities(G0, seed=123)
louv[1]

#https://iut-info.univ-reims.fr/users/blanchard/ISN20181218/utilisation-de-la-bibliotheque-networkx.html

#☺mesure de la modularité
nx.community.modularity(G0, louv)
#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.partition_quality.html#networkx.algorithms.community.quality.partition_quality
#nx.partition_quality(G0, louv)# error

louv = nx.community.louvain_communities(G0u, seed=123)
len(louv)

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
