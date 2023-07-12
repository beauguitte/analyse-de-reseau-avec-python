# installer le module
# pip install networkx

# charger les modules utiles
import pandas as pd                # manipulation de tableaux
import networkx as nx              # analyse de réseau
import numpy as np                 # statistiques
import matplotlib.pyplot as plt    # visualisation

# aide sur une fonction
# ?nom_module.nom_fonction
?nx.find_cliques
?pd.DataFrame.set_index

# import des fichiers

sommets = pd.read_csv("C:Documents/AR_Networkx/data/som_d13.csv", sep = ";")
liens = pd.read_csv("C:Documents/AR_Networkx/data/liens_d13.csv", sep = ";")

sommets = pd.read_csv("data/som_d13.csv", sep = ";")
liens = pd.read_csv("data/liens_d13.csv", sep = ";")

# type des variables
print(sommets.dtypes)
print(liens.dtypes)

# retypage des variables
sommets[['CODGEO']] = sommets[['CODGEO']].astype('string')
sommets[['MARS']] = sommets[['MARS']].astype('bool')

liens[['Origine']] = liens[['Origine']].astype('string')
liens[['Arrivee']] = liens[['Arrivee']].astype('string')

# objet réseau
G = nx.from_pandas_edgelist(liens,                     # data.frame des liens
                            source = "Origine",        # nom de la colonne origine
                            target = "Arrivee",        # nom de la colonne destination
                            edge_attr="weight",        # attribut poids pour un réseau valué
                            create_using=nx.DiGraph()) # création d'un réseau orienté

# attributs des sommets
nodes_attr = sommets.set_index('CODGEO').to_dict(orient = 'index')
nx.set_node_attributes(G, nodes_attr)

#contrôle
G.nodes("MARS")

# propriétés du réseau
print("Ordre", nx.number_of_nodes(G), " sommets")
print("Taille", nx.number_of_edges(G), " liens")
print("Réseau orienté ?", nx.is_directed(G))

# attribut des liens
print("attribut des liens : ", list(list(G.edges(data=True))[0][-1].keys()))

# attributs des sommets
print("attribut des sommets : ", list(list(G.nodes(data=True))[0][-1].keys()))

# visualisation basique
nx.draw_networkx(G, with_labels=False)

# liste ordonnée des composantes connexes
CC = sorted(nx.weakly_connected_components(G),
            key=len,                           # clé de tri - len = longueur
            reverse=True)                      # ordre décroissant
print("Nombre de composantes", len(CC))

# nombre de sommets par composantes
print("Nombre de sommets par composantes",
      [len(c) for c in sorted(nx.weakly_connected_components(G),
       key=len,
       reverse=True)])

# sélection de la composante connexe principale
GD = G.subgraph(CC[0])

# création d'une version non orientée
GU = nx.to_undirected(GD)

# gestion des intensités après transformation
print("lien ij :", GD["13001"]["13201"]['weight'])
print("lien ji :", GD["13201"]["13001"]['weight'])
print("lien ij non orienté : ", GU["13001"]["13201"]['weight'])

# création d'une version non orientée où wij = wij + wji
# créer une copie sans aucun lien
GU = nx.create_empty_copy(GD, with_data=True)
GU = nx.to_undirected(GU)

# éviter message "Frozen graph can't be modified"
GU = nx.Graph(GU)

# récupérer liens avec intensité nulle
GU.add_edges_from(GD.edges(), weight=0)

# pour chaque lien ij + ji
for u, v, d in GD.edges(data=True):
    GU[u][v]['weight'] += d['weight']

# contrôle
#a attributs liens et sommets
list(list(GU.edges(data=True))[0][-1].keys())
list(list(GU.nodes(data=True))[0][-1].keys())

# propriétés du réseau
nx.is_directed(GU)
nx.is_connected(GU)

print("lien ij :", GD["13001"]["13201"]['weight'])
print("lien ji :", GD["13201"]["13001"]['weight'])
print("lien ij non orienté : ", GU["13001"]["13201"]['weight'])

# filtrage des sommets (1)
# sélection des sommets satisfaisant la condition
Mars = [n for n,v in G.nodes(data=True) if v['MARS'] == True]  

# création d'un sous-graphe
Gmars = G.subgraph(Mars)

# visualisation
nx.draw_networkx(Gmars,
                 pos = nx.kamada_kawai_layout(Gmars),
                 with_labels=True)

# filtrage des sommets (2)
# sélection des sommets hors Marseille
nonmars = [n for n,v in G.nodes(data=True) if v['MARS'] == False] 

# copier le réseau de départ
Gmars2 = G

# supprimer les communes hors Marseille
Gmars2.remove_nodes_from(nonmars)

# visualisation
nx.draw_networkx(Gmars2,
                 pos = nx.kamada_kawai_layout(Gmars2),
                 with_labels=True)

# filtrer les liens
# paramètres statistiques
liens.describe()

# fixer un seuil (ici la médiane)
seuil = 212

# identifier les liens sous ce seuil, récupérer les identifiants
long_edges = list(filter(lambda e: e[2] < seuil, (e for e in G.edges.data('weight'))))
le_ids = list(e[:2] for e in long_edges)

# créer une copie du réseau de départ
Gsup = G

# supprimer les liens identifiés
Gsup.remove_edges_from(le_ids)

# ordre, taille et visualisation
print("Nb de sommets : ", nx.number_of_nodes(Gsup))
print("Nb de liens : ", nx.number_of_edges(Gsup))

nx.draw_networkx(Gsup, 
                 pos = nx.kamada_kawai_layout(Gsup),
                 with_labels=True)

# générer un réseau aléatoire avec 2 isolés
rg = nx.gnp_random_graph(20, 0.05, seed = 1)
print("Nb de sommets (isolés compris) : ", nx.number_of_nodes(rg))

# liste des sommets avec un degré nul
isoles = [node for node,degree in dict(rg.degree()).items() if degree < 1]

# suppression des sommets concernés
rg.remove_nodes_from(isoles)
print("Nb de sommets  (isolés exclus) : ", nx.number_of_nodes(rg))

#agrégation sommets
#agréger les arrondissements marseillais

GA = nx.contracted_nodes(G, '13215', '13216', self_loops=True, copy=True)
GA.in_edges('13215')
GA.out_edges('13215')

# MESURES
# mesures portant sur le réseau dans son ensemble
# densité
print("densité (orienté) : ", round(nx.density(GD), 2))
print("densité (non orienté) :", round(nx.density(GU), 2))

# calcul du diamètre
print("diamètre de la CC principale (orientation des liens non prise en compte) : ", nx.diameter(GD))

# rayon du graphe
print("rayon de la CC principale (non orientée) : ", nx.radius(GU))

# barycentre
print("barycentre : ", nx.barycenter(GU))
print("barycentre (valué) : ", nx.barycenter(GU, weight='weight'))

# sommets centraux (excentricité minimale)
print("Sommets centraux : ", nx.center(GU))

# sommets périphériques (excentricité maximale)
print("Sommets périphériques : ", nx.periphery(GU))

# nombre et liste des isthmes
print("nb isthmes : ", len(list(nx.bridges(GU))))
print("isthmes : ",list(nx.bridges(GU)))

# nombre et liste des points d'articulation
print("nb points d'articulation : ", len(list(nx.articulation_points(GU))))
print("points d'articulation : ", list(nx.articulation_points(GU)))

# indice de Wiener
print("Indice de Wiener (orienté) : ", nx.wiener_index(GD))     # distance topologique
print("Indice de Wiener (non orienté) : ", nx.wiener_index(GU)) # distance topologique
print("Indice de Wiener (non orienté, valué) : ", nx.wiener_index(GU, 'weight'))  # somme des intensités

# assortativité (degré par défaut)
# orienté
print("assortativité in-in : ", 
      round(nx.degree_assortativity_coefficient(GD, x="in", y='in'),3))

print("assortativité in-out : ", 
      round(nx.degree_assortativity_coefficient(GD, x="in", y='out'),3))

print("assortativité out-in : ", 
      round(nx.degree_assortativity_coefficient(GD, x="out", y='in'),3))

print("assortativité out-out : ", 
      round(nx.degree_assortativity_coefficient(GD, x="out", y='out'),3))

# non orienté
print("assortativité globale (non orienté) : ", 
      round(nx.degree_assortativity_coefficient(GU),3))

# selon un critère autre que le degré
print(round(nx.numeric_assortativity_coefficient(GD, "MARS"),3))

# rich-club coefficient
print("rich-club coefficient : ", nx.rich_club_coefficient(GU, normalized=False))
print("rich-club coefficient : ", nx.rich_club_coefficient(GU, normalized=True, seed = 42))

# mesures portant sur les sommets ou les liens
# degré
GU.degree()

#degré entrant, sortant
GD.in_degree()
GD.out_degree()

#degré pondéré entrant
GD.in_degree(weight = "weight")

# degrés entrant et sortant normalisés
nx.in_degree_centrality(GD)
nx.out_degree_centrality(GD)

# transformer une mesure en attribut 
deg = nx.degree(GU)
nx.set_node_attributes(GU, 'degree', deg)
print("attribut des sommets : ", list(list(GU.nodes(data=True))[0][-1].keys()))

# visualiser la distribution des degrés

# distribution sous forme d'histogramme
degree_in = sorted((d for n, d in GD.in_degree()), reverse=True)
plt.bar(*np.unique(degree_in, return_counts=True))
plt.title("Degrés entrants")
plt.xlabel("Degré")
plt.ylabel("Fréquence")

# fréquence
degree_sequence = sorted((d for n, d in GU.degree()), reverse=True)

# distribution des degrés
plt.plot(degree_sequence, "r-", marker="o", linewidth=1, markersize=2)
plt.title("Distribution des degrés")
plt.xlabel('Degré')
plt.ylabel('Fréquence')

# distribution degré - échelle log - log
plt.loglog(degree_sequence, "go-", linewidth=1, markersize=2)
plt.title("Distribution des degrés (log - log)")
plt.xlabel('Degré')
plt.ylabel('Fréquence')

#degré moyen des sommets voisins (simple et pondéré)
nx.average_neighbor_degree(GD)
nx.average_neighbor_degree(GD, source="in", target="in")
nx.average_neighbor_degree(GD, weight="weight")

# intermédiarité
nx.betweenness_centrality(GD)
`` 
# intermédiarité des liens
nx.edge_betweenness_centrality(GD)

# proximité
nx.closeness_centrality(GD)

# centralité de vecteur propre
nx.eigenvector_centrality(GD)

# centralité de Katz
nx.katz_centrality(GD)

# triangles
nx.triangles(GU)

# triad census (réseau orienté)
nx.triadic_census(GD)

# transitivité locale
nx.clustering(GD)

# avec prise en compte de l'intensité des liens
nx.clustering(GD, weight="weight")

# transitivité globale et moyenne
print("Transitivité globale (orienté) : ", round(nx.transitivity(GD), 2))
print("Transitivité moyenne (orienté) : ", round(nx.average_clustering(GD), 2))
print("Transitivité globale (non orienté) : ", round(nx.transitivity(GU), 2))
print("Transitivité moyenne (non orienté) : ", round(nx.average_clustering(GU), 2))

# PARTITIONS
# cliques d'ordre 1 à n
print("Nombre de `cliques' : ", sum(1 for c in nx.find_cliques(GU)))
max(len(c) for c in nx.find_cliques(GU))
print("Composition de la plus grande clique \n", max(nx.find_cliques(GU), key=len))

# k-cores
list(nx.k_core(GU))
list(nx.k_components(GU))
list(nx.k_core(GU, k = 8))
list(nx.k_core(GU, k = 7))

# blockmodel
# https://networkx.org/documentation/stable/auto_examples/algorithms/plot_blockmodel.html

# détection de communautés
louv = nx.community.louvain_communities(GU, seed=123) # objet liste
print("nb de communautés :", len(louv))
louv[1]

#mesure de la modularité
print("modularité : ", round(nx.community.modularity(GU, louv),2))

# algorithme maximisant la modularité
greed = nx.community.greedy_modularity_communities(GD)
print("nb de communautés :", len(greed))
print("modularité : ", round(nx.community.modularity(GD, greed),2))

# pour visualiser, nécessiter de passer de la liste au dictionnaire
# https://programminghistorian.org/en/lessons/exploring-and-analyzing-network-data-with-python

# création d'un dictionnaire vide
louvain_dict = {} 

# boucle qui remplit le dictionnaire
for i,c in enumerate(louv): 
    for CODGEO in c: 
        louvain_dict[CODGEO] = i
        
# ligne optionnelle pour ajouter cet attribut 
# nx.set_node_attributes(GD, louvain_dict, 'louvain')

# choix de l'algorithme de placement des sommets
pos = nx.spring_layout(GU)

# importation d'une palette de couleurs 
# nb de couleurs égal au nombre de classes + 1 (Python commence à 0)
cmap = plt.get_cmap('Paired', max(louvain_dict.values()) + 1)

# visualisation des sommets
nx.draw_networkx_nodes(GU,   # sommets à visualiser
                       pos,  # placement
                       louvain_dict.keys(),  # identifiants
                       node_size=40,         # taille 
                       cmap=cmap,            # palette de couleur
                       node_color=list(louvain_dict.values())) # affecter une couleur différente pour chaque classe
nx.draw_networkx_edges(GU, 
                       pos, 
                       alpha=0.5)  # transparence

# VISUALISER
# algorithmes de visualisation

#graphe de Petersen
#pour les plus curieuses : https://fr.wikipedia.org/wiki/Graphe_de_Petersen
G = nx.petersen_graph()

# algorithmes de visualisation
nx.draw_networkx(G)
nx.draw_networkx(G, pos = nx.random_layout(G))
nx.draw_networkx(G, pos = nx.shell_layout(G))
nx.draw_networkx(G, pos = nx.spectral_layout(G))
nx.draw_networkx(G, pos = nx.spring_layout(G))
nx.draw_networkx(G, pos = nx.spiral_layout(G))
nx.draw_networkx(G, pos = nx.spiral_layout(G))
# algorithmes réservés à certains types de réseaux
# multipartite_layout bipartite_layout planar_layout

# Modifier l'apparence

# création du dictionnaire pour les degrés
d = dict(GU.degree)

# création d'une liste des intensités
weights = [GU[u][v]['weight'] for u,v in G0u.edges]

nx.draw_spring(GU, 
               node_color = 'orange',                    # couleur des sommets
               alpha = 0.8,                              # transparence
               nodelist=d.keys(),                        
               node_size = [v * 20 for v in d.values()], # taille des sommets
               edge_cmap=plt.cm.Blues,                   # palette de couleurs
               edge_color=weights,                       # couleur des liens
               width=4)                                  # épaisseur des liens

nx.draw_spring(GU, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_color = "blue",
               width = weights)

# diviser les intensités par 50
weigh2 = [i/50 for i in weights]

nx.draw_spring(GU, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_cmap=plt.cm.Blues,
               edge_color = weights,
               width=weigh2)
