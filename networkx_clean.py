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
nx.draw(G, with_labels=False)

# liste ordonnée des composantes connexes
CC = sorted(nx.weakly_connected_components(G),
            key=len,                            # clé de tri - len = longueur
            reverse=True)                   s    # ordre décroissant
print("Nombre de composantes", len(CC))

# nombre de sommets par composantes
print("Nombre de sommets par composantes",
      [len(c) for c in sorted(nx.weakly_connected_components(G),
       key=len,
       reverse=True)])

# sélection de la composante connexe principale
G0 = G.subgraph(CC[0])

# création d'une version non orientée
G0u = nx.to_undirected(G0)


# gestion des intensités après transformation
print("lien ij :", G0["13001"]["13201"]['weight'])
print("lien ji :", G0["13201"]["13001"]['weight'])
print("lien ij non orienté : ", G0u["13001"]["13201"]['weight'])

# filtrage des sommets (1)
# sélection des sommets satisfaisant la condition
Mars = [n for n,v in G.nodes(data=True) if v['MARS'] == True]  

# création d'un sous-graphe
Gmars = G.subgraph(Mars)

# visualisation
nx.draw_kamada_kawai(Gmars,
                     with_labels=True)

# filtrage des sommets (2)
# sélection des sommets hors Marseille
nonmars = [n for n,v in G.nodes(data=True) if v['MARS'] == False] 

# copier le réseau de départ
Gmars2 = G

# supprimer les communes hors Marseille
Gmars2.remove_nodes_from(nonmars)

# visualisation
nx.draw_kamada_kawai(Gmars2,
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

nx.draw_kamada_kawai(Gsup, with_labels=True)

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

#agréger selon la modalité Marseille
node_attributes = ('MARS',)
GA = nx.snap_aggregation(G, node_attributes = node_attributes)  
print("Nb de sommets : ", nx.number_of_nodes(GA))
print("Nb de liens : ", nx.number_of_edges(GA))
nx.draw_kamada_kawai(GA, with_labels=False)



# mesures portant sur le réseau dans son ensemble
# densité
print("densité (orienté) : ", round(nx.density(G0), 2))
print("densité (non orienté) :", round(nx.density(G0u), 2))

# calcul du diamètre
print("diamètre de la CC principale (orientation des liens non prise en compte) : ", nx.diameter(G0u))

# diamètre pondéré
print("diamètre pondéré : ", nx.diameter(G0u, weight = 'weight'))

# rayon
# rayon du graphe
print("rayon de la CC principale (non orientée) : ", nx.radius(G0u))

# transitivité
print("Transitivité globale (orienté) : ", round(nx.transitivity(G0), 2))
print("Transitivité moyenne (orienté) : ", round(nx.average_clustering(G0), 2))
print("Transitivité globale (non orienté) : ", round(nx.transitivity(G0u), 2))
print("Transitivité moyenne (non orienté) : ", round(nx.average_clustering(G0u), 2))

# visualiser la distribution des degrés
# degré total - échelle log - log
degree_freq = nx.degree_histogram(G0)
degrees = range(len(degree_freq))
plt.loglog(degrees, degree_freq,'go-') 
plt.xlabel('Degré')
plt.ylabel('Fréquence')

# autre option qui permet de différencier in et out
degree_in = sorted((d for n, d in G0.in_degree()), reverse=True)
plt.bar(*np.unique(degree_in, return_counts=True))
plt.title("Degrés entrants")
plt.xlabel("Degré")
plt.ylabel("Fréquence")

degree_out = sorted((d for n, d in G0.out_degree()), reverse=True)
plt.bar(*np.unique(degree_out, return_counts=True))
plt.title("Degrés sortants")
plt.xlabel("Degré")
plt.ylabel("Fréquence")

# mesures portant sur les sommets ou les liens
# degré
G0u.degree()

#degré entrant, sortant
G0.in_degree()
G0.out_degree()

#degré pondéré entrant
G0.in_degree(weight = "weight")

#degré moyen des sommets voisins (simple et pondéré)
nx.average_neighbor_degree(G0)
nx.average_neighbor_degree(G, weight="weight")

# degrés entrant et sortant normalisés
nx.in_degree_centrality(G0)
nx.out_degree_centrality(G0)

# ajouter un attribut - marche pas
#deg = nx.degree(G0u)
#nx.set_node_attributes(G0u, 'degree', deg)
#print("attribut des sommets : ", list(list(G0u.nodes(data=True))[0][-1].keys()))

# intermédiarité
nx.betweenness_centrality(G0)

# intermédiarité des liens
nx.edge_betweenness_centrality(G0)

# proximité
nx.closeness_centrality(G0)

# cliques
print("Nombre de `cliques' : ", sum(1 for c in nx.find_cliques(G0u)))
print("Composition de la plus grande clique \n", max(nx.find_cliques(G0u), key=len))

# algorithme de visualisation

#graphe de Petersen
#pour les plus curieuses : https://fr.wikipedia.org/wiki/Graphe_de_Petersen
G = nx.petersen_graph()

#5 représentations superposées
nx.draw(G)
nx.draw_random(G)
nx.draw_shell(G)
nx.draw_spectral(G)
nx.draw_spring(G)

# Modifier l'apparence

# création du dictionnaire pour les degrés
d = dict(G0u.degree)

# création d'une liste des intensités
weights = [G0u[u][v]['weight'] for u,v in G0u.edges]

nx.draw_spring(G0u, 
               node_color = 'orange',                    # couleur des sommets
               alpha = 0.8,                              # transparence
               nodelist=d.keys(),                        
               node_size = [v * 20 for v in d.values()], # taille des sommets
               edge_cmap=plt.cm.Blues,                   # palette de couleurs
               edge_color=weights,                       # couleur des liens
               width=4)                                  # épaisseur des liens

nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_color = "blue",
               width = weights)

# diviser les intensités par 50
weigh2 = [i/50 for i in weights]

nx.draw_spring(G0u, 
               node_color = 'orange', 
               alpha = 0.8,
               nodelist=d.keys(), 
               node_size = [v * 20 for v in d.values()], 
               edge_cmap=plt.cm.Blues,
               edge_color = weights,
               width=weigh2)
