import numpy as np
import random
import networkx as nx

from sklearn.metrics import *



var_dict = {}
TW = {}
nodes_per_label = {}
methods_list = {}

def max_comm_label (node):
    global var_dict
    G = var_dict['graph']
    all_labels = set()
    for node_neighbour in G.neighbors(node):
        all_labels.add(var_dict[node_neighbour])
    prob_actual = 1
    label_actual = var_dict[node]
    for label in all_labels:
        prob_new = 1
        for node_chk in G.neighbors(node):
            if var_dict[node_chk] == label :
                chk = 0
                if G.has_edge(node,node_chk) :
                    chk = G[node][node_chk]['weight']
                if var_dict['influence'][node][node_chk] == 1 :
                    prob_new = prob_new * (1 - chk)
        if prob_new < prob_actual :
            prob_actual = prob_new
            label_actual = label
            var_dict[node] = label
    return label_actual

def detachability (label) :
    global var_dict
    G = var_dict['graph']
    internal = 0
    external = 0
    DZ = 0
    for node in G :
        if var_dict[node] == label :
            for node_neighbour in G.neighbors(node) :
                if var_dict[node_neighbour] == label :
                    internal = internal + G[node][node_neighbour]['weight']
                else :
                    external = external + G[node][node_neighbour]['weight']
    if internal + external != 0 :
        DZ = internal / (internal + external)
    return DZ

def clustering_ss(graph):

    global var_dict
    global TW
    global nodes_per_label

    # tao = var_dict['tao']
    # theta = var_dict['theta']

    tao = 15
    theta = 0.7


    adj = nx.adjacency_matrix(graph).todense()
    G = var_dict['graph'].copy()
    # print("No. of edges - "+str(len(G.edges())))
    i = 1
    A = []
    A = np.zeros((len(adj), len(adj)))
    var_dict['influence'] = A
    for i in range(len(adj)):
        for j in range(len(adj)):
            A[i][j] = -1
    for node in G:
        var_dict[node] = node
        for node_neighbour in G.neighbors(node):
            if G.edges[node,node_neighbour]['weight'] > random.uniform(0,1) : 
                A[node][node_neighbour] = 1
            else : 
                A[node][node_neighbour] = 0
            var_dict[node_neighbour] = node_neighbour
    i = 0
    while i <= tao :
        # print("for i = "+str(i))
        for node in G:
            old_label = var_dict[node]


            new_label = max_comm_label(node)


            var_dict[node] = new_label
        i = i + 1
        total_labels = set()
        for node in G:
            total_labels.add(var_dict[node])
        # print("number of labels left "+str(len(total_labels)))
    all_labels = set()
    for node in G:
        all_labels.add(var_dict[node])
    for label in all_labels:
        DZ1 = detachability(label)
        if DZ1 < theta :
            just_neighbour = set()
            outer = set()
            TW = np.zeros(len(adj))
            for node in G:
                if var_dict[node] == label :
                    for node_neighbour in G.neighbors(node):
                        just_neighbour.add(node_neighbour)
                        if var_dict[node_neighbour] != label :
                            TW[node_neighbour] += 1
                else :
                    outer.add(node)
            NE = just_neighbour.intersection(outer)
            c_max = 0
            for node_inter in NE :
                if TW[node_inter] > c_max :
                    c_max = TW[node_inter]
            NS = set()
            for node_inter in NE:
                if TW[node_inter] == c_max :
                    NS.add(node_inter)
            CS_label = set()
            for node in NS :
                CS_label.add(var_dict[node])
            MID = -99999
            new_label = label
            for label_other in CS_label :
                factor2 = detachability(label_other)
                to_be_changed = set()
                for node in G :
                    if var_dict[node] == label_other :
                        to_be_changed.add(node)
                        var_dict[node] = label
                factor1 = detachability(label)
                for node in to_be_changed :
                    var_dict[node] = label_other
                TID = factor1 - factor2
                if TID > MID :
                    MID = TID
                    new_label = label_other
            for node in G :
                if var_dict[node] == label :
                    var_dict[node] = new_label

    total_labels = set()
    for node in G :
        total_labels.add(var_dict[node])
    # print("number of labels left finally " + str(len(total_labels)))
    for label in total_labels :
        count = 0
        nodes_per_label[label] = count
        for node in G :
            if var_dict[node] == label :
                count += 1
        nodes_per_label[label] = count
    cluster_matrix = np.zeros((len(adj), len(adj)))
    no_of_nodes = G.number_of_nodes()
    for i in range(len(adj)):
        for j in range(len(adj)) :
            if var_dict[i] == var_dict[j] :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes
            else :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes*-1
    var_dict['cluster_matrix_check'] = cluster_matrix
    return cluster_matrix

def normalize (n):
    max = 0
    for i in range(len(n)):
        for j in range(len(n)) :
            if max < n[i][j] : max = n[i][j]
    for i in n:
        if max > 0 :
            for j in range(len(i)):
                i[j] = i[j] / max
    return n

def clustering_main (G) :

    # adj matrix
    print("Inside clustering_main")

    adj = nx.to_numpy_array(G)

    # n = len(G)
    # adj = np.zeros((n,n))
    # for i in range(n):
    #     for j in graph[i]:
    #         adj[i][j] = 1

    # G = nx.Graph(adj)
    # original_adj = nx.convert_matrix.to_numpy_array(G)

    var_dict['graph'] = G
    for (u, v) in G.edges():
        value = random.uniform(0, 1)
        G.edges[u, v]['weight'] = value

    cluster_matrix = clustering_ss(G)

    # link Prediction

    '''var_dict['cluster_matrix'] = cluster_matrix
    similarity_matrix = np.zeros((len(adj), len(adj)))
    overall_similarity_matrix = np.zeros((len(adj), len(adj)))
    print("making similarity matrix")
    for node1 in G :
        for node2 in G :
            similarity_matrix[node1][node2] = 1
            common_neighbour_factor = 1
            for node_neighbour in G.neighbors(node1) :
                if G.has_edge(node2,node_neighbour) :
                    common_neighbour_factor = common_neighbour_factor * (1 - G.edges[node2, node_neighbour]['weight'])
            neighbour_factor = 0
            if G.has_edge(node1, node2): neighbour_factor = G.edges[node1, node2]['weight']
            similarity_matrix[node1][node2] = 1 - common_neighbour_factor + neighbour_factor
    similarity_matrix = normalize(similarity_matrix)
    var_dict['similarity_matrix'] = similarity_matrix
    print("making overall similarity matrix")
    for i in range(len(adj)) :
        for j in range(len(adj)) :
            overall_similarity_matrix[i][j] = similarity_matrix[i][j]*cluster_matrix[i][j]
    var_dict['overall_similarity_matrix'] = overall_similarity_matrix
    link_pred = np.zeros((len(adj), len(adj)))
    print("making link prediction matrix")
    for node1 in G :
        for node2 in G :
            node_neighbour_common = nx.common_neighbors(G,node1,node2)
            for common_node in node_neighbour_common:
                link_pred[node1][node2] += overall_similarity_matrix[node1][common_node] + overall_similarity_matrix[common_node][node2]
    print("Returning from clustering_main")'''
    return cluster_matrix

def clp_gen(G, is_0_based):
    if is_0_based:
      clpid = clustering_main(G)
      return clpid
    
    #for 0-based indexing
    # Create a new graph with 0-based indexing
    G_0_based = nx.Graph()
    for edge in G.edges():
        edge_0_based = tuple(node - 1 for node in edge)
        G_0_based.add_edge(*edge_0_based)
    
    clpid = clustering_main(G_0_based)
    #for returning 1-based indexing
    adj = nx.to_numpy_array(G)
    clpid_1_based = np.zeros((len(adj)+1, len(adj)+1))
    for i in range(len(clpid)):
        for j in range(len(clpid[i])):
            clpid_1_based[i + 1][j + 1] = clpid[i][j]

    return clpid_1_based
