import argparse
from multiprocessing import process
import numpy as np
import networkx as nx
import pandas as pd
import node2vec
import link_prediction
from sklearn.preprocessing import MinMaxScaler, normalize
import math
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
import scipy.sparse as sp
import torch
from torch import Tensor
import os
import multiprocessing
def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='/content/drive/MyDrive/Thesis2/node2vec/graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='/content/drive/MyDrive/Thesis2/node2vec/emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
	                    help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
	                    help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()
def read_graph(snap):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.from_edgelist(snap, create_using=nx.DiGraph())
	else:
		G = nx.from_edgelist(snap, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	#model.wv.save_word2vec_format(args.output)
	print("Embeddings saved to file successfully")
	
	#return
	return model

def data(m, t):
    data = open("/content/drive/MyDrive/Thesis2/Datasets/" + t + ".txt")
    edgelist = map(lambda q: list(map(int, q.split())),
                   data.read().split("\n")[:-1])
    data.close()
    maxi = 0
    mini = 10000000000
    edgelist = list(edgelist)
    for x in edgelist:
        if x[-1] > maxi:
            maxi = x[-1]
        if x[-1] < mini:
            mini = x[-1]
    min1 = mini
    w = int((maxi - mini) / m)
    edgelist.sort(key=lambda x: x[-1])
    arr = []
    i = 0
    for i in range(0, m + 1):
        arr = arr + [min1 + w * i]
    arri = []
    # print(arr)
    nodes = set()
    for i in range(0, m):
        temp = []
        for j in edgelist:
            if j[-1] >= arr[i] and j[-1] <= arr[i + 1]:
                temp += [[j[0], j[1]]]
        arri += [temp]
    # print(arri)
    # for x in arri:
    #     print(len(x))
    print("after read")
    return arri

def gen_graph(l):
    print("inside gen graph")
    t_graph = []
    node_set = set()
    max = -99999
    min = 99999
    for i in l:
        for edge in i:
            node_set.add(edge[0])
            node_set.add(edge[1])
            u = edge[0]
            v = edge[1]
            if u<v:
                if min > u:
                   min = u
                if max < v:
                    max = v
            else:
                if min > v:
                   min = v
                if max < u:
                    max = u
    print(str(min) + "-" + str(max))
    #sys.exit()
    edgelist_new = []
    count = -1
    for i in l:
        graph = nx.Graph()
        #graph.add_nodes_from(node_set)
        graph.add_edges_from(i)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        t_graph.append(graph)
        edgelist_new.append(list(graph.edges))
    return [t_graph,edgelist_new]
  
def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	l1 = data(5, "CollegeMsg")
	l = []
  #######For appending '0's for non exixtant snapshot##########
	full_edgelist = list()
	for i in l1:
		edgelist = list(set(tuple(sorted(sub)) for sub in i))
		full_edgelist.extend(list(edgelist))
		l.append(edgelist)
	G_master = read_graph(full_edgelist)
	print(G_master)
	l = gen_graph(l)[1]
	embed_dict = dict()
	G = node2vec.Graph(G_master, args.directed, args.p, args.q)
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	process1 = multiprocessing.Process(target=G.get_sim_matrix, args =(1,return_dict,))
	process2 = multiprocessing.Process(target=G.get_sim_matrix, args =(2,return_dict,))
	process3 = multiprocessing.Process(target=G.get_sim_matrix, args =(3,return_dict,))
	process4 = multiprocessing.Process(target=G.get_sim_matrix, args =(4,return_dict,))
	process1.start()
	process2.start()
	process3.start()
	process4.start()
	process1.join()
	process2.join()
	process3.join()
	process4.join()
	G.sim_matrix = {**return_dict[0],**return_dict[1], **return_dict[2], **return_dict[3]}
	#G.get_sim_matrix()
	print("Done with Similarity weght Matrix")  #, G.sim_matrix
	for i in range(4):
		nx_G = read_graph(l[i])
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(args.num_walks, args.walk_length)
		mdl = learn_embeddings(walks)
		vector = np.zeros(args.dimensions)
		for n in G_master.nodes():
			try:
				vector = mdl.wv.get_vector(str(n))
			except KeyError:
				vector = np.zeros(args.dimensions)
			if n in embed_dict:
				np.append(embed_dict[n], vector, axis=0)
			else:
				embed_dict[n] = vector
	value = link_prediction.evaluate_link_prediction(embed_dict, nx_G, l[4])

if __name__ == "__main__":
	args = parse_args()
	main(args)

