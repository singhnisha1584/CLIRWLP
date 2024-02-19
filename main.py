import argparse
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



  

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	'''G = nx.Graph()
	#graph.add_nodes_from(node_set)
	G.add_edges_from(edge_list)
	G.remove_edges_from(nx.selfloop_edges(G))'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
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
	model.wv.save_word2vec_format(args.output)
	print("Embeddings saved to file successfully")
	
	#return
	return model

  
def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	values = []
	#generate weights and add to the file of edgelist
	#genearte_weights()
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.get_sim_matrix()
	#print("Done with Similarity weght Matrix", G.sim_matrix)
	for i in range(10):
		G.preprocess_transition_probs()
		walks = G.simulate_walks(args.num_walks, args.walk_length)
		learn_embeddings(walks)
		embedding_train = learn_embeddings(walks)
		value = link_prediction.evaluate_link_prediction(embedding_train, nx_G)
		values.append(value)
	print("\n#########Printing average of 10 itertions##########")
	val = np.array(values)
	print(val)
	print(np.array2string(np.average(val, axis=0), separator='\n'))

if __name__ == "__main__":
	args = parse_args()
	main(args)

