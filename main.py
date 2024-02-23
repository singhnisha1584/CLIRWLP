import argparse
import numpy as np
import networkx as nx
import node2vec
import node2vec_original
import link_prediction
from gensim.models import Word2Vec
from torch_geometric.utils import to_networkx



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
	parser.add_argument('--res-type', type=int, default=1,
	                    help='Set the result type. 1- Original Node2vec, 2- CC with Node2vec, 3- CLPID with Node2vec. Default is 1.')
	parser.add_argument('--row', type=int, default=30,
	                    help='Set the row number from where you want you result in xlxs file. Default is 30.')

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
	print("no. of nodes -", len(G.nodes()))
	print("no. of edges -", len(G.edges()))
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
	#values = []
	#generate weights and add to the file of edgelist
	#genearte_weights()
	nx_G = read_graph()
	if args.res_type == 1:
		print("Runnning node2vec original")
		G = node2vec_original.Graph(nx_G, args.directed, args.p, args.q)
	else:
		if args.res_type == 2:
			print("Runnning cc (clustering coefficient) with node2vec")
		elif args.res_type == 3:
			print("Runnning clpid with node2vec")
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q, res_type=args.res_type)
		G.get_sim_matrix()
		print("Done with Similarity weght Matrix", G.sim_matrix[0], '\n',  G.sim_matrix[1], '\n', G.sim_matrix[2], '\n', G.sim_matrix[3], '\n')
	# for i in range(10):
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	embedding_train = learn_embeddings(walks)
	link_prediction.evaluate_link_prediction(embedding_train, nx_G, args.res_type, args.row)
	# values.append(value)
	# print("\n#########Printing average of 10 itertions##########")
	# val = np.array(values)
	# print(np.array2string(np.average(val, axis=0), separator='\n'))

if __name__ == "__main__":
	args = parse_args()
	main(args)

