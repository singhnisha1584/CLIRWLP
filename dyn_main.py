import networkx as nx

from sklearn.metrics import *
import argparse
import link_prediction
from dynnode2vec import DynNode2Vec


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

#getting array for timestamps
#one snapshot will be the edges between two consecutive timestamps
def get_timestamps(m,lines):
    maxi=0
    mini=10000000000
    for line in lines:
        line = [int(i) for i in line.split()]

        # print(type(line[-1]))
        if line[-1]>maxi:
            maxi=line[-1]
        if int(line[-1])<mini:
            mini=line[-1]
    min1=mini
    print(maxi,mini)
    width = int((maxi-mini)/m)
    arr=[]
    for i in range(0,m+1):
        arr=arr+[min1+width*i]
    return arr

#snapshots
def get_snapshots(m, lines):
    snapshots = []
    lines = lines[1:]
    lines.sort(key=lambda x: x[-1])
    timestamps = get_timestamps(m, lines)
    print(timestamps)
    V = 0
    for i in range(m):
        temp = []
        for line in lines:
            line = [int(i) for i in line.split()]

            if line[0] > V:
                V = line[0]
            if line[1] > V:
                V = line[1]
            if line[-1] >= timestamps[i] and line[-1] < timestamps[i + 1]:
                temp.append((line[0], line[1]))
        snapshots.append(temp)
    return snapshots, V


def create_graph(edgelist):
    if args.weighted:
        G = nx.DiGraph()  # Create a directed graph with edge weights
        for edge in edgelist:
            u, v, weight = edge  # Assume weighted edges have the format (node1, node2, weight)
            G.add_edge(u, v, weight=weight)
    else:
        G = nx.DiGraph()  # Or nx.DiGraph() if you need a directed graph
        for edge in edgelist:
          G.add_edge(*edge, weight=1) 

    if not args.directed:
        G = G.to_undirected()
    all_nodes = G.nodes()
    print("no. of nodes -", len(all_nodes))
    start_node = min(all_nodes)
    last_node = max(all_nodes)
    print("Node Range:- ", start_node, " - ", last_node)
    print("no. of edges -", len(G.edges()), "\n")

    return G

def get_graphs(snapshots):
    t_graph = []
    for snapshot in snapshots:
        edgelist = list(set(tuple(sorted(sub)) for sub in snapshot))
        t_graph.append(create_graph(edgelist))
    return t_graph

def main(args, t_graph):
    
    embeddings = []
    dynnode2vec = DynNode2Vec(p=args.p, q=args.q, walk_length=args.walk_length, n_walks_per_node=args.num_walks, 
    embedding_size=args.dimensions, window=args.window_size, seed=0, parallel_processes=args.workers, plain_node2vec=False)
    embeddings = dynnode2vec.compute_embeddings(t_graph)
    print(embeddings[-1].vectors)
    link_prediction.evaluate_link_prediction(embedding_train=embeddings[-1], nx_G=t_graph[-1],res_type=args.res_type,row_number=args.row)


args = parse_args()
data = open(args.input)
lines = data.readlines()
snapshots, Vertices = get_snapshots(5, lines)
t_graph = get_graphs(snapshots)
main(args, t_graph)

