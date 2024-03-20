from typing import Any, Iterable, List
from collections import namedtuple
from itertools import chain, starmap
from multiprocessing import Pool
from gensim.models import Word2Vec
import node2vec_original

Embedding = namedtuple("Embedding", ["vectors", "mapping"])

class DynNode2Vec:
    def __init__(
        self,
        p=1.0,
        q=1.0,
        walk_length=80,
        n_walks_per_node=10,
        embedding_size=128,
        window=10,
        seed=None,
        parallel_processes=8,
        plain_node2vec=False,
    ):
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node
        self.embedding_size = embedding_size
        self.window = window
        self.seed = seed
        self.parallel_processes = parallel_processes
        self.plain_node2vec = plain_node2vec
        self.gensim_workers = max(self.parallel_processes - 1, 8)

    def learn_embeddings(self, walks):
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, vector_size=self.embedding_size, window=self.window, min_count=0, 
                         sg=1, workers=self.gensim_workers, epochs=1)
        return model

    def _initialize_embeddings(self, graphs):
        first_graph = graphs[0]
        G = node2vec_original.Graph(first_graph, False, self.p, self.q)
        G.preprocess_transition_probs()
        first_walks = G.simulate_walks(nodes=list(first_graph.nodes()), num_walks=self.n_walks_per_node, walk_length=self.walk_length)
        model = self.learn_embeddings(first_walks)
        mapping = {str(node_id): i for i, node_id in enumerate(model.wv.index_to_key)}
        embedding = Embedding(model.wv.vectors.copy(), mapping)
        return model, [embedding]

    def get_delta_nodes(self, current_graph, previous_graph):
        delta_edges = current_graph.edges ^ previous_graph.edges
        nodes_with_modified_edges = set(chain(*delta_edges))
        delta_nodes = set(current_graph.nodes) & nodes_with_modified_edges
        return delta_nodes

    def generate_updated_walks(self, current_graph, previous_graph):
        if self.plain_node2vec:
            delta_nodes = list(current_graph.nodes())
        else:
            delta_nodes = list(self.get_delta_nodes(current_graph, previous_graph))
        nx_G = node2vec_original.Graph(current_graph, False, self.p, self.q)
        nx_G.preprocess_transition_probs()
        updated_walks = nx_G.simulate_walks(nodes=delta_nodes, num_walks=self.n_walks_per_node, walk_length=self.walk_length)
        return updated_walks

    def _simulate_walks(self, graphs):
        if self.parallel_processes > 1:
            with Pool(self.parallel_processes) as p:
                return p.starmap(self.generate_updated_walks, zip(graphs[1:], graphs))

        return starmap(self.generate_updated_walks, zip(graphs[1:], graphs))
    
    def _update_embeddings(self, embeddings, time_walks, model):
        print("In update_embeddings")
        for walks in time_walks:
            walks = [list(map(str, walk)) for walk in walks]
            if not self.plain_node2vec:
                model.build_vocab(walks, update=True)
                model.train(walks, total_examples=len(walks), epochs=model.epochs)
             # Create a mapping dictionary from node IDs to their vector indices
            mapping = {node_id: i for i, node_id in enumerate(model.wv.index_to_key)}
            embedding = Embedding(model.wv.vectors.copy(), mapping)
            embeddings.append(embedding)
        print(len(embeddings))

    def compute_embeddings(self, graphs):
        model, embeddings = self._initialize_embeddings(graphs)
        time_walks = self._simulate_walks(graphs)
        self._update_embeddings(embeddings, time_walks, model)
        return embeddings
