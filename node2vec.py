from gensim import similarities
import numpy as np
import networkx as nx
import random
import link_prediction
import os
import math
import clp

class Graph():
  def __init__(self, nx_G, is_directed, p, q):
    self.G = nx_G
    self.is_directed = is_directed
    self.p = p
    self.q = q
    self.V = len(nx_G.nodes())+3
    #print("Length of nodes = ", self.V)
    #exit()
    self.sim_matrix = [[0 for x in range(self.V)] for y in range(self.V)]

  def node2vec_walk(self, walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''
    G = self.G
    alias_nodes = self.alias_nodes
    alias_edges_nbr = self.alias_edges_nbr    
    alias_edges_sim = self.alias_edges_sim
    #print("#########alias_edges_nbr-->",alias_edges_nbr)
    #print("#########alias_edges_sim-->",alias_edges_sim)
    alpha = 0.5
    walk = [start_node]
    #alias_edges index [0] has the node umberd, now change the code accordingly
    while len(walk) < walk_length:
      cur = walk[-1]
      cur_nbrs = sorted(G.neighbors(cur))
      if len(cur_nbrs) > 0:
        if len(walk) == 1:
          walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
          prev = walk[-2]          
          if(np.random.rand()< alpha):
            if alias_edges_nbr.get((prev, cur)) is None:
              alias_edges_nbr[(prev, cur)] = self.get_alias_edge_nbr(prev, cur)
            next = cur_nbrs[alias_draw(alias_edges_nbr[(prev, cur)][0], alias_edges_nbr[(prev, cur)][1])]
          else:            
            #print("cur->", cur, "prev->", prev)
            cur_prev_sim = [self.sim_matrix[prev][i] + self.sim_matrix[cur][i] for i in range(0, self.V)]   #        
            sim_weights = sorted(cur_prev_sim)[-10:]
            #print("Similarity array of cur and prev", sim_weights)            
            sim_nodes = [(cur_prev_sim.index(wt)+1) for wt in sim_weights ] #if wt>0]
            #print("Similar node Lsit",sim_nodes)
            if alias_edges_sim.get((prev, cur)) is None:
              alias_edges_sim[(prev, cur)] = self.get_alias_edge_sim(prev, cur)
            #print("Prob distribution",alias_edges_sim[(prev, cur)][0])
            #print("Prob distribution",alias_edges_sim[(prev, cur)][1])            
            next = sim_nodes[alias_draw(alias_edges_sim[(prev, cur)][0], alias_edges_sim[(prev, cur)][1])]
            #print("next node", next)
            #exit()          
          walk.append(next)
      else:
        break

    return walk

  def simulate_walks(self, num_walks, walk_length):
      '''
      Repeatedly simulate random walks from each node.
      '''
      G = self.G
      walks = []
      nodes = list(G.nodes())
      print('Walk iteration:')
      for walk_iter in range(num_walks):
        print(str(walk_iter+1), '/', str(num_walks))
        random.shuffle(nodes)
        for node in nodes:
          walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

      return walks

  def get_alias_edge_nbr(self, src, dst):
    '''
    Get the alias edge setup lists for a given edge.
    '''
    G = self.G
    p = self.p
    q = self.q	
    normalized_probs = 0
    unnormalized_probs = []
    #print("Source", src, "Destination", dst)
    #print("Neighbour Nodes", sorted(G.neighbors(dst)))
    for dst_nbr in sorted(G.neighbors(dst)):
      if dst_nbr == src:        
        unnormalized_probs.append((self.sim_matrix[src][dst_nbr] + self.sim_matrix[dst][dst_nbr])/p)
        #unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)  
      elif G.has_edge(dst_nbr, src):
        unnormalized_probs.append((self.sim_matrix[src][dst_nbr] + self.sim_matrix[dst][dst_nbr]))
        #unnormalized_probs.append(G[dst][dst_nbr]['weight'])
      else:
        unnormalized_probs.append((self.sim_matrix[src][dst_nbr] + self.sim_matrix[dst][dst_nbr])/q)
        #unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
    norm_const = sum(unnormalized_probs)
    #print("unnormalized_probs",unnormalized_probs)
    normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs if u_prob>0]
    #print("normalized_probs",normalized_probs)
    j, q = alias_setup(normalized_probs)
    #print("Node indexs", j)
    #print("probaility after alais", q)
    #exit()
    return j, q
    
  def get_alias_edge_sim(self, src, dst):
      '''
      Get the alias edge setup lists for a given edge.
      '''
      G = self.G
      p = self.p
      q = self.q	
      normalized_probs = 0
      unnormalized_probs = []
      ## Taking  similar nodes for both current and prev node ##
      cur_prev_sim = [self.sim_matrix[src][i] + self.sim_matrix[dst][i] for i in range(0, self.V)] #
      sim_weights = sorted(cur_prev_sim)[-10:]      
      sim_nodes = [(cur_prev_sim.index(wt)+1) for wt in sim_weights ] #if wt>0]      
      for sim_weight in sim_weights:  
        #if sim_weight>0:    
        unnormalized_probs.append(sim_weight)
        norm_const = sum(unnormalized_probs)
      normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs if u_prob>0]         
      return alias_setup(normalized_probs)

  def preprocess_transition_probs(self):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    G = self.G
    is_directed = self.is_directed
    normalized_probs = 0
    alias_nodes = {}
	
    for node in G.nodes():
      unnormalized_probs = [self.sim_matrix[node][nbr] for nbr in sorted(G.neighbors(node))]
      #unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))] 
      norm_const = sum(unnormalized_probs)
      normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs if u_prob>0]
      alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges_nbr = {}
    alias_edges_sim = {}
    triads = {}

    if is_directed:
      for edge in G.edges():
        alias_edges_nbr[edge] = self.get_alias_edge(edge[0], edge[1])
    else:
      for edge in G.edges():
        alias_edges_nbr[edge] = self.get_alias_edge_nbr(edge[0], edge[1])
        alias_edges_nbr[(edge[1], edge[0])] = self.get_alias_edge_nbr(edge[1], edge[0])
        alias_edges_sim[edge] = self.get_alias_edge_sim(edge[0], edge[1])
        alias_edges_sim[(edge[1], edge[0])] = self.get_alias_edge_sim(edge[1], edge[0])

    self.alias_nodes = alias_nodes
    self.alias_edges_nbr = alias_edges_nbr
    self.alias_edges_sim = alias_edges_sim
    

    return
  
  def get_sim_matrix(self):
    print("Inside get_sim_matrix")
    G = self.G
    sim_matrix = self.sim_matrix
    #print("Len of V in get_sim_matrix", sorted(G.nodes()))
    clpid=clp.clp_gen(G)
    for i in G.nodes():
      for j in G.nodes():        
        weight = 0
        #try:
        if i==j:
          sim_matrix[i][j] = weight
          continue
        '''#l3
        l3 = 0	
        for intermediate_node1 in G.neighbors(i):	
          for intermediate_node2 in G.neighbors(intermediate_node1):	
            if G.has_edge(j, intermediate_node2):	
              l3 += 1 / math.sqrt(	
                G.degree[intermediate_node1] * G.degree[intermediate_node2]) # * ((math.log(weight1) + math.log(weight2))/2)		
        weight += l3
        # cclp index
        cclp = 0
        triangles_dict = dict()
        cn = len(sorted(nx.common_neighbors(G, i, j)))
        common_n = nx.common_neighbors(G, i, j)
        if cn > 0:
          for k in common_n:
            if G.degree(k) > 1:
              triangles_curr = 0
              if k in triangles_dict.keys():
                triangles_curr = triangles_dict[k]
              else:
                n1_all = G.neighbors(k)
                n2_all = G.neighbors(k)
                for n1 in n1_all:
                  for n2 in n2_all:
                    if n1 != n2 and G.has_edge(n1, n2):
                      triangles_curr += 1
                triangles_curr = triangles_curr / 2
                triangles_dict[k] = triangles_curr
              cclp += triangles_curr / (G.degree(k) * (G.degree(k) - 1) / 2) 
        weight += cclp'''
        #Clustering Coeficcient of L2
        # cc = 0
        # n1 = G.neighbors(i)
        # n2 = G.neighbors(j)	
        # cluster_i = n1
        # cluster_j = n2
        # #cn = len(sorted(nx.common_neighbors(G, i, j)))
        # #l = len(list(n1)) + len(list(n2)) - cn
        # '''for in1 in n1:
        #   #cluster_i.append(G.neighbors(in1))	
        #   cluster_i = set().union(cluster_i, G.neighbors(in1))
        # for in2 in n2:  
        #   #cluster_j.append(G.neighbors(in2))
        #   cluster_j = set().union(cluster_j, G.neighbors(in2))'''
        # for p in cluster_i:
        #   for q in cluster_j:
        #     if G.has_edge(p, q):              
        #       cc += 1
        # weight += cc


        #CLPID
        weight = clpid[i][j]

        #print("i=",i," j=", j)
        sim_matrix[i][j] = weight
        '''except:
          print("i=",i," j=", j)
          exit()'''
    
    print("returning from get_sim_matrix")
    return sim_matrix


def alias_setup(probs):
  '''
  Compute utility lists for non-uniform sampling from discrete distributions.
  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
  for details
  '''

  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=int)

  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
      q[kk] = K*prob
      if q[kk] < 1.0:
          smaller.append(kk)
      else:
          larger.append(kk)

  while len(smaller) > 0 and len(larger) > 0:
      small = smaller.pop()
      large = larger.pop()

      J[small] = large
      q[large] = q[large] + q[small] - 1.0
      if q[large] < 1.0:
          smaller.append(large)
      else:
          larger.append(large)

  return J, q

def alias_draw(J, q):
  '''
  Draw sample from a non-uniform discrete distribution using alias sampling.
  '''
  K = len(J)
  kk = int(np.floor(np.random.rand()*K))
  if(K == 0 ): return 0
  ############Remove this part when code requested############
  '''if np.random.rand() < q[kk]:
    return kk
  else:'''
  return J[kk]