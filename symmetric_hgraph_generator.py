# Hypergraph generator based on the algorithm proposed in https://github.com/HariniA/EDRW/blob/master/synthetic_dataset_generation.pdf
# and # FENNEL: two models for constructing graphs: 
#       n vertices, k clusters, p_interconnectivity, p_intraconnectivity
#       n vertices, d powerlaw distribution: then use Chung-Lu method to create an instance of the corresponding power law graph
# Paper reference: Extended Discriminative Random Walk: A Hypergraph Approach to Multi-View Multi-Relational Transductive Learning, IJCAI 2015
# We do not care about classes since we are not labelling the vertices
# Based on a power-law distribution (scale free network), generate a specified number of hyperedges from a list of vertices

# evaluate clusterings with Normalized Mutual Information http://dmml.asu.edu/users/xufei/Papers/ICDM2010.pdf or Sillouette 

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
import os
import math

# hypergraph parameters
hypergraph_name = "symmetric_large.hgr"
export_folder = '/media/carlos/2130D25104ACF9E0/'
num_edge_ver = 8100000
hyperedge_degree = 1000

# precalculate values


# build hypergraph one hyperedge at a time
with open(export_folder + hypergraph_name,"w+") as f:
        # write header (NUM_HYPEREDGES NUM_VERTICES)
        f.write("{} {}\n".format(num_edge_ver,num_edge_ver))

        for he_id in range(num_edge_ver):
                if he_id % 1000 == 0: print(he_id)
                # to ensure matrix is symmetric, all hyperedges have same connection pattern but skewed 1 position from previous
                conn_pattern = np.array(range(1, num_edge_ver + 1, math.ceil(num_edge_ver / hyperedge_degree))) + he_id
                conn_pattern[conn_pattern > num_edge_ver] -= num_edge_ver
                conn_pattern = sorted(conn_pattern)
                vertices = [str(v) for v in conn_pattern]
                f.write(" ".join(vertices))
                f.write("\n")


                        