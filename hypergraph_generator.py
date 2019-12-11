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
import math
from random import shuffle
import shutil
import os

# hypergraph parameters
hypergraph_name = "small_uniform_dense_c192.hgr"
export_folder = './'
num_vertices = 200000
num_hyperedges = 50000
num_clusters = 192
cluster_density = [1.0/num_clusters for _ in range(num_clusters)]       # probability that a vertex belongs to each cluster
#cluster_density = [0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.05,0.20,0.25,0.05,0.05]
p_intraconnectivity = 0.99
hyperedge_gamma = 1.5                            # determines the skewness of the distribution (higher values more skewed to the left). Must be >= 0 (gamma == 0 is the uniform distribution)
min_hyperedge_degree = 50
max_hyperedge_degree = 200
vertex_degree_power_law = False          # whether drawing vertex ids is done using power law distribution (much slower)
vertex_gamma = 1.0                     # careful! high values of this may prevent the graph from finishing (since vertices cannot be added twice to the same hyperedge, the roll will be rolled and may always get the same most probable answers)
ensure_no_missing_vertices = True       # ensure that all vertex ids are present in at least one hedge
show_distribution = False
store_clustering = True



# sanity check to test that desired cluster sizes are large enough for desired hyperedge degrees
for i,r in enumerate(cluster_density):
        cluster_size = num_vertices * r
        if cluster_size < max_hyperedge_degree:
                print("Cluster size is not large enough. Cluster {} size is {}, but max hyperedge degree is {}.".format(i,cluster_size,max_hyperedge_degree))
                exit()


#power law distributions
def truncated_power_law(a, m):
    x = np.arange(1, m+1, dtype='float')
    pmf = 1/x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, m+1), pmf))

# will sample from 1 to max_hyperedge_degree with probability power law degree
distribution = truncated_power_law(a=hyperedge_gamma, m=(max_hyperedge_degree-min_hyperedge_degree+1))
#sample = distribution.rvs(size=num_hyperedges)
#plt.hist(sample, bins=np.arange(max_hyperedge_degree)+0.5)
#plt.show()

# assign vertices to clusters (randomly)
print("Generating {} clusters".format(num_clusters))
vertices = set(range(num_vertices))
nodes = np.random.binomial(num_vertices,p=cluster_density,size=num_clusters)
clusters = [[] for _ in range(num_clusters)]
if store_clustering:
        partitioning = np.array([0 for _ in range(num_vertices)])
for k in range(num_clusters):
        # how many vertices will belong to this cluster
        n = min(nodes[k],len(vertices))
        selected = np.random.choice(list(vertices),size=n,replace=False)
        vertices = vertices.difference(set(selected)) #vertices = [v for v in vertices if v not in selected]
        clusters[k].extend(selected.tolist())
        if store_clustering:
                partitioning[selected] = k
# assign remaining vertices randomly
for v in vertices:
        k = np.random.randint(0,num_clusters)
        clusters[k].append(v)

# store clustering
if store_clustering:
        print("Store clustering in file...")
        with open(hypergraph_name + '_clustering',"w") as f:
                for v in partitioning:
                        f.write(str(v) + '\n')
        partitioning = []

# precalculate power law distributions per cluster
if vertex_degree_power_law:
        print("Calculating power law distributions")
        v_dist_per_cluster = []
        for k in range(len(clusters)):
                v_dist_per_cluster.append(truncated_power_law(a=vertex_gamma, m=len(clusters[k])))                 

# build hypergraph one hyperedge at a time
hypergraph = []
used_vertices = set()
print("Generating {} hyperedges".format(num_hyperedges))
with open(export_folder + hypergraph_name,"w+") as f:
        # write header (NUM_HYPEREDGES NUM_VERTICES)
        f.write("{} {}\n".format(num_hyperedges,num_vertices))
        # draw hedge degree from power law distribution (with specific min value)
        hyperedge_degrees = (distribution.rvs(size=num_hyperedges)) + min_hyperedge_degree-1

        for he_id in range(num_hyperedges):
                print(he_id)
                # draw vertices ids randomly
                # needs to account for premade clusters
                # choose a cluster based on cluster density
                k = np.random.choice(num_clusters,p=cluster_density)
                # whether each vertex should be local or not
                nodes = [p_intraconnectivity > np.random.random_sample() for _ in range(hyperedge_degrees[he_id])]
                # select vertices from cluster k if inter_node[x] is True, from any other cluster otherwise
                vertices = set()
                for p in nodes:
                        c = k
                        if not p:
                                # different cluster to k
                                while k == c:
                                        c = np.random.randint(0,num_clusters)
                        while True:
                                if not vertex_degree_power_law:
                                        rand_index = np.random.randint(0,len(clusters[c]))
                                        vertex = clusters[c][rand_index] 
                                else:
                                        # select vertices based on power law distribution too
                                        rand_index = v_dist_per_cluster[c].rvs() - 1
                                        vertex = clusters[c][rand_index]
                                # ensure the same vertex is not added twice to a hyperedge
                                if vertex not in vertices:
                                        vertices.add(vertex)
                                        break
                                # prevent stalling if all vertices have been collected from cluster
                                if len(vertices) >= len(clusters[c]):
                                        break
                         
                
                #write hyperedge to file
                vertices = [v+1 for v in sorted(list(vertices))]
                used_vertices.update(vertices)
                vertices = [str(v) for v in vertices]
                f.write(" ".join(vertices))
                f.write("\n")
                
                # for tracking distribution
                if show_distribution:
                        hypergraph.append(len(vertices))

if show_distribution:
        plt.hist(hypergraph, bins=np.arange(max_hyperedge_degree+1)+0.5)
        plt.show()

if ensure_no_missing_vertices:
        # add missing vertices to new hypergraphs
        # use max cardinality to create as fewer hedges as possible
        missing_verts = []
        for v in range(1,num_vertices+1):
                if v not in used_vertices:
                        missing_verts.append(v)
        shuffle(missing_verts)
        extra_hedges = math.ceil(len(missing_verts) / max_hyperedge_degree)
        print("{} extra hyperedges created to include {} missing vertices".format(extra_hedges,len(missing_verts)))
        with open(export_folder + hypergraph_name,"r") as read_stream:
                with open(export_folder + 'temp.hgr',"w+") as write_stream:
                        # update header
                        header = [int(val) for val in read_stream.readline().split(' ')]
                        header[1] += extra_hedges
                        write_stream.write("{} {}\n".format(num_hyperedges + extra_hedges, num_vertices))
                        # add extra hedges first
                        while len(missing_verts) > 0:
                                new_hedge = sorted(list(itertools.islice(missing_verts,min(max_hyperedge_degree,len(missing_verts)))))
                                for vert in new_hedge:
                                        write_stream.write("{} ".format(vert))
                                        missing_verts.remove(vert)
                                write_stream.write('\n')
                        # add remainder hedges
                        write_stream.writelines(read_stream.readlines())
        # swap temp file for final file
        os.unlink(export_folder + hypergraph_name)
        shutil.move(export_folder + 'temp.hgr',export_folder + hypergraph_name)


                        