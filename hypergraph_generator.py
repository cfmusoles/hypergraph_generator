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
import seaborn as sns


# hypergraph parameters
hypergraph_name = "test.hgr"
export_folder = './'
num_vertices = 10000
num_hyperedges = 10000
num_clusters = 1
cluster_density = [1.0/num_clusters for _ in range(num_clusters)]       # probability that a vertex belongs to each cluster
#cluster_density = [0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.05,0.20,0.25,0.05,0.05]
p_intraconnectivity = 1.0
hyperedge_gamma = -1.0                        # determines the skewness of the distribution (higher values more skewed to the left). Must be != 0 (gamma == 1 for the uniform distribution)
min_hyperedge_degree = 20                     # must be > 1
max_hyperedge_degree = 20                    # must be >= min_hyperedge_degree
vertex_degree_power_law = True          # whether drawing vertex ids is done using power law distribution (much slower)
vertex_gamma = -0.5                    #  Must be != 0 (gamma == 1 for the uniform distribution) careful! low values of this may prevent the graph from finishing (since vertices cannot be added twice to the same hyperedge, the roll will be rolled and may always get the same most probable answers)
ensure_no_missing_vertices = False       # ensure that all vertex ids are present in at least one hedge
avoid_duplicate_vertices = True        # avoid same vertex present more than once in a hyperedge (only set to False for vertex degree distribution graphs)
show_distribution = True
store_clustering = False


# sanity check to test that desired cluster sizes are large enough for desired hyperedge degrees
for i,r in enumerate(cluster_density):
        cluster_size = num_vertices * r
        if cluster_size < max_hyperedge_degree:
                print("Cluster size is not large enough. Cluster {} size is {}, but max hyperedge degree is {}.".format(i,cluster_size,max_hyperedge_degree))
                exit()
if p_intraconnectivity < 1 and num_clusters == 1:
        print("p_intraconnectivity must be 1.0 (no interconnectivity) if clusters == 1.")
        num_clusters = 1
        p_intraconnectivity = 1.0

#power law distributions
def truncated_power_law(a, m):
    x = np.arange(1, m+1, dtype='float')
    pmf = 1/x**a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, m+1), pmf))

def generate_powerlaw(minimum, maximum, gamma, size):
        r = np.random.random(size=size)
        ming, maxg = minimum**gamma, (maximum)**gamma
        return [math.ceil(x) for x in (ming + (maxg - ming)*r)**(1./gamma)]

def powerlaw_pdf(x, minimum, maximum, gamma, stepped=False):
        ag, bg = (minimum)**gamma, (maximum)**gamma
        if stepped:
                # return prob of the real interval for [x, x-1)
                step = 2
                return sum([gamma * (x+decimal/10)**(gamma-1) / (bg - ag) for decimal in range(0, 10, step)])
        else:
                return gamma * x**(gamma-1) / (bg - ag)



# will sample from 1 to max_hyperedge_degree with probability power law degree
distribution = truncated_power_law(a=hyperedge_gamma, m=(max_hyperedge_degree-min_hyperedge_degree+1))

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

# sort vertices in clusters
for k in range(num_clusters):
        clusters[k] = sorted(clusters[k])

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
        hyperedge_degrees = generate_powerlaw(min_hyperedge_degree-1, max_hyperedge_degree, hyperedge_gamma, num_hyperedges)

        # TEST: Plot expected powerlaw distribution
        if show_distribution:
                mybins=np.linspace(min(hyperedge_degrees)-1,max(hyperedge_degrees)+1,num=max(hyperedge_degrees)-min(hyperedge_degrees)+2)
                g = sns.distplot(hyperedge_degrees,kde=False,bins=mybins,hist_kws={'edgecolor':'black'})
                g.set_xscale('linear')
                g.set_yscale('linear')
                g.set_title("Hyperedges sizes")
                g.set_xlabel("Size of hyperedge")
                g.set_ylabel("Count")
                x = range(min_hyperedge_degree,max_hyperedge_degree)
                y = [powerlaw_pdf(value, min_hyperedge_degree, max_hyperedge_degree, hyperedge_gamma)*num_hyperedges for value in x]
                plt.yscale('linear')
                plt.xscale('linear')
                plt.plot(x,y,'r')
                plt.savefig(hypergraph_name + "_hedge_size." + 'pdf',format='pdf',dpi=1000)
                plt.show()

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
                for idx, p in enumerate(nodes):
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
                                        rand_index = generate_powerlaw(1, len(clusters[c])+1, vertex_gamma, 1)[0] - 2
                                        #rand_index = v_dist_per_cluster[c].rvs() - 1
                                        vertex = clusters[c][rand_index]
                                if not avoid_duplicate_vertices:
                                        vertices.add(vertex)
                                        break
                                else:    
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
                # for tracking distribution
                if show_distribution:
                        hypergraph.append(vertices)
                vertices = [str(v) for v in vertices]
                f.write(" ".join(vertices))
                f.write("\n")


if show_distribution:
        # plot vertex degree
        # x ==> how many hyperedges a vertex is present in
        # y ==> count of vertices on each bin
        vertices = [vid for vertex in hypergraph for vid in vertex]
        unique, vertex_counts = np.unique(vertices,return_counts=True)
        vert_count = len(unique) # required because some original vertices may not have been assigned
        # max_value = max(vertex_counts)
        # min_value = min(vertex_counts)
        # mybins=np.linspace(min_value,max_value,num=max_value-min_value)
        # g = sns.distplot(vertex_counts,kde=False,bins=mybins,hist_kws={'edgecolor':'black'})
        # g.set_xscale('linear')
        # g.set_yscale('linear')
        # g.set_title("Vertex degrees")
        # g.set_xlabel("Vertex degree")
        # g.set_ylabel("Count")
        # x = range(min_value,max_value)
        # if not vertex_degree_power_law:
        #         vertex_gamma = 1
        #         plt.xscale('linear')
        # else:
        #         plt.xscale('log')
        # y = [powerlaw_pdf(value, min_value, max_value, vertex_gamma, stepped=True)*vert_count/5 for value in x]
        # plt.yscale('linear')
        
        # plt.plot(x,y,'r')
        # plt.savefig(hypergraph_name + "_vertex_degree." + 'pdf',format='pdf',dpi=1000)
        # plt.show()
        
        # plot vertex selection distribution (per vertex id)
        # low ids should be represented more often
        sorted_zip = sorted(zip(unique, vertex_counts))
        vert_degrees = [count for vid, count in sorted_zip]
        plt.bar(range(1, len(vert_degrees)+1), vert_degrees)
        plt.yscale('linear')
        plt.title("Vertex selection")
        plt.xlabel("Vertex index")
        plt.ylabel("Vertex degree")
        x = range(1, len(vert_degrees)+1)
        average_hyperedge_degree = (max_hyperedge_degree+min_hyperedge_degree)/2
        def expected_outcome(samples, prob, trials):
                outcomes = [1 for value in np.random.binomial(samples, prob, trials) if value > 0]
                return np.sum(outcomes)
        if not vertex_degree_power_law:
                vertex_gamma = 1
                plt.xscale('linear')
                y = [average_hyperedge_degree*num_hyperedges/num_vertices for value in x]
        else:
                plt.xscale('log')
                distrib = [powerlaw_pdf(value, 1, len(vert_degrees)+1, vertex_gamma, stepped=False)*len(vertices) for value in x]
                max_prob = sum(distrib)
                y = [expected_outcome(average_hyperedge_degree, dist/max_prob, num_hyperedges) for dist in distrib]
        plt.plot(x,y,'r')
        plt.savefig(hypergraph_name + "_vertex_sampling." + 'pdf',format='pdf',dpi=1000)
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


                        