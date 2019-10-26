# hypergraph_generator

Parametric hypergraph generator. Simple hypergraph generator that allows the creation of custom hypergraps that follow power-law distribution in both hyperedge cardinality and vertex degree. The user can define the number of seed clusters as well as the density of each cluster.

The currently supported parameters are as follows:

- Number of vertices
- Number of hyperedges
- Number of internal clusters
- Cluster density
- Probability of intra cluster connectivity
- Hyperedge size gamma (power-law distribution parameter)
- Max hyperedge cardinality
- Min hyperedge cardinality
- Vertex degree gamma (power-law distribution parameter)

The current version is a simple python script that includes the parameters at the top of the file `hypergraph_generator.py`. To run it, enter:

```
python hypergraph_generator.py
```

The repository also includes a simple script that can compare two cluster or partitioning allocations using standard information theory metrics. The script is `compare_clusters.py` and is best used to compare ground truth clusters generated with the hypergraph generator and partitioning allocations found by partitioning algorithms. The properties are currently on top of the mentioned script, that can be run entering:

```
python compare_clusters.py
```
