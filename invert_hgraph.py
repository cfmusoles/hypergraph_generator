# inverts hmetis graphs:
# input: hgraph as a list of hyperedges, with the vertices that belong to each on a line
# output: hgraph as a list of vertices with the hyperedges it belongs to on a line
# If stream is shuffled, then a version of the original input graph is produced with the new randomised vertex ids

from random import shuffle

folder = './'
original_hgraphs = ["symmetric_large.hgr"]#, "small_powerlaw_dense_c96.hgr",
                    #"large_uniform_sparse_c96.hgr", "small_uniform_dense_c96.hgr",
                    #"small_uniform_sparse_c96.hgr", "small_uniform_sparse_c48.hgr",
                    #"small_uniform_dense_c192.hgr",
                    #"huge_uniform_packed_c192.hgr"]
shuffle_data = False


for original_hgraph in original_hgraphs:
    print('Inverting hgraph matrix: ' + original_hgraph)
    with open(folder + original_hgraph) as f:
        # load header
        header = f.readline().rstrip().split(" ")
        num_vertices = int(header[1])
        num_hyperedges = int(header[0])
        inverted_header = " ".join([str(num_vertices), str(num_hyperedges)]) + '\n'
        #load rest of the graph
        hedges = [x.rstrip().split(" ") for x in f]

    # convert to new format
    inverted_data = [[] for x in range(num_vertices)]
    for i,hyperedge in enumerate(hedges):
        for vertex in hyperedge:
            # indexing in these files starts in 1
            inverted_data[int(vertex)-1].append(str(i+1))

    #shuffle data stream
    if shuffle_data:
        shuffle(inverted_data)
        inverted_hgraph = 'shuffled_' + original_hgraph
    else:
        inverted_hgraph = original_hgraph
    
    inverted_hgraph = 'inverted_' + inverted_hgraph

        
    #store inverted graph
    inverted_stream = [" ".join(line) + '\n' for line in inverted_data]
    with open(folder + inverted_hgraph,"w") as f:
        f.write(inverted_header)
        f.writelines(inverted_stream)

    #if shuffling, recreate original input file with new vertex ids
    if shuffle_data:
        original_shuffled_file = 'shuffled_' + original_hgraph
        shuffled_original = [[] for x in range(num_hyperedges)]
        for i,vertex in enumerate(inverted_data):
            for hyperedge in vertex:
                # indexing in these files starts in 1
                shuffled_original[int(hyperedge)-1].append(str(i+1))
        
        #store new indexed graph
        shuffled_stream = [" ".join(line) + '\n' for line in shuffled_original]
        with open(folder + original_shuffled_file,"w") as f:
            f.write(" ".join(header) + '\n')
            f.writelines(shuffled_stream)
