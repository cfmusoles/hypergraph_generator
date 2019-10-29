## Compare clustering done by partitioning algorithms with ground truth clustering (for generated hypergraphs) 
## Can use NMI (normalised mutual information) or https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

data_folder = '../hyperPraw/results/parallel streaming/'
ground_truth_cluster = 'small_dense_uniform.hgr_clustering'
num_clusters = 96
experiment_prefix = 'stream_parallelVertex_3_1'
hgraph = 'small_dense_uniform.hgr'
partitioning_candidates = ['parallelVertex']
plotPartitionShare = True
expected_boundaries = 3
storePlot = False
image_format = "pdf"

# load ground truth
ground_truth_clustering = np.genfromtxt(data_folder + ground_truth_cluster,skip_header=0,delimiter=",")
ground_truth_clustering = [int(i)  for i in ground_truth_clustering]

for partition in partitioning_candidates:
    #load partitioning scheme
    partitioning = np.genfromtxt(data_folder + experiment_prefix + '_' + hgraph + '_' + partition + '_partitioning__' + str(num_clusters),skip_header=0,delimiter=",")
    partitioning = [int(i)  for i in partitioning]

    similarity_score = adjusted_rand_score(ground_truth_clustering, partitioning)
    ami = adjusted_mutual_info_score(ground_truth_clustering, partitioning)
    
    #output results
    print("{}:\n Adjusted_rand {:.3f}\n Adjusted mutual info {:.3f}".format(partition,similarity_score,ami))   
    
    if plotPartitionShare:
        df = pd.DataFrame({'Cluster' : ground_truth_clustering, 'Partition' : partitioning})

        #sns.countplot(x='Partition',hue='Cluster',data=df)
        #plt.show()

        df = df.groupby(['Partition'])

        cluster_counts = [[] for _ in range(num_clusters)]
        cluster_df = pd.DataFrame(index=range(num_clusters))
        highest_ratio = [0 for _ in range(num_clusters)]
        highest_value = [0 for _ in range(num_clusters)]
        highest_cluster = [0 for _ in range(num_clusters)]

        for key,group_df in df:
            #print("the group for product '{}' has {} rows".format(key,len(group_df))) 
            cluster_counts[int(key)] = group_df.groupby('Cluster').count()['Partition'].to_list()
            value_list = group_df.groupby('Cluster').count()['Partition'].to_list()
            cluster_df[key] = 0
            for i, row in group_df.groupby('Cluster').count().iterrows():
                cluster_df[key].iloc[i] = row.values[0]
            #cluster_df[key] = value_list
            highest_ratio[int(key)] = max(value_list) / sum(value_list) * 100
            highest_value[int(key)] = sum(value_list)
            highest_cluster[int(key)] = value_list.index(max(value_list))
        
        # colour palettes https://matplotlib.org/examples/color/colormaps_reference.html
        ax = cluster_df.T.plot.bar(stacked=True,colormap='hsv',legend=False)

        ax.set_ylim([0,max(highest_value)*1.25])
        ax.set_xlabel('Partition')
        ax.set_ylabel('Elements')

        # annotate highest ratio cluster for each partition
        for i,ratio in enumerate(highest_ratio):
            if i % 2 == 0:
                y_offset = 1.07 
            else: 
                y_offset = 1.25
            ax.text(i, 
                    highest_value[i]*y_offset, 
                    '{:.0f}%\n({})'.format(ratio,highest_cluster[i]), 
                    horizontalalignment='center', 
                    verticalalignment='center')

        if storePlot:
            plt.savefig("parts." + image_format,format=image_format,dpi=1000)    
        plt.show()  

        #plot highest cluster ratio in partition for all partitions
        #plot highest value (number of elements) with colours depending on high ratio (maximo % de pertenencia a un cluster)
        
        #ratios_df = pd.DataFrame(highest_ratio,index=range(num_clusters))
        #ratios_df.plot.bar(legend=True,width=0.8,align='center')
        ratios_df = pd.DataFrame({'Elements' : highest_value, 'Ratio' : highest_ratio,'Partitions' : range(num_clusters)})
        ax = sns.barplot(x='Partitions',y='Elements',data=ratios_df)
        ax.legend().set_visible(False)

        # draw expected boundaries (lines on the start of streams)
        for i in range(expected_boundaries):
            plt.plot(0,[i*num_clusters/expected_boundaries],[max(highest_value),i*num_clusters/expected_boundaries], linewidth=2)

        plt.show()
        



    