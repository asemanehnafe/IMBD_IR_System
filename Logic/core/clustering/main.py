import numpy as np
import os
import fasttext
import sys
sys.path.append('d:/dars/MIR project 2024/IMBD_IR_System')

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
loaded = np.load('./arrays.npz')
model = fasttext.load_model('./FastText_model.bin')
X = loaded['arr1']
y = loaded['arr2']
embeddings = np.array([model.get_sentence_vector(sentence) for sentence in X])

# 1. Dimension Reduction
#Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

#Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
dimension_reduction = DimensionReduction()
embeddings = dimension_reduction.pca_reduce_dimension(embeddings, 2)
dimension_reduction.wandb_plot_2d_tsne(np.array(embeddings), 'IMBD_IR_System', 'tsne')
dimension_reduction.wandb_plot_explained_variance_by_components(np.array(embeddings), 'IMBD_IR_System', 'pca')

# 2. Clustering
## K-Means Clustering
# Implement the K-means clustering algorithm from scratch.
# Create document clusters using K-Means.
clustering_utils = ClusteringUtils()
_, kmeans_labels = clustering_utils.cluster_kmeans(embeddings, 4)

#Run the algorithm with several different values of k.
# For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
#   Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
#   Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

# for k in range(2, 10):
#    clustering_utils.visualize_kmeans_clustering_wandb(embeddings, k, 'IMBD_IR_System', 'kmeans clustering')

clustering_utils.plot_kmeans_cluster_scores(embeddings, y, [k for k in range(2, 10)], 'IMBD_IR_System', 'kmeans cluster_scores')
clustering_utils.visualize_elbow_method_wcss(embeddings, [k for k in range(2, 10)], 'IMBD_IR_System', 'elbow method wcss')

## Hierarchical Clustering
# Perform hierarchical clustering with all different linkage methods.
#Visualize the results.
single_link_labels = clustering_utils.cluster_hierarchical_single(embeddings)
complete_link_labels = clustering_utils.cluster_hierarchical_complete(embeddings)
average_link_labels = clustering_utils.cluster_hierarchical_average(embeddings)
ward_link_labels = clustering_utils.cluster_hierarchical_ward(embeddings)
for linkage_method in ['single', 'complete', 'average', 'ward']:
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(embeddings, 'IMBD_IR_System', linkage_method, f'{linkage_method} Linkage')

# 3. Evaluation
# Using clustering metrics, evaluate how well your clustering method is performing.
clusteringMetrics = ClusteringMetrics()
print('kmens scores:')
print("Silhouette Score:", clusteringMetrics.silhouette_score(embeddings, kmeans_labels))
print("Purity Score:", clusteringMetrics.purity_score(y, kmeans_labels))
print("Adjusted Rand Score:", clusteringMetrics.adjusted_rand_score(y, kmeans_labels))

print('\nsingle link scores:')
print("Silhouette Score:", clusteringMetrics.silhouette_score(embeddings, single_link_labels))
print("Purity Score:", clusteringMetrics.purity_score(y, single_link_labels))
print("Adjusted Rand Score:", clusteringMetrics.adjusted_rand_score(y, single_link_labels))

print('\ncomplete link scores:')
print("Silhouette Score:", clusteringMetrics.silhouette_score(embeddings, complete_link_labels))
print("Purity Score:", clusteringMetrics.purity_score(y, complete_link_labels))
print("Adjusted Rand Score:", clusteringMetrics.adjusted_rand_score(y, complete_link_labels))

print('\n average link scores')
print("Silhouette Score:", clusteringMetrics.silhouette_score(embeddings, average_link_labels))
print("Purity Score:", clusteringMetrics.purity_score(y, average_link_labels))
print("Adjusted Rand Score:", clusteringMetrics.adjusted_rand_score(y, average_link_labels))

print('ward link score')
print("Silhouette Score:", clusteringMetrics.silhouette_score(embeddings, ward_link_labels))
print("Purity Score:", clusteringMetrics.purity_score(y, ward_link_labels))
print("Adjusted Rand Score:", clusteringMetrics.adjusted_rand_score(y, ward_link_labels))