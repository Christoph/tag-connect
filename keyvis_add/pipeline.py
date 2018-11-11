from importlib import reload

import pandas as pd
import numpy as np
# from anytree import Node, RenderTree
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer

# Add general functions to the project
from os import path
import sys
sys.path.append(path.abspath('../methods'))

import embedding
import vis

# DATA Loading
raw = np.load("../datasets/full.pkl")

# Remove Unclear from groups/clusters
# raw["Clusters"] = data.apply(
#     lambda row: row["Clusters"]
#     .replace(";Unclear", "")
#     .replace(";;", ";"), axis=1
#     )

# train/test split
test = raw[raw["DOI"].str.contains("2013|2012")]  # len = 197
train = raw.drop(test.index)  # len = 1280

dimensions = ["Fulltext", "Abstract", "Keywords", "Title"]
labels = ["Clusters"]

# vectorization/embedding
# x
x_train = train[dimensions]

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in train[labels]["Clusters"].tolist()])

y_train = train[labels].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values

# prepare for visualization

# Analysis
# Similarity
metric = "jaccard"  # cosine, jaccard, emd, cm
 distance = embedding.similarity_matrix(y_train, metric, as_distance=True)

# vis.simMatrix(distance)
vis.scree_plot(distance, vecs, nonlinear=False, uselda=False, usenmf=False)
vis.graph(embedding.graph_from_dist(distance, 0.60), labels)
# vis.cluster_heatmap(
#     np.array(vecs),
#     metric=metric,
#     mode="intersection",
#     order=True)
vis.scatter(vecs, labels)
vis.scatter_tsne(vecs, labels, 0.1)


# Plot local graph for a document
local_neigh = np.argwhere(distance[1, :] < 0.80).flatten()
local_dist = distance[np.ix_(local_neigh, local_neigh)]
local_labels = np.array(labels)[local_neigh]

vis.graph(
    embedding.graph_from_dist(local_dist, local_dist.mean()), local_labels
    )

# Compute jaccard distance
jaccard_distances = np.zeros([n, n])
for i in range(0, n):
    for j in range(0, n):
        a = set(data.iloc[i]["Clusters"].split(";"))
        b = set(data.iloc[j]["Clusters"].split(";"))

        jd = len(a.intersection(b))/len(a.union(b))

        jaccard_distances[i][j] = jd
