from importlib import reload
import pandas as pd
import numpy as np
from anytree import Node, RenderTree
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
# raw

data = raw[["Title", "Clusters", "DOI"]]
n = len(data)
# labels = np.zeros(n)
labels = data.apply(lambda row: " | ".join([row["Title"], row["DOI"]]) + "<br>" + "<br>".join(row["Clusters"].split(";")), axis=1).tolist()

# Create one hot vectors
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in data["Clusters"].tolist()])

data['Vector'] = data.apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1)
# list(enc.classes_)
# data
vecs = data["Vector"].tolist()

# Analysis
# Similarity
metric = "jaccard" # cosine, jaccard, emd, cm
distance = embedding.similarity_matrix(vecs, metric, as_distance=True)

# vis.simMatrix(distance)
vis.scree_plot(distance, vecs, nonlinear=False, uselda=False, usenmf=False)
vis.graph(embedding.graph_from_dist(distance, 0.55), labels)
# vis.cluster_heatmap(
#     np.array(vecs),
#     metric=metric,
#     mode="intersection",
#     order=True)
vis.scatter(vecs, labels)
vis.scatter_tsne(vecs, labels, 0.1)


# Plot local graph for a document
list(enc.classes_)

element = 0
local_neigh = np.argwhere(distance[element, :] < 0.75).flatten()
local_dist = distance[np.ix_(local_neigh,local_neigh)]
local_labels = np.array(labels)[local_neigh]

vis.graph(embedding.graph_from_dist(local_dist, 0.55), local_labels)

# Plot local leighborhood for keyword


# Compute jaccard distance
jaccard_distances = np.zeros([n, n])
for i in range(0, n):
    for j in range(0, n):
        a = set(data.iloc[i]["Clusters"].split(";"))
        b = set(data.iloc[j]["Clusters"].split(";"))

        jd = len(a.intersection(b))/len(a.union(b))

        jaccard_distances[i][j] = jd
