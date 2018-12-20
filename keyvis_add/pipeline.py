from importlib import reload

import pandas as pd
import numpy as np
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
raw = raw.reset_index(drop=True)

# Remove Unclear from groups/clusters
raw["Clusters"] = raw.apply(
    lambda row: row["Clusters"]
    .replace("Unclear", "")
    .replace(";;", ";"), axis=1
    )


# METADATA
meta = raw.drop(["Fulltext"], axis=1)
meta['type'] = meta.apply(lambda row: 'new' if any(x in row["DOI"] for x in ["2013", "2012"]) else 'old', axis=1)
meta.to_json("meta.json", orient="index")

fulltext = raw["Fulltext"]
fulltext.to_json("fulltext.json", orient="index")

# EMBEDDINGS


# CLASSIFICATION
# train/test split for classification
test = raw[raw["DOI"].str.contains("2013|2012")]  # len = 197
train = raw.drop(test.index)  # len = 1280

raw

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in train["Clusters"].tolist()])

y_train = train["Clusters"].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values

# prepare for visualization

# Analysis
# Similarity
metric = "jaccard"  # cosine, jaccard, emd, cm
distance = embedding.similarity_matrix(y_train, metric, as_distance=True)

# vis.simMatrix(distance)
vis.scatter(vecs, labels)
vis.scatter_tsne(vecs, labels, 0.1)
