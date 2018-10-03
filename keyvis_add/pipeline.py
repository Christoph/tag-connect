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
raw = pd.read_csv("../datasets/keyvis_texts.csv")
# raw

data = raw[["Title", "Clusters"]]
n = len(data)
# labels = np.zeros(n)
labels = data["Title"].tolist()

# Create one hot vectors
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in data["Clusters"].tolist()])

data['Vector'] = data.apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1)
# list(enc.classes_)

vecs = data["Vector"].tolist()



distances = np.zeros([n, n])

for i in range(0, n):
    for j in range(0, n):
        a = set(data.iloc[i]["Clusters"].split(";"))
        b = set(data.iloc[j]["Clusters"].split(";"))

        jd = len(a.intersection(b))/len(a.union(b))

        distances[i][j] = jd

np.unique(distances)

vis.graph(embedding.graph_from_sim(distances, 0.5), labels)
