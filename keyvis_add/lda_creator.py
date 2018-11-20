from importlib import reload

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer
from gensim import corpora, models, matutils

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
label_column = ["Clusters"]

# data

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in train[label_column]["Clusters"].tolist()])

y_train = train[label_column].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values

# x
abstract_tfidf = TfidfVectorizer().fit(raw["Abstract"])
abstract_train = abstract_tfidf.transform(train["Abstract"])
abstract_test = abstract_tfidf.transform(test["Abstract"])

fulltext_tfidf = TfidfVectorizer().fit(raw["Fulltext"])
fulltext_train = fulltext_tfidf.transform(train["Fulltext"])
fulltext_test = fulltext_tfidf.transform(test["Fulltext"])


# LatentDirichletAllocation
lda = LatentDirichletAllocation(learning_method="batch")
abstract_lda = lda.fit_transform(abstract_train)
abstract_perp = lda.perplexity(abstract_test)
abstract_perp

lda = LatentDirichletAllocation(learning_method="batch")
fulltext_lda = lda.fit_transform(fulltext_train)
fulltext_perp = lda.perplexity(fulltext_test)
fulltext_perp


# gensim version
texts = [[word for word in document.lower().split()] for document in raw["Abstract"]]
dct = corpora.Dictionary(texts)
corpus = [dct.doc2bow(line) for line in texts]

gensim_lda = models.LdaModel(corpus, num_topics=10)
gensim_lda.
