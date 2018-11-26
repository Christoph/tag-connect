from importlib import reload

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from nltk.corpus import reuters, brown
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity

# Add general functions to the project
from os import path
import sys
sys.path.append(path.abspath('../methods'))

# DATA Loading
raw = np.load("../datasets/full.pkl")

# Remove Unclear from groups/clusters
# raw["Clusters"] = data.apply(
#     lambda row: row["Clusters"]
#     .replace(";Unclear", "")
#     .replace(";;", ";"), axis=1
#     )

# train/test split
# test = raw[raw["DOI"].str.contains("2013|2012")]  # len = 197
# train = raw.drop(test.index)  # len = 1280
label_column = ["Clusters"]

# data
# y
y = np.array(raw["Title"])

# x
abstract_tfidf = TfidfVectorizer().fit(raw["Abstract"])
abstract_train = abstract_tfidf.transform(raw["Abstract"])

fulltext_tfidf = TfidfVectorizer().fit(raw["Fulltext"])
fulltext_train = fulltext_tfidf.transform(raw["Fulltext"])


# LatentDirichletAllocation
def test_settings(n_dim = 10, type = "abstract"):
    if type == "abstract":
        lda = LatentDirichletAllocation(n_dim, learning_method="batch")
        vecs = lda.fit_transform(abstract_train)

    if type == "fulltext":
        lda = LatentDirichletAllocation(n_dim, learning_method="batch")
        vecs = lda.fit_transform(fulltext_train)

    # Analysis
    similarity = cosine_similarity(vecs)

    # Get closest documents
    print(y[similarity[156].argsort()[-10:][::-1]])
    print(y[similarity[357].argsort()[-10:][::-1]])
    print(y[similarity[982].argsort()[-10:][::-1]])

# Order of operations
#  abstract 10 - 1 ok 2 meh 3 meh
#  fulltext 10 - 1 meh 2 ok 3 meh
#  fulltext 20 - 1 ok 2 meh 3 meh
#  fulltext 50 - 1 ok 2 meh 3 okmeh
#  fulltext 100 - 1 ok 2 ok 3 okmeh
#  fulltext 5 - 1 ok 2 ok 3 okmeh
# new 3 (1110 -> 982)
#  fulltext 5 - 1 ok 2 ok 3 ok
#  fulltext 10 - 1 ok 2 good 3 ok -  best
#  fulltext 7 - 1 ok 2 good 3 ok
#  abstract 5 - 1 meh 2 meh 3 meh
#  abstract 25 - 1 meh 2 meh 3 me
#  abstract 50 - 1 meh 2 meh 3 me
#  abstract 100 - 1 meh 2 meh 3 me
#  abstract 4 - 1 meh 2 meh 3 me

test_settings(n_dim = 4, type = "abstract")


def overlapping_settings(a = 10, f = 5, top = 10):
    lda = LatentDirichletAllocation(a, learning_method="batch")
    vecs_a = lda.fit_transform(abstract_train)

    lda = LatentDirichletAllocation(f, learning_method="batch")
    vecs_f = lda.fit_transform(fulltext_train)

    # Analysis
    similarity_a = cosine_similarity(vecs_a)
    similarity_f = cosine_similarity(vecs_f)

    overlap1 = np.intersect1d(y[similarity_a[156].argsort()[-top:][::-1]], y[similarity_f[156].argsort()[-top:][::-1]])
    overlap2 = np.intersect1d(y[similarity_a[357].argsort()[-top:][::-1]], y[similarity_f[357].argsort()[-top:][::-1]])
    overlap3 = np.intersect1d(y[similarity_a[982].argsort()[-top:][::-1]], y[similarity_f[982].argsort()[-top:][::-1]])

    print(len(overlap1)/top, len(overlap2)/top, len(overlap3)/top)

def save_settings(a = 10, f = 5):
    lda = LatentDirichletAllocation(a, learning_method="batch")
    vecs_a = lda.fit_transform(abstract_train)

    lda = LatentDirichletAllocation(f, learning_method="batch")
    vecs_f = lda.fit_transform(fulltext_train)

    A = pd.DataFrame(vecs_a)
    A.insert(0, "key", y)

    F = pd.DataFrame(vecs_f)
    F.insert(0, "key", y)

    A.to_csv("abstract.csv")
    F.to_csv("fulltext.csv")

# top 10 documents
# 4/10 = 0.1 0.05 0.05
overlapping_settings(4, 10, top=20)



save_settings(4, 10)
