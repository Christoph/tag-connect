from importlib import reload

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation, NMF
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


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_top_words(model, feature_names, n_top_words):
    out = []
    for topic_idx, topic in enumerate(model.components_):
        topics = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        out.extend(topics)

    return set(out)

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
abstract_tfidf = TfidfVectorizer(stop_words="english").fit(raw["Abstract"])
abstract_train = abstract_tfidf.transform(raw["Abstract"])

full_text_tfidf = TfidfVectorizer(stop_words="english").fit(raw["Fulltext"])
fulltext_train = full_text_tfidf.transform(raw["Fulltext"])

# params
n_dim = [4,6,8,10,12,15,20,25,30]
algorithms = ["lda", "nmf"]
datasets = ["abstract", "fulltext"]

n_runs_per_setting = 5
topic_consistency_thresholds = [5, 10, 20]

# Create all parameter permutations
protocol = pd.DataFrame(columns=["Algorithm", "Dataset", "Dimensions", "Doc", "JaccTop5", "JaccTop10", "JaccTop20"])
runs = np.stack(np.meshgrid(algorithms, datasets, n_dim), -1).reshape(-1, 3)

# Compute
for run in runs[8:10]:
    ldas = []
    vecs = []
    similarities = []
    row = []

    # Set variables
    if run[1] == "abstract":
        used = abstract_train
        tfidf = abstract_tfidf
    if run[1] == "fulltext":
        used = fulltext_train
        tfidf = full_text_tfidf

    # Fill protocol
    row.extend(run)

    # Compute models
    for iteration in range(0, n_runs_per_setting):
        if run[0] == "lda":
            lda = LatentDirichletAllocation(int(run[2]), learning_method="batch")
            vec = lda.fit_transform(used)
        if run[0] == "nmf":
            lda = NMF(int(run[2]))
            vec = lda.fit_transform(used)

        similarity = cosine_similarity(vec)

        ldas.append(lda)
        vecs.append(vec)
        similarities.append(similarity)

    # Document neighborhood consistency
    doc_sim = []
    for i in range(0, len(similarities)):
        for j in range(i+1, len(similarities)):
            doc_sim.append(abs(similarities[i] - similarities[j]).sum())

    row.append(np.mean(doc_sim))
    # print(y[similarity1[156].argsort()[-10:][::-1]])
    # print(y[similarity2[156].argsort()[-10:][::-1]])

    # Topic consistency
    for thres in topic_consistency_thresholds:
        top_sim = []

        for i in range(0, len(similarities)):
            topics1 = get_top_words(ldas[i], tfidf.get_feature_names(), thres)
            for j in range(i+1, len(similarities)):
                topics2 = get_top_words(ldas[j], tfidf.get_feature_names(), thres)

                top_sim.append(len(topics1.intersection(topics2))/len(topics1.union(topics2)))

        row.append(np.mean(top_sim))
        # print_top_words(lda1, tfidf.get_feature_names(), 10)
        # print_top_words(lda2, tfidf.get_feature_names(), 10)

    protocol = protocol.append(pd.DataFrame([row], columns=protocol.columns))



protocol
protocol.to_csv("runs.csv")

# Filter dimensions
# np.median(vecs1, axis=0)
# np.std(vecs1, axis=0)

print_top_words(ldas[0], tfidf.get_feature_names(), 10)
