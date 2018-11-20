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


lda = LatentDirichletAllocation(learning_method="batch")
fulltext_lda = lda.fit_transform(fulltext_train)

# Articles

articles = pd.read_csv("../datasets/articles.csv")
articles["publication"].unique()
articles

# y
y = articles["publication"]

# x
title_tfidf = TfidfVectorizer().fit(articles["title"])
title_vecs = title_tfidf.transform(articles["title"])

content_tfidf = TfidfVectorizer().fit(articles["content"])
content_vecs = content_tfidf.transform(articles["content"])

# clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.1)
clf = RandomForestClassifier(n_estimators=15)

scores = cross_val_score(clf, title_vecs, y, cv=5)

scores

clf.fit(title_vecs, y)
predicted = clf.predict(title_vecs)

metrics.average_precision_score(y, predicted)

print(metrics.classification_report(y, predicted))
metrics.confusion_matrix(y, predicted)
