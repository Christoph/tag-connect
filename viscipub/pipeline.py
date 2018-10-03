import pandas as pd
import numpy as np
from anytree import Node, RenderTree
from textblob import TextBlob

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist

# DATA Loading
raw = pd.read_csv("../datasets/keyvis.csv")

raw = raw.drop(range(0, len(raw), 2))
raw = raw.reset_index()
raw = raw.drop(["index", "Paper ID"], axis=1)

data = raw[["inspec", "inspec_controlled", "ieee_terms", "author_keywords", "real_author_keywords", "abstract"]]
data = data.drop(range(2737, len(data)))
len(data)
# ROW INSPECTION
row = data.iloc[1]
row["inspec"]
row["inspec_controlled"]
row["ieee_terms"]
row["author_keywords"]
row["real_author_keywords"]
row["abstract"]

# ROW STATISTICS
total_size = len(data)
total_size
total_size - data["inspec"].isnull().sum()
total_size - data["inspec_controlled"].isnull().sum()
total_size - data["ieee_terms"].isnull().sum()
total_size - data["author_keywords"].isnull().sum()
total_size - data["real_author_keywords"].isnull().sum()
total_size - data["abstract"].isnull().sum()

len(data.dropna(axis = 0, how = "any"))
data = data.dropna(axis = 0, how = "any")

# DATA ANALYSIS

# LDA


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
        out.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return out

vectorizers = []
ldas = []

type = "tf"
n_features = 10

for i in range(0, 10):
    if type == "tfidf":
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    else:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    vecs = vectorizer.fit_transform(data["abstract"].tolist())
    lda = LatentDirichletAllocation(learning_method="batch").fit(vecs)

    vectorizers.append(vectorizer)
    ldas.append(lda)


[print_top_words(lda, vectorizer.get_feature_names(), n_features) for lda, vectorizer in zip(ldas, vectorizers)]
words = [[set(d) for d in get_top_words(lda, vectorizer.get_feature_names(), n_features)] for lda, vectorizer in zip(ldas, vectorizers)]

distances = np.eye(len(words))
intersections = np.eye(len(words), dtype=object)
differences = np.eye(len(words), dtype=object)

for i, model1 in enumerate(words):
    for j, model2 in enumerate(words):
        temp = np.eye(n_features)
        inter = np.eye(n_features, dtype=object)
        diff = np.eye(n_features, dtype=object)
        for k, t1 in enumerate(model1):
            for l, t2 in enumerate(model2):
                temp[k][l] = 1 - len(t1.intersection(t2))/len(t1.union(t2))

                inter[k][l] = " ".join(t1.intersection(t2))
                diff[k][l] = " ".join(t1.symmetric_difference(t2))

        distances[i][j] = temp.mean()
        intersections[i][j] = inter
        differences[i][j] = diff

intersections[0, 0:4]
differences[0, 0:4]

# COUNTS
counts = []

for row in data.iterrows():
    if row[1][attribute] is not np.NAN:
        row_data = row[1][attribute].split(";")

        for key in row_data:
            counts.append(len(key.split(" ")))


np.unique(counts, return_counts=True)

# WORD TREES
single = set()
double = set()

for row in data.iterrows():
    if row[1][attribute] is not np.NAN:
        row_data = row[1][attribute].split(";")

        for d in row_data:
            if len(d.split(" ")) > 1:
                double.add(" ".join(TextBlob(d).words.lemmatize().singularize().lower()))
            else:
                single.add(" ".join(TextBlob(d).words.lemmatize().singularize().lower()))

single
double

leafs = []
roots = {}
childs = {}
trees = {}

for s in single:
    root = Node(s)
    roots[s] = root

    for d in double:
        if s in d.split(" "):
            leafs.append(Node(d, parent=root))

    childs[root] = root.children

    if len(childs[root]) > 0:
        trees[s] = root

trees
for pre, fill, node in RenderTree(trees["vector"]):
    print("%s%s" % (pre, node.name))

len(single)
len(trees)
len(double)

double_vocab = set([w for sublist in double for w in sublist.split(" ")])
len(double_vocab)

len(single.intersection(double_vocab))
len(single.difference(double_vocab))
len(double_vocab.difference(single))

len(single.union(double_vocab))

included = set()
not_included = set()
attribute = "inspec_controlled"
included_in = ["author_keywords", "ieee_terms", "abstract"]
total_counter = 0
included_counter = 0
not_included_counter = 0
stopper = 0

for row in data.iterrows():
    row_data = [" ".join(TextBlob(d).words.lemmatize().lower().singularize()) for d in row[1][attribute].split(";")]
    text = " ; ".join([" ".join(TextBlob(row[1][t]).words.lemmatize().lower().singularize()) for t in included_in])

    for keyword in row_data:
        total_counter += 1

        if keyword in text:
            included.add(keyword)
            included_counter += 1
        else:
            not_included.add(keyword)
            not_included_counter += 1

            # if keyword == "visual perception" and stopper < 5:
            #     stopper += 1
            #     print(text)

total_counter
included_counter
not_included_counter
included
not_included
