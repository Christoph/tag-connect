import pandas as pd
import numpy as np
from anytree import Node, RenderTree
from textblob import TextBlob

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
