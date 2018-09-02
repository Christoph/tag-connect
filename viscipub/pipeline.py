import pandas as pd
import numpy as np
from anytree import Node, RenderTree

# DATA Loading
raw = pd.read_csv("../datasets/keyvis.csv")

raw = raw.drop(range(0, len(raw), 2))
raw = raw.reset_index()
raw = raw.drop(["index", "Paper ID"], axis=1)

data = raw[["inspec", "inspec_controlled", "ieee_terms", "author_keywords", "real_author_keywords", "abstract"]]

# ROW INSPECTION
row = data.iloc[1]
row["inspec"]
row["inspec_controlled"]
row["ieee_terms"]
row["author_keywords"]
row["real_author_keywords"]
row["abstract"]

# DATA EXTRACTION
final = row["inspec_controlled"].split(";")
keywords = row["real_author_keywords"].split(";")
abstract = row["abstract"].split(" ")

# DATA ANALYSIS
attribute = "inspec_controlled"

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
                double.add(d.lower())
            else:
                single.add(d.lower())


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

for pre, fill, node in RenderTree(trees["blood"]):
    print("%s%s" % (pre, node.name))
