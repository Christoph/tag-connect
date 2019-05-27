from upsetplot import plot
from upsetplot import from_memberships
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

meta = pd.read_json("datasets/meta.json", orient="index").sort_index()
clusters = np.array([list(filter(None, cluster.split(";"))) for cluster in meta["Clusters"].tolist()])
titles = meta["Title"].tolist()

enc = MultiLabelBinarizer()
sets = enc.fit_transform(clusters)
unique_clusters = enc.classes_

# example = from_memberships(
#      [[],
#       ['set2'],
#       ['set1'],
#       ['set1', 'set2'],
#       ['set0'],
#       ['set0', 'set2'],
#       ['set0', 'set1'],
#       ['set0', 'set1', 'set2'],
#       ],
#       data=[56, 283, 1279, 5882, 24, 90, 429, 1957])

def upset(index):
    selection = clusters[np.where(sets[:, index] > 0 )]
    items, counts = np.unique(selection, return_counts=True)

    subset = from_memberships(items, counts)

    print("Root Class: ", unique_clusters[index])
    plot(subset)

upset(1)