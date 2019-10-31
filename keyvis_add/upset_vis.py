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

pd.DataFrame(unique_clusters)[50:100]

def upset(index):
    selection = clusters[np.where(sets[:, index] > 0 )]
    items, counts = np.unique(selection, return_counts=True)

    subset = from_memberships(items, counts)
    sub_classes = np.unique([item for sublist in items for item in sublist])

    print("Root Class: ", unique_clusters[index])
    print("# Papers: ", len(selection))
    print("# Labels: ", len(sub_classes))
    print("# Classes: ", len(items))
    if len(items) > 40 or len(sub_classes) > 20:
        print("Too many items")
    else:
        plot(subset)

# working 0, 1, 87, 89, 30, 48
upset(1)
