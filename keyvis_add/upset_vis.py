from upsetplot import plot
from upsetplot import from_memberships
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# action: Dataframe
test_data = np.load("datasets/test_data_dict.pckl")
meta = pd.read_json("datasets/meta.json", orient="index").sort_index()
clusters = [list(filter(None, cluster.split(";"))) for cluster in meta["Clusters"].tolist()]
titles = meta["Title"].tolist()

enc = MultiLabelBinarizer()
unique_clusters = enc.classes_
sets = enc.fit_transform(clusters)

upset_data = {}

for c in unique_clusters:
    upset_data[c] = pd.DataFrame(columns=['Title'])

for index, vec in enumerate(sets):
    temp = unique_clusters[vec.astype(bool)]
    for cluster in temp:
        df = upset_data[cluster]
        df = df.append({
            "Title": meta.iloc[index]["Title"]
            }, ignore_index=True)
        upset_data[cluster] = df


subset = from_memberships(clusters)
subset[0]
plot(subset)