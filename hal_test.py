import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("datasets/raw.csv")
subset = data[0:100000]

used_dataset = subset

grouped = used_dataset.groupby("song_id")
keys = list(grouped.groups.keys())

print("Prep done")

tag_docs = []

col = []
neighbour_data = []
row = []

vocabulary = {}

for index in range(0, len(keys) - 1):
    tags = used_dataset.ix[grouped.groups[keys[index]]].tag_name.tolist()

    doc = " ".join(tags)

    tag_docs.append(doc.replace("-", " "))

    for ind in range(0, len(tags)):
        # Add focus word to vocab
        row_index = vocabulary.setdefault(tags[ind], len(vocabulary))

        # neighbours
        for i in range(0, len(tags)):
            if ind is not i:
                term = tags[i]
                value = 1

                index = vocabulary.setdefault(term, len(vocabulary))

                col.append(row_index)
                row.append(index)
                neighbour_data.append(value)

print("Vecs done")

hal = csr_matrix((neighbour_data, (row, col)), shape=(
    len(vocabulary), len(vocabulary)), dtype=float)

max_value = 1 / hal.max()
hal = hal.multiply(max_value)

print("Hal done")

sim = cosine_similarity(hal)

print("Sim done")

# np.save("tag_sim", sim)

# Get context row
context_row = sim[vocab["rock"]].copy()

# Remove self similarity and rescale
min_max_scaler = MinMaxScaler()

context_row[context_row >= 1.] = context_row.min()
rescaled = min_max_scaler.fit_transform(context_row.reshape(-1, 1))

# Get similarity list
similarities = pd.DataFrame(
    {'word': sorted(vocab, key=vocab.get),
     'similarity': rescaled.flatten()
     })

similarities = similarities.sort_values(
    "similarity", ascending=False)
