import pandas as pd
import numpy as np


data = pd.read_csv("datasets/raw.csv")
subset = data[0:100000]

artist = data["artist_name"].unique()
tag = data["tag_name"].unique()
track = data["track_name"].unique()

used_dataset = data
used_dataset["tag_name"].fillna("", inplace=True)

grouped = used_dataset.groupby("song_id")
keys = list(grouped.groups.keys())

tag_docs = []
song_ids = []

for index in range(0, len(keys) - 1):
    tags = used_dataset.ix[grouped.groups[keys[index]]].tag_name.tolist()

    doc = ".".join(tags)

    tag_docs.append(doc)
    song_ids.append(keys[index])

out = pd.DataFrame({
    "song_id": song_ids,
    "tag_doc": tag_docs})

out.to_csv("datasets/tag_docs.csv", index=False)
