import pandas as pd
import numpy as np

data = pd.read_csv("datasets/all_raw.csv")

temp = data[["TagName", "SongID"]]
groups = temp.groupby("TagName").count()
groups = groups.reset_index()
groups.columns = ["TagName", "count"]


greaterfive = groups[groups["count"] > 5]
tofive = groups[groups["count"] <= 5]

ones = tofive[tofive["count"] == 1]
twotofive = tofive[tofive["count"] > 1]

'''
Total tags: 529086
Tags appearing once: 61%
Tags appearing two to five times: 22%
Tags appearing more than five times: 17%
'''

ones = ones.reset_index().drop("index", axis=1)
rnd = np.random.randint(len(ones), size=300)

chris = ones.loc[rnd[0:100], :]
torsten = ones.loc[rnd[100:200], :]
mohsen = ones.loc[rnd[200:300], :]

chris.to_csv("chris.csv", index=False)
torsten.to_csv("torsten.txt", index=False)
mohsen.to_csv("mohsen.txt", index=False)
