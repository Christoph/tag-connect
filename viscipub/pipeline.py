import pandas as pd

raw = pd.read_csv("../datasets/keyvis.csv")

raw = raw.drop(range(0, len(raw), 2))
raw = raw.reset_index()
raw = raw.drop(["index", "Paper ID"], axis=1)

data = raw[["inspec", "inspec_controlled", "ieee_terms", "author_keywords", "real_author_keywords", "abstract"]]

data
