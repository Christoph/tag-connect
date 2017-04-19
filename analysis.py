import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText


# idfs = pd.read_csv("datasets/idf_scores.csv")
counts = pd.read_csv("datasets/counts.csv")
n_counts = pd.read_csv("datasets/neighbour_counts.csv")
wv_model = Word2Vec.load('datasets/wv_model')
ft_model = FastText.load("datasets/ft_model")

wv_model.most_similar("jazz")
top = ft_model.most_similar("jazz", topn=200)

df = pd.DataFrame(columns=('tag_name', 'sim', 'count', 'ncount'))

for i, row in enumerate(top):
    tag = row[0]
    sim = row[1]
    count = counts[counts.tag_name == tag]["count"].values[0]
    ncount = n_counts[n_counts.tag_name == tag]["count"].values[0]

    if count <= 5:
        df.loc[len(df)] = [tag, sim, count, ncount]
