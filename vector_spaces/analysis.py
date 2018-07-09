import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# idfs = pd.read_csv("datasets/idf_scores.csv")
ft_counts = pd.read_csv("datasets/ft_counts_0.9.csv")
ft_neighs = pd.read_csv("datasets/ft_neighbours_0.9.csv")
wv_counts = pd.read_csv("datasets/wv_counts_0.9.csv")
# wv_neighs = pd.read_csv("datasets/wv_neighbours_0.9.csv")

wv_model = Word2Vec.load('datasets/wv_model')
ft_model = FastText.load("datasets/ft_model")

wv_model.most_similar("electronic", topn=10)
top = ft_model.most_similar("electronic", topn=100)

df = pd.DataFrame(columns=('tag_name', 'sim', 'count', 'neighbours_count', 'is_neighbour_count'))

for i, row in enumerate(top):
    tag = row[0]
    sim = row[1]
    count = ft_counts[ft_counts.tag_name == tag]["count"].values[0]
    incount = ft_counts[ft_counts.tag_name == tag]["is_neighbour_count"].values[0]
    nhcount = ft_counts[ft_counts.tag_name == tag]["neighbour_count"].values[0]

    df.loc[len(df)] = [tag, sim, count, nhcount, incount]

    # if count == ncount:
    #     df.loc[len(df)] = [tag, sim, count, nhcount, incount]
