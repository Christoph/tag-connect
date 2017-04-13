import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText

idfs = pd.read_csv("datasets/idf_scores.csv")
wv_model = Word2Vec.load('datasets/wv_model')
ft_model = FastText.load("datasets/ft_model")

wv_model.most_similar("rock")
ft_model.most_similar("rock")
