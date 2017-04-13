import pandas as pd
import numpy as np
import os

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText


data = pd.read_csv("datasets/tag_docs.csv")
subset = data[0:10000]

used_dataset = subset

sentences = [d.split(".") for d in used_dataset.tag_doc]
fasttext = " ".join(used_dataset.tag_doc).replace(".", " ")

with open('datasets/fasttext', 'w') as file:
    file.write(fasttext)

wv_model = Word2Vec(sentences, size=50, window=5, min_count=5, workers=4)
ft_model = FastText.train("/home/chris/source/fastText/fasttext", corpus_file="datasets/fasttext", model="skipgram")

wv_model.most_similar("rock")
ft_model.most_similar("rock")

# ! Already saved complete models in datasets/ !
# wv_model.save("datasets/wv_model")
# ft_model.save("datasets/ft_model")
