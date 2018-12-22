from importlib import reload

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import spacy
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, TruncatedSVD

nlp = spacy.load('en_core_web_md', disable=['ner'])

# Add general functions to the project
from os import path
import sys
sys.path.append(path.abspath('../methods'))

import embedding
import vis

# Helper functions
def lemmatization(text, stopwords):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in text.sents:
        texts_out.extend([token.lemma_ for token in sent if
                          token.lemma_ not in stop_words and
                          not token.lemma_ == "-PRON-"])
    return texts_out

def get_top_words(model, tfidf, n_top_words):
    out = []
    feature_names = tfidf.get_feature_names()
    idf = tfidf.idf_
    vocab = tfidf.vocabulary_

    for topic_idx, topic in enumerate(model.components_):
        words = [(feature_names[i], idf[vocab[feature_names[i]]]) for i in topic.argsort()[:-n_top_words - 1:-1]]
        out.append(words)
    return out

# DATA Loading
# raw = np.load("../datasets/full.pkl")
# raw = raw.reset_index(drop=True)
# stop_words = stopwords.words('english')
# docs = (nlp(text) for text in raw["Fulltext"].tolist())
# full_lemma = (lemmatization(doc, stop_words) for doc in docs)
# fulltext_texts = [" ".join(text) for text in full_lemma]
# pd.DataFrame(fulltext_texts).to_json("fulltext_lemma.json", orient="index")


fulltexts = pd.read_json(
    "../datasets/fulltext_lemma.json", orient="index").sort_index()
meta = pd.read_json("../datasets/meta.json", orient="index")
stop_words = stopwords.words('english')

# CLASSIFICATION
# train/test split for classification
test_index = meta[meta["type"] == "new"].index  # len = 197
train_index = meta.drop(test_index).index  # len = 1280

test = fulltexts.iloc[test_index]
train = fulltexts.iloc[train_index]

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()])

y_train = meta.iloc[train_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values

# x
fulltext_tfidf = TfidfVectorizer(max_df=0.5).fit(train[0].tolist())
fulltext_vecs = fulltext_tfidf.transform(train[0].tolist())

full_nmf = NMF(10)
full_vecs = full_nmf.fit_transform(fulltext_vecs)
full_vecs = np.asarray(full_vecs, dtype=np.object)

x_train = full_vecs

# classifiers
