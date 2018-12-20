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


# DATA Loading
raw = np.load("../datasets/full.pkl")
raw = raw.reset_index(drop=True)

# Remove Unclear from groups/clusters
raw["Clusters"] = raw.apply(
    lambda row: row["Clusters"]
    .replace("Unclear", "")
    .replace(";;", ";"), axis=1
    )


# METADATA
meta = raw.drop(["Fulltext"], axis=1)
meta['type'] = meta.apply(lambda row: 'new' if any(x in row["DOI"] for x in ["2013", "2012"]) else 'old', axis=1)
meta.to_json("meta.json", orient="index")

meta_fulltext = raw["Fulltext"]
meta_fulltext.to_json("fulltext.json", orient="index")

# EMBEDDINGS
stop_words = stopwords.words('english')

abstracts = [nlp(text) for text in raw["Abstract"].tolist()]
abstract_lemma = [lemmatization(ab, [stopwords]) for ab in abstracts]

docs = [nlp(text) for text in raw["Fulltext"].tolist()]
full_lemma = [lemmatization(doc, stop_words) for doc in docs]

abstract_texts = [" ".join(text) for text in abstract_lemma]
abstract_tfidf = TfidfVectorizer().fit(abstract_texts)
abstract = abstract_tfidf.transform(abstract_texts)

fulltext_texts = [" ".join(text) for text in full_lemma]
fulltext_tfidf = TfidfVectorizer().fit(fulltext_texts)
fulltext = abstract_tfidf.transform(fulltext_texts)

abstract_nmf = NMF(10)
abstract_vecs = abstract_nmf.fit_transform(abstract)

full_nmf = NMF(10)
full_vecs = full_nmf.fit_transform(fulltext)

abstract_vecs = np.asarray(abstract_vecs, dtype=np.object)
full_vecs = np.asarray(full_vecs, dtype=np.object)

emb_abstract = pd.DataFrame(abstract_vecs)
emb_abstract.to_json("nmf_abstract.json", orient="index")

emb_full = pd.DataFrame(full_vecs)
emb_full.to_json("nmf_full.json", orient="index")



# CLASSIFICATION
# train/test split for classification
test = raw[raw["DOI"].str.contains("2013|2012")]  # len = 197
train = raw.drop(test.index)  # len = 1280

raw

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in train["Clusters"].tolist()])

y_train = train["Clusters"].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values

# prepare for visualization

# Analysis
# Similarity
metric = "jaccard"  # cosine, jaccard, emd, cm
distance = embedding.similarity_matrix(y_train, metric, as_distance=True)

# vis.simMatrix(distance)
vis.scatter(vecs, labels)
vis.scatter_tsne(vecs, labels, 0.1)
