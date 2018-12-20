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
fulltext = fulltext_tfidf.transform(fulltext_texts)

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

# top topic words
abstract_top_words = get_top_words(abstract_nmf, abstract_tfidf, 100)
pd.DataFrame(abstract_top_words).to_json("top_words_abstract.json", orient="values")

full_top_words = get_top_words(full_nmf, fulltext_tfidf, 100)
pd.DataFrame(full_top_words).to_json("top_words_full.json", orient="values")

# dim reduction
abstract_svd = TruncatedSVD(2).fit_transform(abstract_vecs)
abstract_pca = PCA(2).fit_transform(abstract_vecs)
abstract_tsne = TSNE(2).fit_transform(abstract_vecs)
abstract_mds = MDS(2).fit_transform(abstract_vecs)

abstract_projections = pd.DataFrame({
    'svd': pd.DataFrame(abstract_svd).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'pca': pd.DataFrame(abstract_pca).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'mds': pd.DataFrame(abstract_mds).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'tsne': pd.DataFrame(abstract_tsne).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    })

abstract_projections.to_json("projections_abstract.json", orient="index")

full_svd = TruncatedSVD(2).fit_transform(full_vecs)
full_pca = PCA(2).fit_transform(full_vecs)
full_tsne = TSNE(2).fit_transform(full_vecs)
full_mds = MDS(2).fit_transform(full_vecs)

full_projections = pd.DataFrame({
    'svd': pd.DataFrame(full_svd).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'pca': pd.DataFrame(full_pca).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'mds': pd.DataFrame(full_mds).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    'tsne': pd.DataFrame(full_tsne).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
    })

full_projections.to_json("projections_full.json", orient="index")

# CLASSIFICATION
# train/test split for classification
test = meta[meta["type"] == "new"]  # len = 197
train = meta.drop(test.index)  # len = 1280

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
