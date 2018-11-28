from importlib import reload

import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from nltk.corpus import reuters, brown
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
import gensim.corpora as corpora
from nltk.corpus import stopwords

# Add general functions to the project
from os import path
import sys
sys.path.append(path.abspath('../methods'))

nlp = spacy.load('en_core_web_md', disable=['ner'])

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_top_words(model, feature_names, n_top_words):
    out = []
    for topic_idx, topic in enumerate(model.components_):
        topics = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        out.extend(topics)

    return set(out)

def preprocessing(text, stopwords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in text.sents:
        texts_out.extend([token.lemma_ for token in sent if
                          token.pos_ in allowed_postags and
                          token.lemma_ not in stop_words and
                          not token.like_num and
                          not token.like_url and
                          not token.like_email and
                          not token.lemma_ == "-PRON-" and
                          token.is_alpha and
                          len(token.lemma_) > 1])
    return texts_out

# DATA Loading
raw = np.load("../datasets/full.pkl")

# Remove Unclear from groups/clusters
# raw["Clusters"] = data.apply(
#     lambda row: row["Clusters"]
#     .replace(";Unclear", "")
#     .replace(";;", ";"), axis=1
#     )

# train/test split
# test = raw[raw["DOI"].str.contains("2013|2012")]  # len = 197
# train = raw.drop(test.index)  # len = 1280
label_column = ["Clusters"]

# data
# y
y = np.array(raw["Title"])


# preprocessing
stop_words = stopwords.words('english')

# test = raw.iloc[:100]

docs = [nlp(text) for text in raw["Fulltext"].tolist()]
clean = [preprocessing(doc, stop_words) for doc in docs]


# x
abstract_tfidf = TfidfVectorizer(stop_words=stop_words).fit(raw["Abstract"])
abstract_train = abstract_tfidf.transform(raw["Abstract"])

full_text_tfidf = TfidfVectorizer(stop_words=stop_words).fit(raw["Fulltext"])
fulltext_train = full_text_tfidf.transform(raw["Fulltext"])

id2word = corpora.Dictionary(clean)
corpus = [id2word.doc2bow(text) for text in clean]


# params
n_dim = [4,6,8,10,12,15,20,25,30]
algorithms = ["lda", "nmf"]
datasets = ["abstract", "fulltext"]

n_runs_per_setting = 10
topic_consistency_thresholds = [5, 10, 20]

# Create all parameter permutations
protocol = pd.DataFrame(columns=["Algorithm", "Dataset", "Dimensions", "NumberRuns", "Doc", "JaccTop5Median", "JaccTop10Median", "JaccTop20Median"])
runs = np.stack(np.meshgrid(algorithms, datasets, n_dim), -1).reshape(-1, 3)

# Compute
for run in runs:
    ldas = []
    vecs = []
    similarities = []
    row = []

    # Set variables
    if run[1] == "abstract":
        used = abstract_train
        tfidf = abstract_tfidf
    if run[1] == "fulltext":
        used = fulltext_train
        tfidf = full_text_tfidf

    # Fill protocol
    row.extend(run)
    row.append(n_runs_per_setting)
    # Compute models
    for iteration in range(0, n_runs_per_setting):
        if run[0] == "lda":
            lda = LatentDirichletAllocation(int(run[2]), learning_method="batch")
            vec = lda.fit_transform(used)
        if run[0] == "nmf":
            lda = NMF(int(run[2]))
            vec = lda.fit_transform(used)

        similarity = cosine_similarity(vec)

        ldas.append(lda)
        vecs.append(vec)
        similarities.append(similarity)

    # Document neighborhood consistency
    doc_sim = []
    for i in range(0, len(similarities)):
        for j in range(i+1, len(similarities)):
            doc_sim.append(abs(similarities[i] - similarities[j]).sum())

    row.append(np.median(doc_sim))
    # print(y[similarity1[156].argsort()[-10:][::-1]])
    # print(y[similarity2[156].argsort()[-10:][::-1]])

    # Topic consistency
    for thres in topic_consistency_thresholds:
        top_sim = []

        for i in range(0, len(similarities)):
            topics1 = get_top_words(ldas[i], tfidf.get_feature_names(), thres)
            for j in range(i+1, len(similarities)):
                topics2 = get_top_words(ldas[j], tfidf.get_feature_names(), thres)

                top_sim.append(len(topics1.intersection(topics2))/len(topics1.union(topics2)))

        row.append(np.median(top_sim))
        # print_top_words(lda1, tfidf.get_feature_names(), 10)
        # print_top_words(lda2, tfidf.get_feature_names(), 10)

    protocol = protocol.append(pd.DataFrame([row], columns=protocol.columns))

# LDA Fulltext dim 6 is best (drop on 8)
# LDA Abstract dim 4 is best (far worse then the NMF 4 or 6)
# NMF Fulltext dim 10 is best (drop on 12)
# NMF Abstract dim 6 is best (drop on 8)
protocol.to_csv("runs.csv")

# Filter dimensions
# np.median(vecs1, axis=0)
# np.std(vecs1, axis=0)

print_top_words(ldas[1], tfidf.get_feature_names(), 10)
print_top_words(ldas[0], tfidf.get_feature_names(), 10)

# Create embeddings
dois = raw["DOI"]

# LDA_NMF data
# lda = LatentDirichletAllocation(6, learning_method="batch")
# vecs_lda = lda.fit_transform(fulltext_train)
#
# vecs_lda = np.asarray(vecs_lda, dtype=np.object)
# vecs_lda = np.insert(vecs_lda, 0, dois, axis=1)
#
# emb1 = pd.DataFrame(vecs_lda)
# emb1.to_csv("lda_nmf_1.csv", header=False, index=False)
#
# nmf = NMF(10)
# vecs_nmf = nmf.fit_transform(fulltext_train)
#
# vecs_nmf = np.asarray(vecs_nmf, dtype=np.object)
# vecs_nmf = np.insert(vecs_nmf, 0, dois, axis=1)
#
# emb2 = pd.DataFrame(vecs_nmf)
# emb2.to_csv("lda_nmf_2.csv", header=False, index=False)

# abstract_fulltext data
lda = NMF(6)
vecs_lda = lda.fit_transform(abstract_train)

vecs_lda = np.asarray(vecs_lda, dtype=np.object)
vecs_lda = np.insert(vecs_lda, 0, dois, axis=1)

emb1 = pd.DataFrame(vecs_lda)
emb1.to_csv("full_abstract_1.csv", header=False, index=False)

nmf = NMF(10)
vecs_nmf = nmf.fit_transform(fulltext_train)

vecs_nmf = np.asarray(vecs_nmf, dtype=np.object)
vecs_nmf = np.insert(vecs_nmf, 0, dois, axis=1)

emb2 = pd.DataFrame(vecs_nmf)
emb2.to_csv("full_abstract_2.csv", header=False, index=False)
