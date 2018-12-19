from importlib import reload

import os
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
import gensim
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

def get_top_words_lda(model, n_top_words):
    out = []
    for topic_idx, topic in enumerate(model.get_topics()):
        topics = model.get_topic_terms(2, topn=n_top_words)
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

def lemmatization(text, stopwords):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in text.sents:
        texts_out.extend([token.lemma_ for token in sent if
                          token.lemma_ not in stop_words and
                          not token.lemma_ == "-PRON-"])
    return texts_out

def raw_stopwords(text, stopwords):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in text.sents:
        texts_out.extend([token.text for token in sent if
                          token.lemma_ not in stop_words])
    return texts_out

# DATA Loading
raw = np.load("../datasets/full.pkl")

# preprocessing
stop_words = stopwords.words('english')

abstracts = [nlp(text) for text in raw["Abstract"].tolist()]
abstract_lemma = [lemmatization(ab, [stopwords]) for ab in abstracts]

docs = [nlp(text) for text in raw["Fulltext"].tolist()]
full_raw = [raw_stopwords(doc, stop_words) for doc in docs]
full_lemma = [lemmatization(doc, stop_words) for doc in docs]
full_clean = [preprocessing(doc, stop_words) for doc in docs]

# id2word = corpora.Dictionary(full_clean)
# corpus = [id2word.doc2bow(text) for text in full_clean]
#
# model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                         id2word=id2word,
#                                         num_topics=5,
#                                         update_every=1,
#                                         chunksize=100,
#                                         passes=10,
#                                         alpha='auto',
#                                         per_word_topics=False)
#
# vec = [gensim.matutils.sparse2full(spa, 5) for spa in model[corpus]]

# params
n_dim = [4,8,12,18,24,30,50]
algorithms = ["lda", "nmf"]
datasets = ["lemma_abstract", "lemma_fulltext", "clean_fulltext", "full_raw"]

n_runs_per_setting = 5
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
    if run[1] == "full_raw":
        texts = [" ".join(text) for text in full_raw]
        tfidf = TfidfVectorizer().fit(texts)
        used = tfidf.transform(texts)

        id2word = corpora.Dictionary(full_raw)
        corpus = [id2word.doc2bow(text) for text in full_raw]
    if run[1] == "lemma_abstract":
        texts = [" ".join(text) for text in abstract_lemma]
        tfidf = TfidfVectorizer().fit(texts)
        used = tfidf.transform(texts)

        id2word = corpora.Dictionary(abstract_lemma)
        corpus = [id2word.doc2bow(text) for text in abstract_lemma]
    if run[1] == "lemma_fulltext":
        texts = [" ".join(text) for text in full_lemma]
        tfidf = TfidfVectorizer().fit(texts)
        used = tfidf.transform(texts)

        id2word = corpora.Dictionary(full_lemma)
        corpus = [id2word.doc2bow(text) for text in full_lemma]
    if run[1] == "clean_fulltext":
        texts = [" ".join(text) for text in full_clean]
        tfidf = TfidfVectorizer().fit(texts)
        used = tfidf.transform(texts)

        id2word = corpora.Dictionary(full_clean)
        corpus = [id2word.doc2bow(text) for text in full_clean]

    # Fill protocol
    row.extend(run)
    row.append(n_runs_per_setting)
    # Compute models
    for iteration in range(0, n_runs_per_setting):
        if run[0] == "lda":
            model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=int(run[2]),
                                               workers=3)
            # model = gensim.models.ldamodel.LdaModel(corpus=corpus,
            #                                         id2word=id2word,
            #                                         num_topics=int(run[2]),
            #                                         update_every=1,
            #                                         chunksize=100,
            #                                         passes=10,
            #                                         alpha='auto',
            #                                         per_word_topics=False)
            vec = [gensim.matutils.sparse2full(spa, int(run[2])) for spa in model[corpus]]
        if run[0] == "nmf":
            model = NMF(int(run[2]))
            vec = model.fit_transform(used)

        similarity = cosine_similarity(vec)

        ldas.append(model)
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
            if run[0] == "nmf":
                topics1 = get_top_words(ldas[i], tfidf.get_feature_names(), thres)
            if run[0] == "lda":
                topics1 = get_top_words_lda(ldas[i], thres)
                pass
            for j in range(i+1, len(similarities)):
                if run[0] == "nmf":
                    topics2 = get_top_words(ldas[i], tfidf.get_feature_names(), thres)
                if run[0] == "lda":
                    topics2 = get_top_words_lda(ldas[i], thres)

                top_sim.append(len(topics1.intersection(topics2))/len(topics1.union(topics2)))

        row.append(np.median(top_sim))
        # print_top_words(lda1, tfidf.get_feature_names(), 10)
        # print_top_words(lda2, tfidf.get_feature_names(), 10)

    protocol = protocol.append(pd.DataFrame([row], columns=protocol.columns))

# Results using sklearn LDA
# LDA Fulltext dim 6 is best (drop on 8)
# LDA Abstract dim 4 is best (far worse then the NMF 4 or 6)
# NMF Fulltext dim 10 is best (drop on 12)
# NMF Abstract dim 6 is best (drop on 8)
# Results using gensim LDA + preprosessing with spacy

protocol.to_csv("runs.csv")

# NMF
texts = [" ".join(text) for text in full_lemma]
tfidf = TfidfVectorizer().fit(texts)
used = tfidf.transform(texts)
model = NMF(int(10))
vec = model.fit_transform(used)

# LDA
id2word = corpora.Dictionary(full_clean)
corpus = [id2word.doc2bow(text) for text in full_clean]
model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=20,
                                        workers=3)

vec = [gensim.matutils.sparse2full(spa, 20) for spa in model[corpus]]

save_vec(vec, "lda_clean_full_20.csv")

similarity = cosine_similarity(vec)
print_top_words(model, tfidf.get_feature_names(), 10)

# Filter dimensions
np.median(vec, axis=0)
# np.std(vecs1, axis=0)

# print_top_words(ldas[1], tfidf.get_feature_names(), 10)
# print_top_words(ldas[0], tfidf.get_feature_names(), 10)

# Create embeddings
dois = raw["DOI"]


def save_vec(vec, name):
    vec = np.asarray(vec, dtype=np.object)
    vec = np.insert(vec, 0, dois, axis=1)

    emb1 = pd.DataFrame(vec)
    emb1.to_csv(name, header=False, index=False)

# LDA_NMF data
lda = LatentDirichletAllocation(6, learning_method="batch")
vecs_lda = lda.fit_transform(fulltext_train)

vecs_lda = np.asarray(vecs_lda, dtype=np.object)
vecs_lda = np.insert(vecs_lda, 0, dois, axis=1)

emb1 = pd.DataFrame(vecs_lda)
emb1.to_csv("lda_nmf_1.csv", header=False, index=False)

nmf = NMF(10)
vecs_nmf = nmf.fit_transform(fulltext_train)

vecs_nmf = np.asarray(vecs_nmf, dtype=np.object)
vecs_nmf = np.insert(vecs_nmf, 0, dois, axis=1)

emb2 = pd.DataFrame(vecs_nmf)
emb2.to_csv("lda_nmf_2.csv", header=False, index=False)

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
