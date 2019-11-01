from importlib import reload

import re
import pandas as pd
import numpy as np
# from textblob import TextBlob
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import spacy
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import RMDL

nlp = spacy.load('en', disable=['ner'])
stop_words = stopwords.words('english')

# Add general functions to the project
from os import path
import sys
sys.path.append(path.abspath('../methods'))

# import embedding
# import vis


# Helper functions
def lemmatization(text, stopwords):
    """https://spacy.io/api/annotation"""
    texts_out = []
    regexr = text.replace(";", " ")
    for sent in nlp(regexr).sents:
        temp = " ".join((token.lemma_ for token in sent if
                                   token.lemma_ not in stop_words and
                                   len(token.lemma_) > 1 and
                                   not token.lemma_ == "-PRON-"))
        texts_out.append(temp)
    return " ".join(texts_out)


def preprocess(text, stopwords):
    """https://spacy.io/api/annotation"""
    texts_out = []
    regexr = re.sub(r"[^a-zA-Z0-9 _]*", "", text.replace(";", " "))
    for doc in nlp(regexr).sents:
        temp = " ".join((token.lemma_ for token in doc if
                                   not token.like_num and
                                   not token.like_url and
                                   not token.like_email and
                                   token.lemma_ not in stop_words and
                                   len(token.lemma_) > 1 and
                                   not token.lemma_ == "-PRON-"))
        texts_out.append(temp)
    return " ".join(texts_out)


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
# docs = (nlp(text) for text in raw["Fulltext"].tolist())
# full_lemma = (lemmatization(doc, stop_words) for doc in docs)
# fulltext_texts = [" ".join(text) for text in full_lemma]
# pd.DataFrame(fulltext_texts).to_json("fulltext_lemma.json", orient="index")


# meta = pd.read_json("../datasets/meta.json", orient="index").sort_index()
# full = pd.read_json("../datasets/fulltext.json", typ='series').sort_index()
#

# fulltexts = pd.read_json("datasets/fulltext_lemma.json", orient="index").sort_index()
# meta = pd.read_json("datasets/meta.json", orient="index").sort_index()
# keywords = meta["Keywords"]
# # Remove leading and trailing ;
# meta['Clusters'] = meta['Clusters'].apply(lambda x: x.strip(';'))



# # CLASSIFICATION
# # train/test split for classification
# test_index = meta[meta["type"] == "new"].index  # len = 197
# train_index = meta.drop(test_index).index  # len = 1280

# # y
# enc = MultiLabelBinarizer()
# enc.fit([cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()])

# y_train = np.vstack(meta.iloc[train_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)
# # y_train = meta.iloc[train_index].apply(lambda row: [row["Clusters"].split(";")][0], axis=1).values
# y_test = np.vstack(meta.iloc[test_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)

# # x
# #fulltext
# fulltext_tfidf = TfidfVectorizer(max_df=0.5).fit(fulltexts[0].tolist())
# fulltext_vecs = fulltext_tfidf.transform(fulltexts[0].tolist())

# x_train = fulltext_vecs[train_index]
# x_test = fulltext_vecs[test_index]

# nmf = NMF(10)
# vecs = nmf.fit_transform(fulltext_vecs)
# vecs = np.asarray(vecs, dtype=np.object)

# x_train_nmf = vecs[train_index]
# x_test_nmf = vecs[test_index]

# svd = TruncatedSVD(300).fit_transform(fulltext_vecs)
# x_train_svd = svd[train_index]
# x_test_svd = svd[test_index]


# # keywords multiword
# # multi = [lemmatization(key.replace(" ", "_"), stopwords) for key in keywords.tolist()]
# multi = [preprocess(key.replace(" ", "_"), stopwords) for key in keywords.tolist()]
# # multi = [key.replace(" ", "_") for key in keywords.tolist()]
# multi_tfidf = TfidfVectorizer().fit(multi)
# multi_vecs = multi_tfidf.transform(multi)

# x_train_multi = multi_vecs[train_index]
# x_test_multi = multi_vecs[test_index]

# # keywords single word
# single = [preprocess(key, stopwords) for key in keywords.tolist()]
# single_tfidf = TfidfVectorizer().fit(single)
# single_vecs = single_tfidf.transform(single)

# x_train_single = single_vecs[train_index]
# x_test_single = single_vecs[test_index]

# # concept vectors
# concept = 4

# # get all topic vectors and check internal consistency
# concept_vectors = x_train_nmf[np.nonzero(y_test.T[:,concept])[0]]
# cosine_similarity(concept_vectors)

# # check topic consistency
# concept = np.mean(concept_vectors, axis=0)
# cosine_similarity(np.vstack((concept_vectors, concept)))[-1,:]

# # compare two different concepts
# t1 = 20
# t2 = 30

# v3 = x_train_svd[np.nonzero(y_test.T[:,t1])[0]]
# print(len(v3))
# c3 = np.mean(v3, axis=0)

# v4 = x_train_svd[np.nonzero(y_test.T[:,t2])[0]]
# print(len(v4))
# c4 = np.mean(v4, axis=0)

# cosine_similarity(np.vstack((v3, v4, c3, c4)))[-1,:]  # c4 sim
# cosine_similarity(np.vstack((v3, v4, c3, c4)))[-2,:]  # c3 sim

# # build all concept concept_vectors
# used_train = x_train_nmf
# used_test = x_test_nmf

# nmf = NMF(20)
# vecs = nmf.fit_transform(fulltext_vecs)
# vecs = np.asarray(vecs, dtype=np.object)

# used_train = vecs[train_index]
# used_test = vecs[test_index]

# vector_sets = [used_train[np.nonzero(concept)[0]] for concept in y_train.T]
# concept_vectors = [np.mean(vecs, axis=0) for vecs in vector_sets]

# # classify based on concept vectors
# sim = cosine_similarity(np.vstack((used_test, concept_vectors)))
# prediction_vecs = sim[:-179, len(used_test):]
# prediction = np.array([vec > vec.mean() for vec in prediction_vecs.T])
# prediction = prediction_vecs > prediction_vecs.mean()

# sim = cosine_similarity(np.vstack((used_train, concept_vectors)))
# train_test_vecs = sim[:-179, len(used_train):]
# train_test = train_test_vecs > train_test_vecs.mean()

# print(classification_report(y_train, train_test))
# print(classification_report(y_test, prediction))

# reduced_meta = meta[meta["type"]=="new"]
# reduced_meta["Vector"] = [v.tolist() for v in used_test]
# # reduced_full = pd.Series(np.array(full)[meta["type"]=="new"])

# reduced_meta.to_json("reduced_meta.json", orient="index")
# # reduced_full.to_json("reduced_fulltext.json", orient="index")

# classes = pd.DataFrame(enc.classes_, columns=["Cluster"])
# classes["Vector"] = [v.tolist() for v in concept_vectors]

# classes.to_json("classes.json", orient="index")

# # classifiers
# # onevsrest = OneVsRestClassifier(SVC()).fit(x_train, y_train)
# # onevsrest.score(x_test, y_test)
# # tree = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
# # extra = ExtraTreesClassifier(n_estimators=200).fit(x_train, y_train)
# ovr_ada = MultiOutputClassifier(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300)).fit(x_train_single, y_train)
# ovr_ada.score(x_test_single, y_test)
# ovr_tree = MultiOutputClassifier(DecisionTreeClassifier(criterion="entropy")).fit(x_train_single, y_train)
# ovr_tree.score(x_test_single, y_test)
# chain_tree = ClassifierChain(DecisionTreeClassifier(criterion="entropy")).fit(x_train_single, y_train)
# chain_tree.score(x_test_single, y_test)
# # chain_extra = ClassifierChain(ExtraTreesClassifier(n_estimators=100)).fit(x_train, y_train)
# # mcp = MLPClassifier(max_iter=500).fit(x_train, y_train)
# # mcp2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500).fit(x_train, y_train)
# mnb = MultiOutputClassifier(MultinomialNB()).fit(x_train_single, y_train)
# mnb.score(x_test_single, y_test)
# lgd = MultiOutputClassifier(SGDClassifier()).fit(x_train_single, y_train)
# lgd.score(x_test_single, y_test)
# log = MultiOutputClassifier(LogisticRegression()).fit(x_train_single, y_train)
# log.score(x_test_single, y_test)

# # https://github.com/kk7nc/RMDL
# train_single = np.array(single)[train_index]
# test_single = np.array(single)[test_index]

# RMDL.RMDL_Text.Text_Classification(train_single, y_train, test_single, y_test,
#             #  batch_size=batch_size,
#             #  sparse_categorical=True,
#             #  random_deep=Random_Deep,
#              epochs=[20, 50, 50]) ## DNN--RNN-CNN

# print_cls = ovr_tree

# print(classification_report(y_train, print_cls.predict(x_train_single), target_names=classes["Cluster"]))
# print(classification_report(y_test, print_cls.predict(x_test_single), target_names=classes["Cluster"]))

# # custom classifier
# clusters = [cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()]
# keywords = [r.split() for r in multi]

# pairs = zip(keywords[:100], clusters[:100])
# mapping = {}
# for pair in pairs:
#     for keyword in pair[0]:
#         if keyword in mapping:
#             temp = mapping[keyword]
#             temp.union(pair[1])
#         else:
#             mapping[keyword] = set(pair[1])

# out_keywords = pd.DataFrame(pd.Series(mapping))[0].apply(lambda x: list(x))
# out_keywords.to_json("keyword_mapping.json", orient="index")

# # save the jsons
# pd.DataFrame(enc.classes_).to_json("classes.json", orient="values")
# pd.DataFrame(y_train).to_json("old_labels.json", orient="values")
# pd.DataFrame(y_test).to_json("ground_truth_labels.json", orient="values")
# pd.DataFrame(ovr_tree.predict(x_test)).to_json("new_labels_1.json", orient="values")
# pd.DataFrame(chain_tree.predict(x_test)).to_json("new_labels_2.json", orient="values")
# pd.DataFrame(chain_tree.predict(x_test)).to_json("new_labels_3.json", orient="values")

# # dim reduction
# enc_key = MultiLabelBinarizer()
# vecs = enc_key.fit_transform([m.split(" ") for m in single])

# svd = TruncatedSVD(2).fit_transform(vecs)
# pca = PCA(2).fit_transform(vecs)
# tsne = TSNE(2).fit_transform(vecs)
# mds = MDS(2).fit_transform(vecs)

# projections = pd.DataFrame({
#     'svd': pd.DataFrame(svd).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
#     'pca': pd.DataFrame(pca).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
#     'mds': pd.DataFrame(mds).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
#     'tsne': pd.DataFrame(tsne).apply(lambda row: str(row[0])+ ","+str(row[1]), axis = 1),
#     })

# projections.to_json("projections_keywords_single.json", orient="index")
# Ground truth
# enc = MultiLabelBinarizer()
# enc.fit([cluster.split(";") for cluster in meta["Clusters"].tolist()])

# y = np.array([enc.transform([x.split(";")])[0] for x in meta["Clusters"]])

# pd.DataFrame(y).to_json("all_labels.json", orient="values")

# New data preparation
meta = pd.read_json("datasets/new_data.json", orient="index").sort_index().dropna().reset_index()

abstracts = list(meta["Abstract"])
keywords = ["" if key == None else key for key in list(list(meta["Keywords"]))]

single = [preprocess(key, stopwords) for key in keywords]
single_tfidf = TfidfVectorizer().fit(single)
single_vecs = single_tfidf.transform(single)

abstract_tfidf = TfidfVectorizer(max_df=0.5).fit(abstracts)
abstract_vecs = abstract_tfidf.transform(abstracts)

abstract_svd = TruncatedSVD(20).fit_transform(abstract_vecs)
keyword_svd = TruncatedSVD(20).fit_transform(single_vecs)

meta["Keyword_Vector"] = ""
meta['Keyword_Vector'] = meta['Keyword_Vector'].astype(object)

meta["Abstract_Vector"] = ""
meta['Abstract_Vector'] = meta['Abstract_Vector'].astype(object)

for i in meta.index:
    meta.at[i, "Keyword_Vector"] = list(pd.Series(keyword_svd[i]))
    meta.at[i, "Abstract_Vector"] = list(pd.Series(abstract_svd[i]))

meta.to_json("new_data.json", orient="index")

# Automatic performance measurement

# Parameters:
# Data - fulltexts, abstracts, keywords_single, keywords_multi
# Embedding -

# data
meta = pd.read_json("datasets/meta.json", orient="index").sort_index()

# Remove leading and trailing ;
meta['Clusters'] = meta['Clusters'].apply(lambda x: x.strip(';'))

# train/test split for classification
test_index = meta[meta["type"] == "new"].index  # len = 197
train_index = meta.drop(test_index).index  # len = 1280

# fulltexts
fulltexts = pd.read_json("datasets/fulltext_lemma.json", orient="index").sort_index()

# abstracts
abstracts = list(meta["Abstract"])

# keywords
keywords = meta["Keywords"]
multi = [preprocess(key.replace(" ", "_"), stopwords) for key in keywords.tolist()]
single = [preprocess(key, stopwords) for key in keywords.tolist()]

# embedding
# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()])

y_train = np.vstack(meta.iloc[train_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)
y_test = np.vstack(meta.iloc[test_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)

classes = pd.DataFrame(enc.classes_, columns=["Cluster"])

# x
#fulltext
fulltext_tfidf = TfidfVectorizer(max_df=0.5).fit(fulltexts[0].tolist())
fulltext_vecs = fulltext_tfidf.transform(fulltexts[0].tolist())

x_train_full = fulltext_vecs[train_index]
x_test_full = fulltext_vecs[test_index]

abstract_tfidf = TfidfVectorizer(max_df=0.5).fit(abstracts)
abstract_vecs = fulltext_tfidf.transform(abstracts)

x_train_abstract = abstract_vecs[train_index]
x_test_abstract = abstract_vecs[test_index]

nmf = NMF(10)
vecs = nmf.fit_transform(fulltext_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_nmf_10 = vecs[train_index]
x_test_nmf_10 = vecs[test_index]

nmf = NMF(15)
vecs = nmf.fit_transform(fulltext_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_nmf_15 = vecs[train_index]
x_test_nmf_15 = vecs[test_index]

nmf = NMF(20)
vecs = nmf.fit_transform(fulltext_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_nmf_20 = vecs[train_index]
x_test_nmf_20 = vecs[test_index]

nmf = NMF(10)
vecs = nmf.fit_transform(abstract_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_abstract_nmf_10 = vecs[train_index]
x_test_abstract_nmf_10 = vecs[test_index]

nmf = NMF(15)
vecs = nmf.fit_transform(abstract_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_abstract_nmf_15 = vecs[train_index]
x_test_abstract_nmf_15 = vecs[test_index]

nmf = NMF(20)
vecs = nmf.fit_transform(abstract_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train_abstract_nmf_20 = vecs[train_index]
x_test_abstract_nmf_20 = vecs[test_index]

svd = TruncatedSVD(50).fit_transform(fulltext_vecs)
x_train_svd_50 = svd[train_index]
x_test_svd_50 = svd[test_index]

svd = TruncatedSVD(20).fit_transform(fulltext_vecs)
x_train_svd_20= svd[train_index]
x_test_svd_20= svd[test_index]

svd = TruncatedSVD(100).fit_transform(fulltext_vecs)
x_train_svd_100 = svd[train_index]
x_test_svd_100 = svd[test_index]

svd = TruncatedSVD(50).fit_transform(abstract_vecs)
x_train_abstract_svd_50 = svd[train_index]
x_test_abstract_svd_50 = svd[test_index]

svd = TruncatedSVD(20).fit_transform(abstract_vecs)
x_train_abstract_svd_20= svd[train_index]
x_test_abstract_svd_20= svd[test_index]

svd = TruncatedSVD(100).fit_transform(abstract_vecs)
x_train_abstract_svd_100 = svd[train_index]
x_test_abstract_svd_100 = svd[test_index]
# abstract

# keywords multi
multi_tfidf = TfidfVectorizer().fit(multi)
multi_vecs = multi_tfidf.transform(multi)

x_train_multi = multi_vecs[train_index]
x_test_multi = multi_vecs[test_index]

# keywords single
single_tfidf = TfidfVectorizer().fit(single)
single_vecs = single_tfidf.transform(single)

x_train_single = single_vecs[train_index]
x_test_single = single_vecs[test_index]

datasets = [
    # ["fulltext tfidf",x_train_full, x_test_full],
    ["fulltext nmf 10",x_train_nmf_10, x_test_nmf_10],
    ["fulltext nmf 15",x_train_nmf_15, x_test_nmf_15],
    ["fulltext nmf 20",x_train_nmf_20, x_test_nmf_20],
    ["fulltext tfidf svd 20",x_train_svd_20, x_test_svd_20],
    ["fulltext tfidf svd 50",x_train_svd_50, x_test_svd_50],
    ["fulltext tfidf svd 100",x_train_svd_100, x_test_svd_100],
    # ["abstract tfidf",x_train_abstract, x_test_abstract],
    ["abstract nmf 10",x_train_abstract_nmf_10, x_test_abstract_nmf_10],
    ["abstract nmf 15",x_train_abstract_nmf_15, x_test_abstract_nmf_15],
    ["abstract nmf 20",x_train_abstract_nmf_20, x_test_abstract_nmf_20],
    ["abstract tfidf svd 20",x_train_abstract_svd_20, x_test_abstract_svd_20],
    ["abstract tfidf svd 50",x_train_abstract_svd_50, x_test_abstract_svd_50],
    ["abstract tfidf svd 100",x_train_abstract_svd_100,x_test_abstract_svd_100],
    ["keywords multi-word",x_train_multi, x_test_multi],
    ["keywords single-word",x_train_single, x_test_single]
    ]

# classification
out = pd.DataFrame(columns=["Dataset", "Method", "Params", "Accuracy"])

classifications = [
    ["DecisionTree", DecisionTreeClassifier, [
        {"criterion": "gini", "min_samples_leaf": 5},
        # {"criterion": "gini", "min_samples_leaf": 10},
        {"criterion": "entropy", "min_samples_leaf": 5},
        # {"criterion": "entropy", "min_samples_leaf": 10},
        ]],
    # ["ExtraTree", ExtraTreeClassifier, [
    #     {"criterion": "gini", "min_samples_leaf": 5},
    #     {"criterion": "gini", "min_samples_leaf": 10},
    #     {"criterion": "entropy", "min_samples_leaf": 5},
    #     {"criterion": "entropy", "min_samples_leaf": 10},
    #     ]],
    ["Extra Tree Ensemble", ExtraTreesClassifier, [
        # {"n_estimators": 100, "min_samples_leaf": 5},
        {"n_estimators": 200, "min_samples_leaf": 5},
        # {"n_estimators": 100, "min_samples_leaf": 10},
        {"n_estimators": 200, "min_samples_leaf": 10},
        # {"n_estimators": 100, "min_samples_leaf": 5},
        # {"n_estimators": 200, "min_samples_leaf": 5},
        # {"n_estimators": 100, "min_samples_leaf": 10},
        # {"n_estimators": 200, "min_samples_leaf": 10},
        ]],
    ["kneighbors", KNeighborsClassifier, [
        {"n_neighbors": 5},
        {"n_neighbors": 10},
        {"n_neighbors": 15},
        ]],
    ["Random Forest", RandomForestClassifier, [
        # {"n_estimators": 100, "criterion": "gini", "min_samples_leaf": 5},
        # {"n_estimators": 200, "criterion": "gini", "min_samples_leaf": 5},
        {"n_estimators": 300, "criterion": "gini", "min_samples_leaf": 5},
        # {"n_estimators": 100, "criterion": "entropy", "min_samples_leaf": 5},
        # {"n_estimators": 200, "criterion": "entropy", "min_samples_leaf": 5},
        {"n_estimators": 300, "criterion": "entropy", "min_samples_leaf": 5},
        # {"n_estimators": 100, "criterion": "gini", "min_samples_leaf": 10},
        # {"n_estimators": 200, "criterion": "gini", "min_samples_leaf": 10},
        {"n_estimators": 300, "criterion": "gini", "min_samples_leaf": 10},
        # {"n_estimators": 100, "criterion": "entropy", "min_samples_leaf": 10},
        # {"n_estimators": 200, "criterion": "entropy", "min_samples_leaf": 10},
        {"n_estimators": 300, "criterion": "entropy", "min_samples_leaf": 10},
        ]],
    ["MLP", MLPClassifier, [
        # {"hidden_layer_sizes": 50, "activation": "relu", "learning_rate": "constant"},
        # {"hidden_layer_sizes": 50, "activation": "tanh", "learning_rate": "constant"},
        # {"hidden_layer_sizes": 50, "activation": "relu", "learning_rate": "invscaling"},
        # {"hidden_layer_sizes": 50, "activation": "tanh", "learning_rate": "invscaling"},
        # {"hidden_layer_sizes": 50, "activation": "relu", "learning_rate": "adaptive"},
        # {"hidden_layer_sizes": 50, "activation": "tanh", "learning_rate": "adaptive"},
        # {"hidden_layer_sizes": 100, "activation": "relu", "learning_rate": "constant"},
        # {"hidden_layer_sizes": 100, "activation": "tanh", "learning_rate": "constant"},
        {"hidden_layer_sizes": 100, "activation": "relu", "learning_rate": "invscaling"},
        {"hidden_layer_sizes": 100, "activation": "tanh", "learning_rate": "invscaling"},
        {"hidden_layer_sizes": 100, "activation": "relu", "learning_rate": "adaptive"},
        {"hidden_layer_sizes": 100, "activation": "tanh", "learning_rate": "adaptive"},
        # {"hidden_layer_sizes": (100, 100), "activation": "relu", "learning_rate": "constant"},
        # {"hidden_layer_sizes": (100, 100), "activation": "tanh", "learning_rate": "constant"},
        # {"hidden_layer_sizes": (100, 100), "activation": "relu", "learning_rate": "invscaling"},
        # {"hidden_layer_sizes": (100, 100), "activation": "tanh", "learning_rate": "invscaling"},
        # {"hidden_layer_sizes": (100, 100), "activation": "relu", "learning_rate": "adaptive"},
        # {"hidden_layer_sizes": (100, 100), "activation": "tanh", "learning_rate": "adaptive"},
        # {"hidden_layer_sizes": (100, 50), "activation": "relu", "learning_rate": "constant"},
        # {"hidden_layer_sizes": (100, 50), "activation": "tanh", "learning_rate": "constant"},
        {"hidden_layer_sizes": (100, 50), "activation": "relu", "learning_rate": "invscaling"},
        {"hidden_layer_sizes": (100, 50), "activation": "tanh", "learning_rate": "invscaling"},
        {"hidden_layer_sizes": (100, 50), "activation": "relu", "learning_rate": "adaptive"},
        {"hidden_layer_sizes": (100, 50), "activation": "tanh", "learning_rate": "adaptive"},
        ]]
]
out = pd.DataFrame(columns=["Dataset", "Method", "Params", "Accuracy"])

for data_id, dataset in enumerate(datasets):
    name = dataset[0]
    train = dataset[1]
    test = dataset[2]

    for cls_id, classification in enumerate(classifications):
        clf_name = classification[0]
        clf_params = classification[2]

        if isinstance(clf_params, list):
            for param in clf_params:
                clf = classification[1](**param)
                clf.fit(train, y_train)

                clf_acc = clf.score(test, y_test)

                out = out.append(pd.DataFrame([[name,clf_name,str(param),clf_acc]], columns=["Dataset", "Method", "Params","Accuracy"]), ignore_index=True)
        else:
            params = clf_params
            clf = classification[1](**params)
            clf.fit(train, y_train)

            clf_acc = clf.score(test, y_test)

            out = out.append(pd.DataFrame([[name,clf_name,str(clf_params),clf_acc]], columns=["Dataset", "Method", "Params","Accuracy"]), ignore_index=True)

        print("Dataset: "+str(data_id+1)+"/"+str(len(datasets))+", Classification: "+str(cls_id+1)+"/"+str(len(classifications)))

out.to_csv("results.csv")



multiclass = [
    ["Gradient Boosting", GradientBoostingClassifier, [
        {"learning_rate": 0.1, "n_estimators": 100},
        {"learning_rate": 0.1, "n_estimators": 200},
        {"learning_rate": 0.1, "n_estimators": 300},
        {"learning_rate": 0.02, "n_estimators": 100},
        {"learning_rate": 0.02, "n_estimators": 200},
        {"learning_rate": 0.02, "n_estimators": 300},
        ]],
    ["Decision Tree", DecisionTreeClassifier, [
        {"criterion": "gini", "min_samples_leaf": 5},
        {"criterion": "gini", "min_samples_leaf": 10},
        {"criterion": "entropy", "min_samples_leaf": 5},
        {"criterion": "entropy", "min_samples_leaf": 10},
        ]],
    ["Multinomial NB", MultinomialNB, {}],
    ["Naive Bayer Gaussian", GaussianNB, {}],
    ["SGD", SGDClassifier, [
        {"loss": "hinge", "penalty": "l2"},
        {"loss": "log", "penalty": "l2"},
        {"loss": "perceptron", "penalty": "l2"},
        {"loss": "hinge", "penalty": "l1"},
        {"loss": "log", "penalty": "l1"},
        {"loss": "perceptron", "penalty": "l1"},
        {"loss": "hinge", "penalty": "elasticnet"},
        {"loss": "log", "penalty": "elasticnet"},
        {"loss": "perceptron", "penalty": "elasticnet"},
        ]],
    ["Logistic Regression", LogisticRegression, {"multi_class": "Multinomial"}],
    ["Linear Discriminant Analysis", LinearDiscriminantAnalysis, {}]
]
out = pd.DataFrame(columns=["Dataset", "Method", "Params", "Accuracy"])

for dataset in datasets:
    name = dataset[0]
    train = dataset[1]
    test = dataset[2]

    for classification in multiclass:
        clf_name = classification[0]
        clf_params = classification[2]

        if isinstance(clf_params, list):
            for param in clf_params:
                clf = MultiOutputClassifier(classification[1](**param))
                clf.fit(train, y_train)

                clf_acc = clf.score(test, y_test)

                out = out.append(pd.DataFrame([["Multioutput "+name,clf_name,str(param),clf_acc]], columns=["Dataset", "Method", "Params","Accuracy"]), ignore_index=True)
        else:
            params = clf_params
            clf = MultiOutputClassifier(classification[1](**params))
            clf.fit(train, y_train)

            clf_acc = clf.score(test, y_test)

            out = out.append(pd.DataFrame([["Multipoutput "+name,clf_name,str(clf_params),clf_acc]], columns=["Dataset", "Method", "Params","Accuracy"]), ignore_index=True)

out.to_csv("results_multioutput.csv")

# # onevsrest = OneVsRestClassifier(SVC()).fit(x_train, y_train)
# # onevsrest.score(x_test, y_test)
# # tree = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
# # extra = ExtraTreesClassifier(n_estimators=200).fit(x_train, y_train)
# ovr_ada = MultiOutputClassifier(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300)).fit(x_train_single, y_train)
# ovr_ada.score(x_test_single, y_test)
# ovr_tree = MultiOutputClassifier(DecisionTreeClassifier(criterion="entropy")).fit(x_train_single, y_train)
# ovr_tree.score(x_test_single, y_test)
# chain_tree = ClassifierChain(DecisionTreeClassifier(criterion="entropy")).fit(x_train_single, y_train)
# chain_tree.score(x_test_single, y_test)
# # chain_extra = ClassifierChain(ExtraTreesClassifier(n_estimators=100)).fit(x_train, y_train)
# mcp = MLPClassifier().fit(x_train, y_train)
# # mcp2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500).fit(x_train, y_train)
# mnb = MultiOutputClassifier(MultinomialNB()).fit(x_train_single, y_train)
# mnb.score(x_test_single, y_test)
# lgd = MultiOutputClassifier(SGDClassifier()).fit(x_train_single, y_train)
# lgd.score(x_test_single, y_test)
# log = MultiOutputClassifier(LogisticRegression()).fit(x_train_single, y_train)
# log.score(x_test_single, y_test)

# https://github.com/kk7nc/RMDL
# train_single = np.array(single)[train_index]
# test_single = np.array(single)[test_index]

# RMDL.RMDL_Text.Text_Classification(train_single, y_train, test_single, y_test,
#             #  batch_size=batch_size,
#             #  sparse_categorical=True,
#             #  random_deep=Random_Deep,
#              epochs=[20, 50, 50]) ## DNN--RNN-CNN

# print_cls = ovr_tree

# print(classification_report(y_train, print_cls.predict(x_train_single), target_names=classes["Cluster"]))
# print(classification_report(y_test, print_cls.predict(x_test_single), target_names=classes["Cluster"]))
