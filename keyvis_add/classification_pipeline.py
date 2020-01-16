import sys
from os import path
from importlib import reload
from collections import Counter
import random

import re
import pandas as pd
import numpy as np
import torch
# from textblob import TextBlob
from sklearn.decomposition import NMF, LatentDirichletAllocation, FastICA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import spacy
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, precision_recall_fscore_support
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

nlp = spacy.load('en_core_web_sm', disable=['ner'])
stop_words = stopwords.words('english')

# Add general functions to the project
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


def preprocess_keywords(text, sep=";", merge_char=";"):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # replace non characers with space
    regexr = re.sub(r"[^a-zA-Z0-9. ]+", " ", text.replace(sep, "."))
    # merge multiple spaces to a single one
    cleared = re.sub(r"[[ ]+", " ", regexr)

    # for doc in nlp(cleared).sents:
    for keyword in cleared.split("."):
        doc = nlp(keyword)
        temp = " ".join((token.lemma_ for token in doc if
                         len(token.lemma_) > 1 and
                         not token.lemma_ == "-PRON-" and
                         str(token) != "."))
        if len(temp) > 0:
            texts_out.append(temp.lower())

    # Make sure each keyword is unique
    texts_out = list(set(texts_out))
    return merge_char.join(texts_out)


def preprocess_text(text, stopwords, remove_num=True, merge_char=" "):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # replace non characers with space
    regexr = re.sub(r"[^a-zA-Z0-9.!? ]+", " ", text)
    # merge multiple spaces to a single one
    cleared = re.sub(r"[ ]+", " ", regexr)

    for doc in nlp(cleared).sents:
        if(remove_num):
            temp = " ".join((token.lemma_ for token in doc if
                             not token.like_num and
                             not token.like_url and
                             not token.like_email and
                             token.lemma_ not in stop_words and
                             len(token.lemma_) > 1 and
                             not token.lemma_ == "-PRON-"))
        else:
            temp = " ".join((token.lemma_ for token in doc if
                             not token.like_url and
                             not token.like_email and
                             token.lemma_ not in stop_words and
                             len(token.lemma_) > 1 and
                             not token.lemma_ == "-PRON-"))
        texts_out.append(temp)

    return merge_char.join(texts_out)


def get_top_words(model, tfidf, n_top_words):
    out = []
    feature_names = tfidf.get_feature_names()
    idf = tfidf.idf_
    vocab = tfidf.vocabulary_

    for topic_idx, topic in enumerate(model.components_):
        words = [(feature_names[i], idf[vocab[feature_names[i]]])
                 for i in topic.argsort()[:-n_top_words - 1:-1]]
        out.append(words)
    return out

def select_svd_dim(vecs, explained_variance_threshold=0.3, step_size=2, max_dim=200):
    dim = 0
    target = 0
    iteration = 0

    print("Find optimal dimension")
    while target < explained_variance_threshold:
        iteration += 1
        # Increase dimensionality
        dim += step_size

        # Fit svd
        temp_svd = TruncatedSVD(dim)

        temp_svd.fit(vecs)

        # Get explained variance
        variance = temp_svd.explained_variance_ratio_.sum()

        if(variance >= target):
            target = variance

        if iteration % 5 == 0:
            step_size *= 2

        print("Current dim: ", dim, " Current var: ",
              variance, "Current step_size: ", step_size)

        # if(dim > max_dim - step_size or dim + step_size >= vec.get_shape()[1]):
        #     target = explained_variance_threshold

    return dim


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
old_data = pd.read_json("datasets/old_data.json", orient="index")
new_data = pd.read_excel("datasets/manual_data.xlsx",
                         orient="index", header=1).iloc[0:50]
# pd.read_json("datasets/new_data.json", orient="index")
datasets = [
    old_data,
    new_data
]

keywords = []
keyword_tfidf_vecs = []
keyword_svd_vecs = []

abstracts = []
abstract_tfidf_vecs = []
abstract_svd_vecs = []


def preprocessData(datasets):
    ### KEYWORDS ###
    # Preprocess keywords
    for data in datasets:
        keywords.append(
            ["" if key == None else preprocess_keywords(key) for key in list(list(data["Keywords"]))])

    # Vectorize keywords
    keyword_tfidf = TfidfVectorizer(stop_words=stop_words)

    for data in keywords:
        keyword_tfidf.fit(data)

    # Transform keywords
    for data in keywords:
        keyword_tfidf_vecs.append(keyword_tfidf.transform(data))

    # Create svd transformer
    keyword_svd = TruncatedSVD(select_svd_dim(keyword_tfidf_vecs))

    # Fit the SVD
    for data in keyword_tfidf_vecs:
        keyword_svd.fit(data)

    # Transform TFIDF -> SVD
    for data in keyword_tfidf_vecs:
        keyword_svd_vecs.append(keyword_svd.transform(data))

    ### ABSTRACTS ###
    # Preprocess abstracts
    for data in datasets:
        abstracts.append(
            ["" if ab == None else preprocess_text(ab, stop_words, remove_num=False) for ab in list(data["Abstract"])])

    # Vectorize keywords
    abstract_tfidf = TfidfVectorizer(max_df=0.7)

    for data in abstracts:
        abstract_tfidf.fit(data)

    # Transform keywords
    for data in abstracts:
        abstract_tfidf_vecs.append(abstract_tfidf.transform(data))

    # Create svd transformer
    abstract_svd = TruncatedSVD(select_svd_dim(abstract_tfidf_vecs))

    # Fit the SVD
    for data in abstract_tfidf_vecs:
        abstract_svd.fit(data)

    # Transform TFIDF -> SVD
    for data in abstract_tfidf_vecs:
        abstract_svd_vecs.append(abstract_svd.transform(data))

    # Save data
    for index, data in enumerate(datasets):
        # Set processed keywords
        data["Keywords_Processed"] = keywords[index]

        data["Keyword_Vector"] = ""
        data['Keyword_Vector'] = data['Keyword_Vector'].astype(object)

        data["Abstract_Vector"] = ""
        data['Abstract_Vector'] = data['Abstract_Vector'].astype(object)

        for i in data.index:
            data.at[i, "Keyword_Vector"] = list(
                pd.Series(keyword_svd_vecs[index][i]))
            data.at[i, "Abstract_Vector"] = list(
                pd.Series(abstract_svd_vecs[index][i]))



# Saving data
new_data.to_json("new_data.json", orient="index")
old_data.to_json("old_data.json", orient="index")

# Normalize all keywords
mapping = pd.read_json("datasets/mapping.json", orient="index")

for index, row in mapping.iterrows():
    keyword = row["AuthorKeyword"]
    label = row["ExpertKeyword"]

    cleared = preprocess_keywords(keyword)
    clear_label = re.sub(r"[ ]+", " ", label.replace("-", " ").replace(
        "/", " ").replace("+", " ").replace("&", " ").replace(",", " "))

    # fixed_label = "".join([word.capitalize()
    #    for word in clear_label.split(" ")])
    fixed_label = clear_label.replace(" ", "")

    mapping.set_value(index, 'AuthorKeyword', cleared)
    mapping.set_value(index, 'ExpertKeyword', fixed_label)

mapping.to_json("mapping.json", orient="index")

# Export study data
meta = pd.read_json("../datasets/old_data.json", orient="index")
mapping = pd.read_json("../datasets/mapping.json", orient="index")
classes = pd.read_json("../datasets/classes.json", orient="index")

study_data = meta.drop(["Abstract_Vector", "Keyword_Vector"], axis=1)
study_data["Labels"] = ""

# Select study datasets based on
# Manual data is from 2013
# Tool data is from 2012
# Minimum duplicate authors for each set
# Same amount of Keywords for each author -> 100?
# More then 3 and less then 7 Keywords per publication

manual_data_all, tool_data_all = train_test_split(study_data, test_size=0.5)

filtered_manual_data = manual_data_all[manual_data_all["DOI"].str.contains('2011|2012|2013', regex=True)]
filtered_tool_data = tool_data_all[tool_data_all["DOI"].str.contains('2011|2012|2013', regex=True)]

authors = set()
keywords = []
manual_data = pd.DataFrame(columns=filtered_manual_data.columns)
tool_data = pd.DataFrame(columns=filtered_manual_data.columns)

# Only unique authors and keyword count between 3 and 7
for index, row in filtered_manual_data.iterrows():
    temp = row["Authors"].split(";")
    # if len(temp) >= 3:
    #     temp = [temp[0], temp[-1]]
        
    if any(n in authors for n in temp):
        pass
    else:
        keys = row["Keywords"].split(";")
        if len(keys) >= 3 and len(keys) <= 7:
            authors.update(temp)
            manual_data = manual_data.append(row, ignore_index=True)

            keywords.extend(keys)

count = Counter(keywords)
frequent = {x : count[x] for x in count if count[x] >= 2}
rest = {x : count[x] for x in count if count[x] < 2}

manual_docs = pd.DataFrame(columns=manual_data.columns)
keyword_counter = set()

for i, row in manual_data.iterrows():
    keys = row["Keywords"].split(";")
    if any(n in list(frequent.keys()) for n in keys):
        if (len(keyword_counter) + len(keys)) <= 100:
            keyword_counter.update(keys)
            manual_docs = manual_docs.append(row, ignore_index=True)

manual_keywords = list(keyword_counter)

authors = set()
keywords = []

# Only unique authors and keyword count between 3 and 7
for index, row in filtered_tool_data.iterrows():
    temp = row["Authors"].split(";")
    # if len(temp) >= 3:
    #     temp = [temp[0], temp[-1]]
        
    if any(n in authors for n in temp):
        pass
    else:
        keys = row["Keywords"].split(";")
        if len(keys) >= 3 and len(keys) <= 7:
            authors.update(temp)
            manual_data = manual_data.append(row, ignore_index=True)

            keywords.extend(keys)

count = Counter(keywords)
frequent = {x : count[x] for x in count if count[x] >= 2}

count = Counter(keywords)
frequent = {x : count[x] for x in count if count[x] >= 2}
rest = {x : count[x] for x in count if count[x] < 2}

tool_docs = pd.DataFrame(columns=manual_data.columns)
keyword_counter = set()

for i, row in manual_data.iterrows():
    keys = row["Keywords"].split(";")
    if any(n in list(frequent.keys()) for n in keys):
        if (len(keyword_counter) + len(keys)) <= 100:
            keyword_counter.update(keys)
            tool_docs = tool_docs.append(row, ignore_index=True)

tool_keywords = list(keyword_counter)

# Output
manual_docs.to_csv("manual_docs.csv", index=False)
tool_docs.to_csv("tool_docs.csv", index=False)

tkout = pd.DataFrame(tool_keywords, columns=["keyword"])
tkout["label"] = ""
tkout["time"] = ""
tkout["truth"] = ""

for i, row in tkout.iterrows():
    m = mapping[mapping["AuthorKeyword"] == row["keyword"]]
    row["truth"] = list(m["ExpertKeyword"])[0]

tkout.to_csv("tool_keywords.csv", index=True)

mkout = pd.DataFrame(manual_keywords, columns=["keyword"])
mkout["label"] = ""
mkout["time"] = ""
mkout["truth"] = ""

for i, row in mkout.iterrows():
    m = mapping[mapping["AuthorKeyword"] == row["keyword"]]
    row["truth"] = list(m["ExpertKeyword"])[0]

mkout.to_csv("manual_keywords.csv", index=True)

# mapping_data = mapping.drop(
#     ["AuthorKeywordCount", "ExpertKeywordCount"], axis=1)
# mapping_data.columns = ["Keyword", "Label"]
# label_data = classes.drop(["Vector"], axis=1)
# label_data.columns = ["Label"]

# manual_data.to_csv("manual_data.csv", index=False)
# tool_data.to_csv("tool_data.csv", index=False)
# mapping_data.to_csv("mapping.csv", index=False)
# label_data.to_csv("labels.csv", index=False)

# subset = manual_data.iloc[:35]
# subset.to_csv("subset.csv", index=False)
# print(frequent)

# subset_keywords = []

# for index, row in subset.iterrows():
#     keys = preprocess_keywords(row["Keywords"])

#     for word in keys.split(";"):
#         subset_keywords.append(word)

# subset_keywords = list(set(subset_keywords))

# new_mappings = pd.DataFrame(subset_keywords)
# new_mappings["1"] = ""
# new_mappings.columns = ["Keyword", "Label"]

# for index, row in new_mappings.iterrows():
#     result = mapping.loc[mapping['AuthorKeyword']
#                          == row["Keyword"]]["ExpertKeyword"]

#     if len(result) > 0:
#         word = result.iloc[0].replace(",", "")
#     else:
#         word = ""

#     new_mappings.at[index, "Label"] = word

# sum(new_mappings["Label"] != "")
# new_mappings = new_mappings[new_mappings["Label"] == ""]

# new_mappings.to_csv("new_mapping.csv", index=False)

# Automatic performance measurement

# Parameters:
# Data - fulltexts, abstracts, keywords_single, keywords_multi
# Embedding -

# data
meta = pd.read_json("datasets/old_data.json", orient="index").sort_index()
new_data = pd.read_excel(
    "datasets/manual_data.xlsx", orient="index", header=1).iloc[0:50]

# Remove leading and trailing ;
meta['Clusters'] = meta['Clusters'].apply(lambda x: x.strip(';'))

# abstracts
abstracts = ["" if ab == None else preprocess_text(
    ab, stop_words, remove_num=False) for ab in list(meta["Abstract"])]

new_abstracts = ["" if ab == None else preprocess_text(
    ab, stop_words, remove_num=False) for ab in list(new_data["Abstract"])]

# keywords
keywords = meta["Keywords_Processed"]
multi = [key.replace(" ", "_") for key in keywords.tolist()]
single = [key for key in keywords.tolist()]

new_keywords = new_data["Keywords_Processed"]
new_multi = [key.replace(" ", "_") for key in new_keywords.tolist()]
new_single = [key for key in new_keywords.tolist()]

# embedding
# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";")
         for cluster in meta["Clusters"].tolist()])

y = np.vstack(meta.apply(
    lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)

classes = pd.DataFrame(enc.classes_, columns=["Cluster"])

# x
# TFIDF
abstract_tfidf = TfidfVectorizer(max_df=0.8)
abstract_tfidf.fit(abstracts)
abstract_tfidf.fit(new_abstracts)

abstract_vecs = abstract_tfidf.transform(abstracts)
new_abstract_vecs = abstract_tfidf.transform(new_abstracts)

abstract_tfidf_60 = TfidfVectorizer(max_df=0.6)
abstract_tfidf_60.fit(abstracts)
abstract_tfidf_60.fit(new_abstracts)

abstract_60_vecs = abstract_tfidf_60.transform(abstracts)
new_abstract_60_vecs = abstract_tfidf_60.transform(new_abstracts)

single_keyword_tfidf = TfidfVectorizer()
single_keyword_tfidf.fit(single)
single_keyword_tfidf.fit(new_single)

single_keyword_vecs = single_keyword_tfidf.transform(single)
new_single_keyword_vecs = single_keyword_tfidf.transform(new_single)

multi_keyword_tfidf = TfidfVectorizer()
multi_keyword_tfidf.fit(multi)
multi_keyword_tfidf.fit(new_multi)

multi_keyword_vecs = multi_keyword_tfidf.transform(multi)
new_multi_keyword_vecs = multi_keyword_tfidf.transform(new_multi)

# BERT embedding
nlp_bert = spacy.load('en_trf_bertbaseuncased_lg')

is_using_gpu = spacy.prefer_gpu()  
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

bert_single_vecs = []
new_bert_single_vecs = []

# Embed all docs
for doc in single:
    bert_single_vecs.append(nlp_bert(doc).vector)

for doc in new_single:
    new_bert_single_vecs.append(nlp_bert(doc).vector)

bert_abstract_vecs = []
new_bert_abstract_vecs = []

# Embed all docs
for doc in abstracts:
    bert_abstract_vecs.append(nlp_bert(doc).vector)

for doc in new_abstracts:
    new_bert_abstract_vecs.append(nlp_bert(doc).vector)

# classification
datasets = [
    ["abstract max_df=0.8", abstract_vecs],
    ["abstract max_df=0.6", abstract_60_vecs],
    ["single keywords", single_keyword_vecs],
    ["bert single keywords", np.array(bert_single_vecs)],
    ["bert abstracts", np.array(bert_abstract_vecs)],
    ["multi keywords", multi_keyword_vecs]
]

dimension_reductions = [
    ["SVD",
     TruncatedSVD,
     [
         {
             "explained_variance_threshold": 0.4,
             "step_size": 5,
             "max_dim": 400,
         },
        #  {
        #      "explained_variance_threshold": 0.6,
        #      "step_size": 10,
        #      "max_dim": 400,
        #  },
        #  {
        #      "explained_variance_threshold": 0.8,
        #      "step_size": 15,
        #      "max_dim": 600,
        #  },
     ]],
    # ["NMF",
    #  NMF,
    #  [
    #      {

    #      }
    #  ]]
]

# NMF(20, init="nndsvda").fit(abstract_vecs).reconstruction_err_

classifications = [
    ["DecisionTree", DecisionTreeClassifier, [
        {"criterion": "gini", "min_samples_split": 0.01},
        {"criterion": "entropy", "min_samples_split": 0.01},
        {"criterion": "gini", "min_samples_split": 0.05},
        {"criterion": "entropy", "min_samples_split": 0.05},
        {"criterion": "gini"},
        {"criterion": "entropy"},
    ]],
    # Very slow
    ["AdaBoost", AdaBoostClassifier, [
        {"n_estimators": 25, "learning_rate": 1},
        {"n_estimators": 25, "learning_rate": 0.5},
        {"n_estimators": 50, "learning_rate": 1},
        {"n_estimators": 100, "learning_rate": 1},
        # {"n_estimators": 200, "learning_rate": 1},
        # {"n_estimators": 300, "learning_rate": 1},
    ]],
    ["GradientBoostingClassifier", GradientBoostingClassifier, [
        {"n_estimators": 25},
        {"n_estimators": 50},
        {"n_estimators": 100},
        {"n_estimators": 200},
        # {"n_estimators": 300},
    ]],
    ["SVM", SVC, [
        {"gamma": "scale"},
        {"c": 2, "gamma": "scale"},
        {"gamma": "scale", "kernel": "linear"},
        {"c": 2, "gamma": "scale", "kernel": "linear"},
    ]],
    ["Random Forest", RandomForestClassifier, [
        {"n_estimators": 200, "criterion": "entropy", "min_samples_split": 0.01},
        {"n_estimators": 200, "criterion": "entropy", "min_samples_split": 0.05},
        {"n_estimators": 100, "criterion": "gini"},
        {"n_estimators": 100, "criterion": "entropy"},
        {"n_estimators": 200, "criterion": "gini"},
        {"n_estimators": 200, "criterion": "entropy"},
        {"n_estimators": 300, "criterion": "gini"},
        {"n_estimators": 300, "criterion": "entropy"},
        {"n_estimators": 200, "criterion": "gini", "max_leaf_nodes": 179},
        {"n_estimators": 200, "criterion": "entropy", "max_leaf_nodes": 179},
    ]],
    ["MLP", MLPClassifier, [
        {"hidden_layer_sizes": 20, "activation": "relu",
            "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": 50, "activation": "relu",
            "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": 100, "activation": "relu",
            "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": 200, "activation": "relu",
            "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": (20, 20), "activation": "relu",
         "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": (50, 50), "activation": "relu",
         "solver": "lbfgs", "max_iter": 200},
        {"hidden_layer_sizes": (100, 100), "activation": "relu",
         "solver": "lbfgs", "max_iter": 200},
    ]]
]


def find_best_classifier(datasets, dimension_reductions, classifications):
    out = pd.DataFrame(
        columns=["Dataset", "DR", "Dimensions", "Method", "Params", "Accuracy", "Precision", "Recall"])

    # Iterate datasets
    for data_id, dataset in enumerate(datasets):
        name = dataset[0]
        data = dataset[1]
        skf = ShuffleSplit(n_splits=2)
        split_indices = []
                
        for train_index, test_index in skf.split(data, y):
            split_indices.append((train_index, test_index))

        print("datasets: ", str(data_id+1), "/", str(len(datasets)))

        # Iterate classifications
        for cls_id, classification in enumerate(classifications):
            clf_name = classification[0]
            clf_params = classification[2]

            print("classifier: ", clf_name, ", ", str(cls_id+1), "/", len(classifications))

            # Iterate parametrizations
            for p_id, param in enumerate(clf_params):
                print("Params: ", param, ", ", str(p_id+1), "/"+str(len(clf_params)))

                acc_scores = []
                pre_scores = []
                rec_scores = []

                # Iterate splits
                for train_index, test_index in split_indices:

                    X_train, X_test = data[train_index], data[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    clf = MultiOutputClassifier(classification[1](**param))
                    try:
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        prfs = precision_recall_fscore_support(y_test, y_pred, warn_for=[])

                        acc_scores.append(clf.score(X_test, y_test))
                        pre_scores.append(prfs[0].mean())
                        rec_scores.append(prfs[1].mean())
                    except:
                        print("Exception during fitting")
                        acc_scores.append(0)
                        pre_scores.append(0)
                        rec_scores.append(0)

                clf_acc = np.array(acc_scores).mean()
                clf_pre = np.array(pre_scores).mean()
                clf_rec = np.array(rec_scores).mean()
                out = out.append(pd.DataFrame([[name, "None", "original", clf_name, str(param), clf_acc, clf_pre, clf_rec]], columns=[
                    "Dataset", "DR", "Dimensions", "Method", "Params", "Accuracy", "Precision", "Recall"]), ignore_index=True)
                
            out.to_csv("results.csv", index=False)

        # Iterate the dimension reductions
        if "bert" not in name:
            for dr_m_id, dr_method in enumerate(dimension_reductions):
                dr_name = dr_method[0]
                dr_params = dr_method[2]

                print("DR Method: ", dr_method, ", ", str(dr_m_id+1), "/"+str(len(dimension_reductions)))

                # Iterate the dr parametrizations
                for dr_id, dr_params in enumerate(dr_params):
                    print("Params: ", dr_params, ", ", str(dr_id+1), "/"+str(len(clf_params)))

                    dim = select_svd_dim(data, **dr_params)
                    dr = dr_method[1](dim).fit_transform(data)

                    # Iterate the classifications
                    for cls_id, classification in enumerate(classifications):
                        clf_name = classification[0]
                        clf_params = classification[2]

                        print("classifier: ", clf_name, ", ", str(cls_id+1), "/", str(len(classifications)))

                        # Iterate the clf params
                        for p_id, param in enumerate(clf_params):
                            print("Params: ", param, ", ", p_id+1, "/"+str(len(clf_params)))

                            acc_scores = []
                            pre_scores = []
                            rec_scores = []

                            for train_index, test_index in split_indices:
                                X_train, X_test = dr[train_index], dr[test_index]
                                y_train, y_test = y[train_index], y[test_index]

                                try:
                                    clf = MultiOutputClassifier(
                                        classification[1](**param))
                                    clf.fit(X_train, y_train)

                                    y_pred = clf.predict(X_test)

                                    prfs = precision_recall_fscore_support(

                                    y_test, y_pred, warn_for=[])
                                    acc_scores.append(clf.score(X_test, y_test))
                                    pre_scores.append(prfs[0].mean())
                                    rec_scores.append(prfs[1].mean())
                                except:
                                    print("Exception during fitting")
                                    acc_scores.append(0)
                                    pre_scores.append(0)
                                    rec_scores.append(0)

                            clf_acc = np.array(acc_scores).mean()
                            clf_pre = np.array(pre_scores).mean()
                            clf_rec = np.array(rec_scores).mean()
                            out = out.append(pd.DataFrame([[name, dr_name, dim, clf_name, str(param), clf_acc, clf_pre, clf_rec]], columns=[
                                "Dataset", "DR", "Dimensions", "Method", "Params", "Accuracy", "Precision", "Recall"]), ignore_index=True)

                        # Save after each classification
                        out.to_csv("results.csv", index=False)

    # Final save
    out.to_csv("results.csv", index=False)

    print("DONE!")

# Train with best classifier
# DecisionTree

classifier = DecisionTreeClassifier(criterion = "entropy").fit(single_keyword_vecs)

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
