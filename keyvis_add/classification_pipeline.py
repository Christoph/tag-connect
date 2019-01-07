from importlib import reload

import re
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

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import ClassifierChain
from sklearn.neural_network import MLPClassifier

nlp = spacy.load('en_core_web_md', disable=['ner'])

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
stop_words = stopwords.words('english')
# docs = (nlp(text) for text in raw["Fulltext"].tolist())
# full_lemma = (lemmatization(doc, stop_words) for doc in docs)
# fulltext_texts = [" ".join(text) for text in full_lemma]
# pd.DataFrame(fulltext_texts).to_json("fulltext_lemma.json", orient="index")


# fulltexts = pd.read_json("../datasets/fulltext_lemma.json", orient="index").sort_index()
meta = pd.read_json("../datasets/meta.json", orient="index").sort_index()
keywords = meta["Keywords"]
# Remove leading and trailing ;
meta['Clusters'] = meta['Clusters'].apply(lambda x: x.strip(';'))

# CLASSIFICATION
# train/test split for classification
test_index = meta[meta["type"] == "new"].index  # len = 197
train_index = meta.drop(test_index).index  # len = 1280

# y
enc = MultiLabelBinarizer()
enc.fit([cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()])

y_train = np.vstack(meta.iloc[train_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)
# y_train = meta.iloc[train_index].apply(lambda row: [row["Clusters"].split(";")][0], axis=1).values
y_test = np.vstack(meta.iloc[test_index].apply(lambda row: enc.transform([row["Clusters"].split(";")])[0], axis=1).values)

# x
#fulltext
fulltext_tfidf = TfidfVectorizer(max_df=0.5).fit(fulltexts[0].tolist())
fulltext_vecs = fulltext_tfidf.transform(fulltexts[0].tolist())

nmf = NMF(10)
vecs = nmf.fit_transform(fulltext_vecs)
vecs = np.asarray(vecs, dtype=np.object)

x_train = vecs[train_index]
x_test = vecs[test_index]

# keywords multiword
# multi = [lemmatization(key.replace(" ", "_"), stopwords) for key in keywords.tolist()]
multi = [preprocess(key.replace(" ", "_"), stopwords) for key in keywords.tolist()]
# multi = [key.replace(" ", "_") for key in keywords.tolist()]
multi_tfidf = TfidfVectorizer().fit(multi)
multi_vecs = multi_tfidf.transform(multi)

x_train = multi_vecs[train_index]
x_test = multi_vecs[test_index]

# keywords single word
single = [preprocess(key, stopwords) for key in keywords.tolist()]
single_tfidf = TfidfVectorizer().fit(single)
single_vecs = single_tfidf.transform(single)

x_train = single_vecs[train_index]
x_test = single_vecs[test_index]

# classifiers
# onevsrest = OneVsRestClassifier(SVC()).fit(x_train, y_train)
tree = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
# extra = ExtraTreesClassifier(n_estimators=200).fit(x_train, y_train)
ovr_ada = MultiOutputClassifier(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300)).fit(x_train, y_train)
ovr_tree = MultiOutputClassifier(DecisionTreeClassifier(criterion="entropy")).fit(x_train, y_train)
chain_tree = ClassifierChain(DecisionTreeClassifier(criterion="entropy")).fit(x_train, y_train)
chain_extra = ClassifierChain(ExtraTreesClassifier(n_estimators=100)).fit(x_train, y_train)
mcp = MLPClassifier(max_iter=500).fit(x_train, y_train)
mcp2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500).fit(x_train, y_train)

print(classification_report(y_train, ovr_tree.predict(x_train)))
print(classification_report(y_test, ovr_tree.predict(x_test)))

# custom classifier
clusters = [cluster.split(";") for cluster in meta.iloc[train_index]["Clusters"].tolist()]
keywords = [r.split() for r in multi]

pairs = zip(keywords[:100], clusters[:100])
mapping = {}
for pair in pairs:
    for keyword in pair[0]:
        if keyword in mapping:
            temp = mapping[keyword]
            temp.union(pair[1])
        else:
            mapping[keyword] = set(pair[1])

out_keywords = pd.DataFrame(pd.Series(mapping))[0].apply(lambda x: list(x))
out_keywords.to_json("keyword_mapping.json", orient="index")

# save the jsons
pd.DataFrame(enc.classes_).to_json("classes.json", orient="values")
pd.DataFrame(y_train).to_json("old_labels.json", orient="values")
pd.DataFrame(ovr_tree.predict(x_test)).to_json("new_labels_1.json", orient="values")
pd.DataFrame(chain_tree.predict(x_test)).to_json("new_labels_2.json", orient="values")
pd.DataFrame(chain_tree.predict(x_test)).to_json("new_labels_3.json", orient="values")
