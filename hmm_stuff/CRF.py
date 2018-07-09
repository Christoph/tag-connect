from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import pandas as pd
import spacy


nlp = spacy.load('en')
data = pd.read_csv("data/train_small.csv")
test = pd.read_csv("data/test_hand_labeled.csv")
docs = [nlp(row) for row in data["text"]]
labels = [row for row in data["label"]]

def word2feature(word):
    postag = word.pos_

    features = [
        'bias',
        'word.lower=%s' % word.is_lower,
        'word[-3:]=' + word.text[-3:],
        'word[-2:]=' + word.text[-2:],
        'word.isupper=%s' % word.is_upper,
        'word.istitle=%s' % word.is_title,
        'word.isdigit=%s' % word.is_digit,
        'postag=' + postag,
    ]

    if word.i > 0:
        word1 = word.nbor(-1)
        postag1 = word1.pos_
        features.extend([
            '-1:word.lower=%s' % word1.is_lower,
            '-1:word.istitle=%s' % word1.is_title,
            '-1:word.isupper=%s' % word1.is_upper,
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')

    if word.i < len(word.doc)-1:
        word1 = word.nbor(1)
        postag1 = word1.pos_
        features.extend([
            '+1:word.lower=%s' % word1.is_lower,
            '+1:word.istitle=%s' % word1.is_title,
            '+1:word.isupper=%s' % word1.is_upper,
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features

def doc2features(doc):
    return [word2feature(t) for t in doc]

X_train = [doc2features(d) for d in docs]
y_train = labels

X_test = [doc2features(nlp(d)) for d in test["text"]]
y_test = [d for d in test["label"]]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
