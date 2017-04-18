import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv("datasets/tag_docs.csv")
subset = data[0:10000]

used_dataset = data.tag_doc.str.lower()

sentences = [d.split(".") for d in used_dataset]
docs = [d.replace(".", " ") for d in used_dataset]
fasttext = " ".join(used_dataset).replace(".", " ")

with open('datasets/fasttext', 'w') as file:
    file.write(fasttext)

counts = {}

for song in docs:
    for tag in song.split(" "):
        if tag in counts:
            counts[tag] += 1
        else:
            counts.setdefault(tag, 1)

tag_counts = pd.DataFrame.from_dict(counts, orient="index").reset_index()
tag_counts.columns = ["tag_name", "count"]
wv_model = Word2Vec(sentences, window=5, min_count=1, workers=4)
ft_model = FastText.train("/home/chris/source/fastText/fasttext", corpus_file="datasets/fasttext", model="skipgram", min_count=1)
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)

features = vectorizer.get_feature_names()
idf = vectorizer.idf_

feats = pd.DataFrame({
    "feature": features,
    "idf": idf
})
feats = feats.sort_values(
    "idf", ascending=False)

# ! Already saved complete models in datasets/ !
feats.to_csv("datasets/idf_scores.csv", index=False)
tag_counts.to_csv("datasets/counts.csv", index=False)
wv_model.save("datasets/wv_model")
ft_model.save("datasets/ft_model")
