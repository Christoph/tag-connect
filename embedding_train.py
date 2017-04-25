import re
import pandas as pd

from gensim.models import Word2Vec
from gensim.models.wrappers import FastText


data = pd.read_csv("song_docs.csv")
data.tag_name = data.tag_doc.astype(str)
subset = data[0:1000]

used_dataset = subset.tag_doc.str.lower()

# Preprocessing
# Inner regex replaces all non alpha numerics except point and space by space
# Outer replaces all multiple spaces with one space
# The two strip function remove outside spaces
clean_dataset = used_dataset.apply(
    lambda x: re.sub(r"\s+", " ", re.sub(r"[^\w\d .]", " ", x)).rstrip().lstrip())


sentences = [d.split(".") for d in clean_dataset]
docs = [d.replace(".", " ") for d in clean_dataset]
fasttext = " ".join(used_dataset).replace(".", " ")

with open('datasets/fasttext', 'w') as file:
    file.write(fasttext)

wv_model = Word2Vec(sentences, window=5, min_count=1, workers=4, batch_words=200)
ft_model = FastText.train(
    "/home/chris/source/fastText/fasttext",
    corpus_file="datasets/fasttext", model="skipgram", min_count=1)

for name, model in {"wv": wv_model, "ft": ft_model}:
    threshold = 0.90
    counts = {}
    n_counts = {}
    n_neighs = {}
    neighbours = {}

    # Build up neighbours lists
    for song in docs:
        if len(song) > 0:
            for tag in song.split(" "):
                if tag not in neighbours:
                    number_of_top_tags = 10

                    neighs = model.most_similar(tag, topn=number_of_top_tags)

                    while neighs[-1][1] >= threshold:
                        number_of_top_tags = number_of_top_tags * 2

                        neighs = model.most_similar(tag, topn=number_of_top_tags)

                    neighbours.setdefault(tag, [n for n in neighs if n[1] >= threshold])

    # Compute all counts
    for song in docs:
        if len(song) > 0:
            for tag in song.split(" "):
                # Get neighbours
                neighs = neighbours[tag]

                # Save tag count
                if tag in counts:
                    counts[tag] += 1
                else:
                    counts.setdefault(str(tag), 1)

                # Save neighbourhood size
                if tag not in n_neighs:
                    n_neighs.setdefault(tag, len(neighs))

                # Add center tag to neighbour counts
                if tag in n_counts:
                    n_counts[tag] += 1
                else:
                    n_counts.setdefault(tag, 1)

                # Get neighbour counts
                for n in [a[0] for a in neighs]:
                    if n in n_counts:
                        n_counts[n] += 1
                    else:
                        n_counts.setdefault(n, 1)

    tag_counts = pd.DataFrame.from_dict(counts, orient="index").reset_index()
    neigh_counts = pd.DataFrame.from_dict(n_counts, orient="index").reset_index()
    neigh_neighs = pd.DataFrame.from_dict(n_counts, orient="index").reset_index()
    neighs_out = pd.DataFrame.from_dict(
        {k: ",".join([t[1] for t in v]) for k, v in neighbours.items()}, orient="index").reset_index()

    out = pd.merge(pd.merge(tag_counts, neigh_counts, on="index"), neigh_neighs, on="index")

    out.columns = ["tag_name", "count", "is_neighbour_count", "neighbour_count"]
    neighs_out.columns = ["tag_name", "neighbours"]

    # ! Already saved complete models in datasets/ !
    out.to_csv("datasets/"+name+"_counts_"+str(threshold)+".csv", index=False)
    neighs_out.to_csv("datasets/"+name+"_neighbours_"+str(threshold)+".csv", index=False)

wv_model.save("datasets/wv_model")
ft_model.save("datasets/ft_model")
