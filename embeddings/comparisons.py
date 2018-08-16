from importlib import reload
import numpy as np

import lib.embedding as embedding
import lib.details as details
import lib.helpers as helpers
import lib.vis as vis

# LOAD DATA
(texts, clean, clean_fancy, labels) = helpers.load_reuters_data("more")

used = clean_fancy
used = clean

truth = np.zeros([len(used), len(used)])
l0 = sum(np.array(labels) == 0)

truth[l0:, :l0] = 1
truth[:l0, l0:] = 1

# EMBEDDING COMPARISONS
# Document vectors
(vecs, count_vectorizer) = embedding.count(used)
(vecs, tfidf_vectorizer) = embedding.tfidf(used)
(vecs, lda_vectorizer) = embedding.lda(vecs)
vecs = embedding.average_word_vectors(used)
vecs = embedding.count_weighted_average_word_vectors(used)
vecs = embedding.tfidf_weighted_average_word_vectors(used, tfidf_vectorizer)

# Word vectors
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.HAL(used, True)
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.W2V(used)

# High space clustering of word vectors
(clust_labels, dist, vecs, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "gm", 50)
(clust_labels, dist, vecs, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "birch", 50)
(clust_labels, dist, vecs, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "aff")
(clust_labels, dist, vecs, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "hdbs")

# vecs = embedding.set_vectors(used)
# sim = embedding.similarity_matrix(vecs, "jaccard")

# reload(embedding)
# reload(vis)

# Similarity simMatrix
metric = "cosine" # cosine, emd, cm
sim = embedding.similarity_matrix(vecs, metric)

# VIS
vis.simMatrixIntersection(sim, used, truth)
vis.scree_plot(sim, vecs, nonlinear=False, uselda=True, usenmf=False)
vis.scree_plot(truth, vecs, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(sim, sim.mean()), labels)
vis.cluster_heatmap(
    vecs,
    used,
    metric=metric,
    mode="intersection",
    order=True,
    truth=truth)
vis.scatter(vecs, labels)

out = details.word_sets(used, [0, 5])
out = details.group_comp(used, [1, 7, 6], [12, 13, 14, 15])

top_words = helpers.get_top_idf_words(out, tfidf_vectorizer, 10)

top_words
texts[12]

# Reduced VIS
reduced = embedding.reduced(vecs, "svd", 80)
reduced_sim = embedding.similarity_matrix(reduced, metric)

vis.simMatrixIntersection(reduced_sim, used, truth)
vis.scree_plot(reduced_sim, reduced, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(reduced_sim, (reduced_sim.mean())), labels)
vis.cluster_heatmap(
    reduced,
    used,
    metric=metric,
    mode="intersection",
    order=True,
    truth=truth)
vis.scatter(reduced, labels)

texts[14]
