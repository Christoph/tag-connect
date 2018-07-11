from importlib import reload

import embedding
import helpers
import vis

# LOAD DATA
(texts, clean, clean_fancy, labels) = helpers.load_reuters_data()

used = clean_fancy
used = clean

# EMBEDDING COMPARISONS
# Document vectors
(vecs, vectorizer) = embedding.count(used)
(vecs, vectorizer) = embedding.tfidf(used)
vecs = embedding.average_word_vectors(used)
vecs = embedding.count_weighted_average_word_vectors(used)
vecs = embedding.tfidf_weighted_average_word_vectors(used, vectorizer)

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

reload(embedding)
reload(vis)

# Similarity simMatrix
metric = "cosine" # cosine, emd, cm
sim = embedding.similarity_matrix(vecs, metric)

# VIS
vis.simMatrixIntersection(sim, used)
vis.scree_plot(sim, vecs, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(sim, 0.25))
vis.cluster_heatmap(
    vecs,
    used,
    metric=metric,
    mode="intersection",
    order=True)
vis.scatter(vecs, labels)

texts[14]

# Reduced VIS
reduced = embedding.reduced(vecs, "svd", 19)
reduced_sim = embedding.similarity_matrix(reduced, metric)

vis.simMatrixIntersection(reduced_sim, used)
vis.scree_plot(reduced_sim, reduced, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(reduced_sim, 0.25))
vis.cluster_heatmap(
    reduced,
    used,
    metric=metric,
    mode="intersection",
    order=True)
vis.scatter(reduced, labels)

texts[14]
