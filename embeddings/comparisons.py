from importlib import reload

import embedding
import details
import helpers
import vis

# LOAD DATA
(texts, clean, clean_fancy, labels) = helpers.load_reuters_data()

used = clean_fancy
used = clean

# EMBEDDING COMPARISONS
# Document vectors
(vecs, count_vectorizer) = embedding.count(used)
(vecs, tfidf_vectorizer) = embedding.tfidf(used)
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

# reload(embedding)
# reload(vis)

# Similarity simMatrix
metric = "cosine" # cosine, emd, cm
sim = embedding.similarity_matrix(vecs, metric)

# VIS
vis.simMatrixIntersection(sim, used)
vis.scree_plot(sim, vecs, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(sim, 1.7))
vis.graph(embedding.graph_from_sim(sim, sim.mean()))
vis.cluster_heatmap(
    vecs,
    used,
    metric=metric,
    mode="intersection",
    order=True)
vis.scatter(vecs, labels)

out = details.word_sets(used, [12, 13, 14, 15, 16])
out = details.group_comp(used, [1, 7, 6], [12, 13, 14, 15])

top_words = helpers.get_top_idf_words(out, tfidf_vectorizer, 10)

top_words
texts[12]

# Reduced VIS
reduced = embedding.reduced(vecs, "svd", 20)
reduced_sim = embedding.similarity_matrix(reduced, metric)

vis.simMatrixIntersection(reduced_sim, used)
vis.scree_plot(reduced_sim, reduced, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(reduced_sim, reduced_sim.mean()))
vis.graph(embedding.graph_from_sim(reduced_sim, 0.25))
vis.cluster_heatmap(
    reduced,
    used,
    metric=metric,
    mode="intersection",
    order=True)
vis.scatter(reduced, labels)

texts[14]
