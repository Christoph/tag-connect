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
(vecs_tf, count_vectorizer) = embedding.count(used)
(vecs_tfidf, tfidf_vectorizer) = embedding.tfidf(used)
vecs_wv = embedding.average_word_vectors(used)
vecs_cwwv = embedding.count_weighted_average_word_vectors(used)
vecs_twvw = embedding.tfidf_weighted_average_word_vectors(used, tfidf_vectorizer)

# Word vectors
# (word_vecs, word_docs, vocab_vecs, vocab) = embedding.HAL(used, True)
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.W2V(used)

# High space clustering of word vectors
(clust_labels, dist, vecs_gm, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "gm", 50)
(clust_labels, dist, vecs_birch, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "birch", 50)
(clust_labels, dist, vecs_aff, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "aff")
(clust_labels, dist, vecs_hdbs, clusterer) = embedding.high_space_binning(
    word_vecs, vocab_vecs, "hdbs")

embedding_models = [
    vecs_wv, #0
    vecs_gm, #1
    vecs_tf, #2
    vecs_aff, #3
    vecs_cwwv, #4
    vecs_hdbs, #5
    vecs_twvw, #6
    vecs_tfidf, #7
    vecs_birch #8
    ]

reload(embedding)
# Similarity simMatrix
embedding_metric = "cosine" # cosine, emd, cm
comparison_metric = "cosine"

# vecs = embedding.embedding_vector(embedding_models, embedding_metric, probabilistic=True)
# vecs = embedding.embedding_vector_radius(embedding_models, embedding_metric, radius=0.1, probabilistic=True)
vecs = embedding.set_vectors(used)

sim = embedding.similarity_matrix(vecs, metric="jaccard")

# VIS
vis.simMatrixIntersection(sim, used)
# vis.scree_plot(sim, vecs, nonlinear=False, uselda=True, usenmf=False)
vis.graph(embedding.graph_from_sim(sim, 0.6))
vis.graph(embedding.graph_from_sim(sim, sim.mean()))
vis.cluster_heatmap(
    vecs,
    used,
    metric=comparison_metric,
    mode="intersection",
    order=True)
# vis.scatter(vecs, labels)
