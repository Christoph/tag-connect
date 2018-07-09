from importlib import reload
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import embedding
import vis
import helpers
import cluster_analysis

# LOAD DATA
(texts, clean, clean_fancy, labels) = helpers.load_reuters_data()

used = clean_fancy
used = clean

# EMBEDDING
# Word vectors
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.HAL(used, True)
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.W2V(used)

# Document vectors
(vecs, vectorizer) = embedding.count(used)
(vecs, vectorizer) = embedding.tfidf(used)

truth = cosine_similarity(vecs)

# COMPARISONS
reload(embedding)
reload(vis)

# Word counts
(vecs, vectorizer) = embedding.count(used)
sim = cosine_similarity(vecs)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs.toarray(), dense=True)

# TFIDF
(vecs, vectorizer) = embedding.tfidf(used)
sim = cosine_similarity(vecs)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs.toarray(), dense=True)

# Average word vectors
vecs = embedding.average_word_vectors(used)
sim = cosine_similarity(vecs)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs, dense=True)

# Weighted average word vectors
vecs = embedding.count_weighted_average_word_vectors(used)
sim = cosine_similarity(vecs)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs, dense=True)

vecs = embedding.tfidf_weighted_average_word_vectors(used, vectorizer)
sim = cosine_similarity(vecs)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs, dense=True)

# Histograms with EMD
# fancy, tfidf, birch 20 show interesting results
(clust_labels, dist, clusterer) = embedding.high_space_binning(word_vecs, vocab_vecs, "maxclust", 20, vocab=vocab, docs=used)
(clust_labels, dist, clusterer) = embedding.high_space_binning(word_vecs, vocab_vecs, "birch", 20)
(clust_labels, dist, clusterer) = embedding.high_space_binning(word_vecs, vocab_vecs, "aff", 20)

vecs = np.vstack(dist.values())
sim = embedding.earth_mover_distance(dist)
G = embedding.graph_from_sim(sim, 0.75)
link = embedding.create_linkage(vecs, dense=True)


# VIS
vis.simMatrixIntersection(sim, used)
vis.scree_plot(truth, vecs, dense=True)
vis.graph(G)
vis.cluster_heatmap(link, used, mode="intersection")

G_link = embedding.graph_from_sim(embedding.link_sim(link, len(sim)-1), 0.5)
vis.graph(G_link)

vis.scatter_tsne(vecs, labels)
vis.scatter_mds(vecs, labels)
vis.scatter_svd(vecs, labels)

texts[14]

# Reduced VIS
reduced = embedding.reduced(vecs, "pca", 6)
reduced_sim = cosine_similarity(reduced)
vis.simMatrixIntersection(reduced_sim, used)
G = embedding.graph_from_sim(reduced_sim, 0.75)
vis.graph(G)
reduced_link = embedding.create_linkage(reduced, dense=True)
vis.cluster_heatmap(reduced_link, used, mode="intersection")

reduced_G_link = embedding.graph_from_sim(embedding.link_sim(reduced_link, len(sim)-1), 0.5)
vis.graph(reduced_G_link)

vis.scatter_tsne(reduced, labels)
vis.scatter_mds(reduced, labels)
vis.scatter_svd(reduced, labels)

texts[14]
