from importlib import reload
import spacy
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import numpy as np

import embedding
import vis
import helpers
import cluster_analysis
# nlp = spacy.load('en')

# Load data
(texts, clean, clean_fancy, labels) = helpers.load_reuters_data()

used = clean_fancy
used = clean

len(clean[0])
len(clean_fancy[0])

reload(vis)

# Word vectors
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.HAL(used, True)
(word_vecs, word_docs, vocab_vecs, vocab) = embedding.W2V(used)

# High dimensional clustering dependent document vectors
(cluster_labels, dist, clusterer) = embedding.hsc_auto(word_vecs, vocab_vecs)
(cluster_labels, dist, clusterer) = embedding.hsc_binned(word_vecs, vocab_vecs, "birch", 16)
vis.simMatrixIntersection(embedding.earth_mover_distance(dist), used)

# Document vectors
(vecs, vectorizer) = embedding.count(used)
(vecs, vectorizer) = embedding.tfidf(used)

reduced = TruncatedSVD(20).fit_transform(vecs)
reduced = PCA(3).fit_transform(vecs.toarray())
reduced = NMF(10, beta_loss='frobenius').fit_transform(vecs)
reduced = NMF(50, beta_loss="kullback-leibler", solver="mu").fit_transform(vecs)
reduced = LatentDirichletAllocation(5).fit_transform(vecs)

vecs = reduced
sim = cosine_similarity(vecs)

vis.simMatrixIntersection(sim, used)
vis.scree_plot(sim, vecs, nonlinear=False)
vis.scatter_tsne(vecs, labels)
vis.scatter_mds(vecs, labels)
vis.scatter_svd(vecs, labels)

# Comparing description axis
axis_words = helpers.get_dimension_words(vectorizer, vecs, reduced)
print(axis_words)

reload(cluster_analysis)
reload(vis)
# Cluster analysis
classes = cluster_analysis.prepare_label_data(labels, word_vecs, word_docs)
documents = cluster_analysis.prepare_data(word_vecs, word_docs)
tfidf_documents = cluster_analysis.prepare_tfidf_data(vecs, used)


# Algorithms: agglo, km, gauss, aff
# predictions, vocab_labels = cluster_analysis.cluster_space(classes, vocab_vecs, vocab, bins=16, algorithm="aff")
predictions, dist_predictions, vocab_labels = cluster_analysis.cluster_space(documents, vocab_vecs, vocab, bins=16, algorithm="agglo")
predictions, vocab_labels = cluster_analysis.cluster_space_hdb(classes, vocab_vecs, vocab)

vis.cluster_document(dist_predictions, vocab_labels)
vis.multi_histogram(predictions)

sim = embedding.earth_mover_distance(dist_predictions)

wasserstein_distance(predictions["13"], predictions["14"])
[v[0] for v in vocab_labels.items() if v[1] == 14]
set([v[0] for v in vocab_labels.items() if v[1] == 13]).intersection(
    [v[0] for v in vocab_labels.items() if v[1] == 14])
