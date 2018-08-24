from importlib import reload

from hdbscan import HDBSCAN
from sklearn.cluster import (DBSCAN, SpectralClustering, KMeans)

from sklearn import metrics

import methods.forchristoph as forchristoph

import methods.embedding as embedding
import methods.details as details
import methods.helpers as helpers
import methods.vis as vis
import path
import exam
import numpy as np

# LOAD DATA
# vecs = np.load('../datasets/paths.npy')
labels = np.load('../datasets/labels.npy')
en_dist = np.load('../datasets/endist.npy')
em_dist = np.load('../datasets/emdist.npy')
dlev_dist = np.load('../datasets/dlevdist.npy')
raph_dist = np.load('../datasets/raphdist.npy')
# dist = np.load('../datasets/raphdist.npy')

reload(embedding)
reload(forchristoph)
# reload(vis)

# Similarity simMatrix
metric = "raph_earth"  # cosine, emd, cm, raph_raph, raph_earth, raph_energy

paths = np.array(vecs).reshape(-1, 1)
dist = embedding.similarity_matrix(paths, metric, scaled=False)



ll = [int(l) for l in labels]

en_dist = (en_dist - en_dist.min()) / (en_dist.max() - en_dist.min())
em_dist = (em_dist - em_dist.min()) / (em_dist.max() - em_dist.min())
raph_dist = (raph_dist - raph_dist.min()) / (raph_dist.max() - raph_dist.min())
# sim = 1 - dist

can = DBSCAN().fit_predict(raph_dist)
metrics.adjusted_rand_score(can, ll)

# VIS
vis.simMatrix(raph_dist)
# vis.graph(embedding.graph_from_sim(sim, 4))
# vis.graph(embedding.graph_from_sim(sim, sim.mean()))
vis.cluster_heatmap(
    paths,
    metric=metric,
    mode="intersection",
    order=True)
