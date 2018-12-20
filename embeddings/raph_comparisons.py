from importlib import reload

from hdbscan import HDBSCAN
from sklearn.cluster import (DBSCAN, SpectralClustering, KMeans)
import numpy as np

from sklearn import metrics

from os import path
import sys
sys.path.append(path.abspath('../methods'))

import embedding
import details
import helpers
import vis
import forchristoph
import path
import exam


# LOAD DATA
# vecs = np.load('../datasets/paths.npy')
labels = np.load('../datasets/labels.npy')
en_dist = np.load('../datasets/endist.npy')
em_dist = np.load('../datasets/emdist.npy')
dlev_dist = np.load('../datasets/dlevdist.npy')
raph_dist = np.load('../datasets/raphdist.npy')
raphim_dist = np.load('../datasets/raphimdist.npy')
# dist = np.load('../datasets/raphdist.npy')

reload(embedding)
reload(forchristoph)
# reload(vis)

# Similarity simMatrix
metric = "raph_earth"  # cosine, emd, cm, raph_raph, raph_earth, raph_energy

# paths = np.array(vecs).reshape(-1, 1)
# dist = embedding.similarity_matrix(paths, metric, scaled=False)



ll = [int(l) for l in labels]

en_dist = (en_dist - en_dist.min()) / (en_dist.max() - en_dist.min())
em_dist = (em_dist - em_dist.min()) / (em_dist.max() - em_dist.min())
# raph_dist = (raph_dist - raph_dist.min()) / (raph_dist.max() - raph_dist.min())
# raphim_dist = (raphim_dist - raphim_dist.min()) / (raphim_dist.max() - raphim_dist.min())
# sim = 1 - dist

can = DBSCAN().fit_predict(raphim_dist)
metrics.adjusted_rand_score(can, ll)

# VIS
vis.simMatrixSensitive(raph_dist)
vis.simMatrix(raphim_dist)
# vis.graph(embedding.graph_from_sim(sim, 4))
# vis.graph(embedding.graph_from_sim(sim, sim.mean()))
vis.cluster_heatmap(
    paths,
    metric=metric,
    mode="intersection",
    order=True)
