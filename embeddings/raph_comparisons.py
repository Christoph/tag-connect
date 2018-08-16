from importlib import reload

import lib.forchristoph as forchristoph
import lib.path as path
import lib.exam as exam

import lib.embedding as embedding
import lib.details as details
import lib.helpers as helpers
import lib.vis as vis

import numpy as np

# LOAD DATA
vecs = np.load('../datasets/paths.npy')
# dist = np.load('../datasets/raphdist.npy')

reload(embedding)
reload(forchristoph)
# reload(vis)

# Similarity simMatrix
metric = "raph_earth" # cosine, emd, cm, raph_raph, raph_earth, raph_energy

paths = np.array(vecs).reshape(-1,1)
dist = embedding.similarity_matrix(paths, metric, scaled=False)

dist = (dist - dist.min()) / (dist.max() - dist.min())
sim = 1 - dist

# VIS
vis.simMatrix(dist)
vis.graph(embedding.graph_from_sim(sim, 4))
vis.graph(embedding.graph_from_sim(sim, sim.mean()))
vis.cluster_heatmap(
    vecs,
    metric=metric,
    mode="intersection",
    order=True)
