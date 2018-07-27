from importlib import reload

import forchristoph
import embedding
import details
import helpers
import vis
import path
import exam

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
