import networkx as nx
import numpy as np
import spacy
from scipy.sparse.csr import csr_matrix
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import (cophenet, fcluster, leaves_list, linkage)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.cluster import (DBSCAN, AffinityPropagation, Birch, KMeans)
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

import hal_embedding
import forchristoph

from importlib import reload

reload(forchristoph)

nlp = spacy.load('en')

def count(data):
    vectorizer = CountVectorizer()
    vecs = vectorizer.fit_transform(data)

    norm_vecs = Normalizer().fit_transform(vecs)
    return norm_vecs, vectorizer

def tfidf(data):
    def space_tokenizer(s):
        return s.split(' ')

    vectorizer = TfidfVectorizer(tokenizer=space_tokenizer)
    doc_vecs = vectorizer.fit_transform(data)

    norm_vecs = Normalizer().fit_transform(doc_vecs)
    return norm_vecs, vectorizer

def lda(data, components=10):
    vectorizer = LatentDirichletAllocation(components, learning_method="batch")
    doc_vecs = vectorizer.fit_transform(data)

    # norm_vecs = Normalizer().fit_transform(doc_vecs)
    return doc_vecs, vectorizer

def reduced(vecs, type, dim):
    if type == "svd":
        reduced = TruncatedSVD(dim).fit_transform(vecs)
    elif type == "pca":
        reduced = PCA(dim).fit_transform(vecs)
    elif type == "nmf_fro":
        reduced = NMF(dim, beta_loss='frobenius').fit_transform(vecs)
    elif type == "nmf_kl":
        reduced = NMF(dim, beta_loss="kullback-leibler", solver="mu").fit_transform(vecs)
    elif type == "lda":
        reduced = LatentDirichletAllocation(dim).fit_transform(vecs)

    return reduced

def create_linkage(vecs, metric="cosine", order=True):
    link = linkfun(vecs, metric, order)

    c, coph_dists = cophenet(link, pdist(vecs, metric))
    print("Cophenet Distance between linkage and original vecs: "+str(c))

    return link

def linkfun(vecs, metric="cosine", order=True):
    return linkage(vecs, method='weighted', metric=dist_func(metric), optimal_ordering=order)

def link_sim(vecs, leaves, metric="cosine"):
    sim = similarity_matrix(vecs, metric=metric)
    sim = sim[leaves, :]
    sim = sim[:, leaves]

    return sim

def high_space_binning(word_vecs, vocab_vecs, clustering="birch", bins=16, vocab=None, docs=None):
    hasLabels = True
    cluster = None
    if clustering == "birch":
        cluster = Birch(n_clusters=bins).fit(vocab_vecs)
        doc_labels = {i: cluster.predict(d) for i, d in enumerate(word_vecs)}
    elif clustering == "km":
        cluster = KMeans(bins).fit(vocab_vecs)
        doc_labels = {i: cluster.predict(d) for i, d in enumerate(word_vecs)}
    elif clustering == "gm":
        cluster = GaussianMixture(bins).fit(vocab_vecs)
        doc_labels = {i: cluster.predict(d) for i, d in enumerate(word_vecs)}
        clusters = np.arange(0, bins)
        hasLabels = False
    elif clustering == "aff":
        cluster = AffinityPropagation().fit(vocab_vecs)
        doc_labels = {i: cluster.predict(d) for i, d in enumerate(word_vecs)}
        print("# labels: "+str(len(np.unique(cluster.labels_))))
    elif clustering == "hdbs":
        cluster = HDBSCAN(metric="sqeuclidean", min_cluster_size=2, min_samples=None).fit(vocab_vecs)
        doc_labels = {i: cluster.fit_predict(d) for i, d in enumerate(word_vecs)}
        print("# labels: "+str(len(np.unique(cluster.labels_))))
    elif clustering == "maxclust":
        link = linkage(vocab_vecs, method='weighted', metric="cosine", optimal_ordering=True)
        clusters = fcluster(link, bins, criterion='maxclust')
        hasLabels = False

        mapping = dict(zip([w.text for w in vocab], clusters))

        doc_labels = {i: [mapping[w]-1 for w in d.split(" ")] for i, d in enumerate(docs)}

    u_labels = np.unique(cluster.labels_) if hasLabels else np.unique(clusters)-1
    bins = len(u_labels)
    norm_labels = {}

    for doc_id, labels in doc_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))
        temp = np.zeros(bins)

        for i in u_labels:
            if i in label_counts:
                temp[i] = label_counts[i]
            else:
                temp[i] = 0

        norm_labels[doc_id] = ((temp - temp.min()) / (temp - temp.min()).sum())

    vecs = np.vstack(norm_labels.values())

    return doc_labels, norm_labels, vecs, cluster

def dist_func(metric):
    if metric == "cosine":
        return "cosine"
    elif metric == "emd":
        return wasserstein_distance
    elif metric == "cm":
        return energy_distance
    elif metric == "raph_raph":
        return forchristoph.raphRaphDistFun

def similarity_matrix(vecs, metric="cosine", as_distance=True, scaled=True):
    if isinstance(vecs, csr_matrix):
        checked = vecs.todense()
    else:
        checked = vecs

    if metric == "cosine":
        data_dist = pdist(checked, "cosine")
    elif metric == "emd":
        data_dist = pdist(checked, wasserstein_distance)
    elif metric == "cm":
        data_dist = pdist(checked, energy_distance)
    elif metric == "jaccard":
        data_dist = pdist(checked, lambda x, y: jaccard(x, y))
    elif metric == "raph_raph":
        data_dist = pdist(checked, lambda x, y: forchristoph.raphRaphDistFun(x[0], y[0]))
    elif metric == "raph_earth":
        data_dist = pdist(checked, lambda x, y: forchristoph.raphEarthDistFun(x[0], y[0]))
    elif metric == "raph_energy":
        data_dist = pdist(checked, lambda x, y: forchristoph.raphEnergyDistFun(x[0], y[0]))

    if scaled:
        data_dist /= data_dist.max()

    if as_distance:
        sim = squareform(data_dist)
    else:
        sim = 1 - squareform(data_dist)

    return sim

def jaccard(d1, d2):
    s1 = set([w for w in d1[0]])
    s2 = set([w for w in d2[0]])

    dist = len(s1.intersection(s2))/len(s1.union(s2))

    print(dist)

    return 1 - dist

def embedding_vector(embeddings, metric, probabilistic=False):
    out = []

    for emb in embeddings:
        D = similarity_matrix(emb, metric=metric, as_distance=True, scaled=True)
        vec = np.mean(D, axis=0)

        if probabilistic:
            vec = (vec - vec.min()) / (vec - vec.min()).sum()
            vec /= vec.sum()

        out.append(vec)

    return np.array(out)

def embedding_vector_radius(embeddings, metric, radius=0.1, probabilistic=True):
    out = []

    for emb in embeddings:
        D = similarity_matrix(emb, metric=metric, as_distance=True, scaled=True)
        vec = (D <= radius).sum(axis = 0) - 1

        if probabilistic:
            vec = (vec - vec.min()) / (vec.max() - vec.min())
            vec /= vec.sum()

        out.append(vec)

    return np.array(out)

def set_vectors(documents):
    docs = []

    for document in documents:
        doc = np.array(document.split(" "))

        docs.append(doc)

    docs = np.array(docs).reshape(-1,1)

    return docs

def graph_from_sim(sim, value):
    mask = np.copy(sim)
    mask[mask < value] = 0
    G = nx.from_numpy_matrix(mask)

    return G

def HAL(data, reduce=False):
    word_vecs, word_vocab = hal_embedding.HAL(data)

    if reduce:
        word_vecs = TruncatedSVD(300).fit_transform(word_vecs)

    hal_vecs = [[word_vecs[word_vocab[w]] for w in doc.split(" ")] for doc in data]
    hal_docs = [[w for w in doc.split(" ")] for doc in data]

    vocab = list(word_vocab.keys())
    vocab_vecs = [word_vecs[word_vocab[w]] for w in vocab]

    return hal_vecs, hal_docs, vocab_vecs, vocab

def W2V(data):
    docs = [nlp(t) for t in data]
    w2v_vecs = [[w.vector for w in doc] for doc in docs]
    w2v_docs = [[w.text for w in doc] for doc in docs]

    vocab = np.unique([item for sublist in docs for item in sublist])
    vocab_vecs = [w.vector for w in vocab]

    return w2v_vecs, w2v_docs, vocab_vecs, vocab

def average_word_vectors(data):
    docs = [nlp(t) for t in data]
    w2v_vecs = [doc.vector for doc in docs]

    return w2v_vecs

def tfidf_weighted_average_word_vectors(data, vectorizer):
    docs = [nlp(t) for t in data]
    w2v_vecs = [[w.vector for w in doc] for doc in docs]
    w2v_weights = [[np.count_nonzero(np.array(doc.text.split(" ")) == w.text) * vectorizer.idf_[vectorizer.vocabulary_.get(w.text)] for w in doc] for doc in docs]

    average = []
    for i in range(0, len(w2v_vecs)):
        average.append(np.average(w2v_vecs[i], axis=0, weights=w2v_weights[i]))

    norm = Normalizer().fit_transform(average)
    return norm

def count_weighted_average_word_vectors(data):
    docs = [nlp(t) for t in data]
    w2v_vecs = [[w.vector for w in doc] for doc in docs]
    w2v_weights = [[np.count_nonzero(np.array(doc.text.split(" ")) == w.text) for w in doc] for doc in docs]

    average = []
    for i in range(0, len(w2v_vecs)):
        average.append(np.average(w2v_vecs[i], axis=0, weights=w2v_weights[i]))

    norm = Normalizer().fit_transform(average)
    return norm
