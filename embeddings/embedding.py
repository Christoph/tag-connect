import hal_embedding
import numpy as np
import spacy
from hdbscan import HDBSCAN
from scipy.stats import entropy, wasserstein_distance
from sklearn.cluster import (DBSCAN, AffinityPropagation, Birch, KMeans,
                             SpectralClustering)
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
import networkx as nx

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

def create_linkage(vecs, dense=False):
    link = linkage(vecs if dense else vecs.toarray(), method='weighted', metric="cosine", optimal_ordering=True)

    c, coph_dists = cophenet(link, pdist(vecs))
    print("Cophenet Distance between linkage and original vecs: "+str(c))

    return link

def high_space_binning(word_vecs, vocab_vecs, clustering="birch", bins=16, vocab=None, docs=None):
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
    elif clustering == "aff":
        cluster = AffinityPropagation().fit(vocab_vecs)
        doc_labels = {i: cluster.predict(d) for i, d in enumerate(word_vecs)}
    elif clustering == "hdbs":
        cluster = HDBSCAN(metric="sqeuclidean", min_cluster_size=3, min_samples=None).fit(vocab_vecs)
        doc_labels = {i: cluster.fit_predict(d) for i, d in enumerate(word_vecs)}
    elif clustering == "maxclust":
        link = linkage(vocab_vecs, method='weighted', metric="cosine", optimal_ordering=True)
        clusters = fcluster(link, bins, criterion='maxclust')

        mapping = dict(zip([w.text for w in vocab], clusters))

        doc_labels = {i: [mapping[w]-1 for w in d.split(" ")] for i, d in enumerate(docs)}

    u_labels = np.unique(cluster.labels_) if cluster else np.unique(clusters)-1
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

    return doc_labels, norm_labels, cluster

def earth_mover_distance(dist):
    emd_matrix = [[wasserstein_distance(a, b) for b in dist.values()] for a in dist.values()]
    sim = 1-emd_matrix/max(max(emd_matrix))

    return sim

def graph_from_sim(sim, value):
    mask = np.copy(sim)
    mask[mask < value] = 0
    G=nx.from_numpy_matrix(mask)

    return G

def link_sim(link, dim):
    data_dist = pdist(link)
    data_dist /= data_dist.max()
    sim = 1 - squareform(data_dist)
    print(len(sim))
    sim = sim[np.arange(0, dim),:]
    sim = sim[:,np.arange(0, dim)]

    return sim

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
