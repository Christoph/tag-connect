from sklearn.cluster import KMeans, SpectralClustering, Birch, AffinityPropagation, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import numpy as np
from hdbscan import HDBSCAN



def prepare_label_data(labels, vecs, docs):
    u_labels = np.unique(labels)
    classes = {}

    classes["all"] = {
        "vecs": [item for sublist in vecs for item in sublist],
        "docs": [item for sublist in docs for item in sublist]
        }

    for l in u_labels:
        c_vecs = [item for sublist in np.array(vecs)[labels == l] for item in sublist]
        c_docs = [item for sublist in np.array(docs)[labels == l] for item in sublist]

        classes[l] = { "vecs": c_vecs, "docs": c_docs }

    return classes

def prepare_data(vecs, docs):
    classes = {}

    for i in range(0, len(vecs)):
        classes[str(i)] = { "vecs": vecs[i], "docs": docs[i] }

    return classes

def prepare_tfidf_data(vecs, docs):
    classes = {}

    for i in range(0, len(vecs)):
        classes[str(i)] = { "vecs": vecs[i], "docs": docs[i] }

    return classes

def cluster_space(classes, vocab_vecs, vocab, bins = 16, algorithm = "agglo"):
    if algorithm == "gauss":
        cl = GaussianMixture(bins).fit(np.array(vocab_vecs))
    elif algorithm == "km":
        cl = KMeans(bins, random_state=1).fit(vocab_vecs)
    elif algorithm == "agglo":
        cl = Birch(n_clusters=bins).fit(vocab_vecs)
    elif algorithm == "aff":
        cl = AffinityPropagation().fit(vocab_vecs)

    c_labels = {}
    vocab_labels = {}
    norm_labels = {}

    for i in range(0, len(vocab)):
        vocab_labels[vocab[i].text] = cl.labels_[i]

    for l in classes.keys():
        c_labels[l] = cl.predict(classes[l]["vecs"])

    clusters = np.unique(list(vocab_labels.values()))
    n_documents = len(c_labels)
    dist_labels = np.zeros([n_documents, len(clusters)])

    for doc_id, labels in c_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))

        for i in clusters:
            if i in label_counts:
                dist_labels[int(doc_id), i] = label_counts[i]
            else:
                dist_labels[int(doc_id), i] = 0

        norm_labels[doc_id] = ((dist_labels[int(doc_id), :] - dist_labels[int(doc_id), :].min()) / (dist_labels[int(doc_id), :] - dist_labels[int(doc_id), :].min()).sum())

    return c_labels, norm_labels, vocab_labels

def cluster_space_hdb(classes, vocab_vecs, vocab, min_cluster_size = 5, metric="sqeuclidean", min_samples=None):
    cl = HDBSCAN(metric=metric, min_cluster_size=min_cluster_size, min_samples=min_samples).fit(vocab_vecs)

    c_labels = {}
    vocab_labels = {}

    for i in range(0, len(vocab)):
        vocab_labels[vocab[i].text] = cl.labels_[i]

    for l in classes.keys():
        c_labels[l] = cl.fit_predict(classes[l]["vecs"])

    return c_labels, vocab_labels

# def set_histogram(prediction, focus_bar = "0"):
#     bins = np.unique(prediction["all"])
#     n_bins = len(bins)
#     hist_data = {}
#     keys = []
#
#     for c in prediction.keys():
#         if c == "all":
#             hist_data["intersection"] = {
#                 "counts": np.zeros(n_bins),
#                 "words": []
#             }
#             keys.append("intersection")
#         elif c != focus_bar:
#             hist_data[focus_bar+" difference("+c+")"] = {
#                 "counts": np.zeros(n_bins),
#                 "words": []
#             }
#             keys.append(focus_bar+" difference("+c+")")
#
#     for b in bins:
#         inter = set(np.array(c2_words)[l2 == i]).intersection(np.array(c1_words)[l1 == i])
#         diff1 = set(np.array(c1_words)[l1 == i]).difference(np.array(c2_words)[l2 == i])
#         diff2 = set(np.array(c2_words)[l2 == i]).difference(np.array(c1_words)[l1 == i])
#
#         w_int.append(", ".join(inter))
#         w_diff1.append(", ".join(diff1))
#         w_diff2.append(", ".join(diff2))
#
#         for w in c_all_words:
#             if w in inter:
#                 l_int[i] += 1
#             if w in diff1:
#                 l_diff1[i] += 1
#             if w in diff2:
#                 l_diff2[i] += 1
