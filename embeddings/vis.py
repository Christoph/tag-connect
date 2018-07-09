from plotly import offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import TruncatedSVD, PCA, NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import Normalizer
import networkx as nx

py.init_notebook_mode()

def cluster_document(predictions, vocab_labels):
    clusters = np.unique(list(vocab_labels.values()))
    n_documents = len(predictions)
    x = list(range(0, n_documents))
    traces = []
    data = np.zeros([n_documents, len(clusters)])

    if len(predictions[next(iter(predictions))]) == len(clusters):
        for doc_id, labels in predictions.items():
            data[int(doc_id), :] = labels
    else:
        for doc_id, labels in predictions.items():
            unique, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique, counts))

            for i in clusters:
                if i in label_counts:
                    data[int(doc_id), i] = label_counts[i]
                else:
                    data[int(doc_id), i] = 0

    for i in range(0, len(clusters)):
        traces.append(go.Bar(
            x=x,
            y=data[:,i],
            name=clusters[i],
            text=data[:,i],
            textposition = 'auto',
        ))

    fig = tools.make_subplots(rows=len(clusters), cols=1,
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=0.01)

    for i in range(1, len(traces)+1):
        fig.append_trace(traces[i-1], i, 1)
        fig['layout']['yaxis'+str(i)].update(title=i-1, range=[0, data.max().max()])

    fig['layout'].update(height=len(clusters)*100, width=600)
    py.iplot(fig)

def histogram(label):
    data = [go.Histogram(
        x=label,
        xbins=dict(
            start=0,
            end=10,
            size=1
        ),
        )]
    py.iplot(data)

def multi_histogram(prediction, unique=False):
    data = []
    n_bins = len(np.unique(prediction[next(iter(prediction))]))

    for l in prediction.keys():
        if l is not "all":
            data.append(go.Histogram(
                x=np.unique(prediction[l]) if unique else prediction[l],
                opacity=0.75,
                name=l,
                xbins=dict(
                    start=0,
                    end=n_bins,
                    size=1
                ),
            ))

    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)

def graph(G):
    pos=nx.spring_layout(G)
    nx.set_node_attributes(G, name="pos", values=pos)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='bottom center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['marker']['color'].append(G.degree(node))
        node_trace['text'].append(str(node))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename='networkx')

def cluster_heatmap(link, docs, sim=None, mode="intersection"):
    labels = np.arange(0, len(docs))

    # Initialize figure by creating upper dendrogram
    figure = ff.create_dendrogram(link, orientation='bottom', labels=labels)
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(link, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    figure['data'].extend(dendro_side['data'])

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))

    if sim is None:
        data_dist = pdist(link)
        data_dist /= data_dist.max()
        sim = 1 - squareform(data_dist)
        sim = sim[dendro_leaves,:]
        sim = sim[:,dendro_leaves]

    hovertext = list()

    for i in range(0, len(sim)):
        hovertext.append(list())

        for j in range(0, len(sim)):
            if mode == "intersection":
                elements = list(l + '<br>' * (n % 10 == 9) for n, l in enumerate(np.intersect1d(docs[i].split(" "), docs[j].split(" "))))
            elif mode == "difference":
                elements = list(l + '<br>' * (n % 10 == 9) for n, l in enumerate(np.setdiff1d(docs[i].split(" "), docs[j].split(" "))))

            hovertext[i].append(" ".join(elements))

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = sim,
            colorscale=[
                [0, 'rgb(255,255,204)'],
                [0.2, 'rgb(255,255,204)'],

                [0.2, 'rgb(161,218,180)'],
                [0.4, 'rgb(161,218,180)'],

                [0.4, 'rgb(65,182,196)'],
                [0.6, 'rgb(65,182,196)'],

                [0.6, 'rgb(44,127,184)'],
                [0.8, 'rgb(44,127,184)'],

                [0.8, 'rgb(37,52,148)'],
                [1.0, 'rgb(37,52,148)']
            ],
            text=hovertext
        )
    ]

    heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    figure['data'].extend(heatmap)

    # Edit Layout
    figure['layout'].update({'width':600, 'height':600,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    # Edit xaxis
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'ticks':""})
    # Edit xaxis2
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    # Edit yaxis
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
    # Edit yaxis2
    figure['layout'].update({'yaxis2':{'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    # Plot!
    py.iplot(figure)

def set_histogram(inter, w_inter, diff1, w_diff1, diff2, w_diff2):
    data = []
    x = list(range(0, len(inter)))

    t_inter = ['<br>'.join(text[i:i+80] for i in range(0, len(text), 80)) for text in w_inter]
    t_diff1 = ['<br>'.join(text[i:i+80] for i in range(0, len(text), 80)) for text in w_diff1]
    t_diff2 = ['<br>'.join(text[i:i+80] for i in range(0, len(text), 80)) for text in w_diff2]

    data.append(go.Bar(
        x=x,
        y=inter,
        name="Intersection",
        text=t_inter
    ))

    data.append(go.Bar(
        x=x,
        y=diff1,
        name="C1.difference(C2)",
        text=t_diff1
    ))

    data.append(go.Bar(
        x=x,
        y=diff2,
        name="C2.difference(C1)",
        text=t_diff2
    ))

    layout = go.Layout(barmode='stack')
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)

def scree_plot(truth, vecs, maxdim=300, dense=False, nonlinear=False, uselda=True):
    n_dims = len(vecs[0]) if dense else vecs.shape[1]

    if n_dims > 50:
        dimensions = list(range(1, 50)) + list(range(50, min(maxdim, n_dims), 10))
    else:
        dimensions = list(range(1, min(n_dims, maxdim)))

    metrics = {}

    for d in dimensions:
        if "svd" not in metrics:
            metrics["svd"] = []
        else:
            svd = TruncatedSVD(d).fit_transform(vecs)
            svd_sim = cosine_similarity(svd)
            svd_diff = np.absolute(truth - svd_sim)

            metrics.get("svd").append(1 - np.average(svd_diff))

        if "pca" not in metrics:
            metrics["pca"] = []
        else:
            pca = PCA(d).fit_transform(vecs if dense else vecs.toarray())
            pca_sim = cosine_similarity(pca)
            pca_diff = np.absolute(truth - pca_sim)

            metrics.get("pca").append(1 - np.average(pca_diff))

        if uselda:
            if "lda" not in metrics:
                metrics["lda"] = []
            else:
                lda = LatentDirichletAllocation(d, learning_method="batch").fit_transform(vecs)
                lda_sim = cosine_similarity(lda)
                lda_diff = np.absolute(truth - lda_sim)

                metrics.get("lda").append(1 - np.average(lda_diff))

        if nonlinear:
            if "tsne" not in metrics:
                metrics["tsne"] = []
            else:
                tsne = TSNE(d, method="exact").fit_transform(vecs if dense else vecs.toarray())
                tsne_sim = cosine_similarity(tsne)
                tsne_diff = np.absolute(truth - tsne_sim)

                metrics.get("tsne").append(1 - np.average(tsne_diff))

            if "mds" not in metrics:
                metrics["mds"] = []
            else:
                mds = MDS(d).fit_transform(vecs if dense else vecs.toarray())
                mds_sim = cosine_similarity(mds)
                mds_diff = np.absolute(truth - mds_sim)

                metrics.get("mds").append(1 - np.average(mds_diff))

    data = []
    for name, y in metrics.items():
        data.append(go.Scatter(
            x=dimensions,
            y=y,
            name=name
        ))

    layout = dict(
        title = 'Average similarity between two matrices whereas the second one is dimension reduced',
        xaxis = dict(title = 'Dimension'),
        yaxis = dict(title = 'Average similarity'),
        )

    fig = dict(data=data, layout=layout)

    py.iplot(fig)


def scatter_tsne(vecs, labels):
    # if isinstance(vecs, np.ndarray):
    #     reduced = TSNE().fit_transform(vecs.toarray())
    if sp.issparse(vecs):
        reduced = TSNE().fit_transform(vecs.todense())
    else:
        reduced = TSNE().fit_transform(vecs)

    ids = list(range(0, len(reduced)))

    trace = go.Scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode='markers',
        text=ids,
        marker=dict(
            size=14,
            color=labels
            )
    )

    data = [trace]
    py.iplot(data)

def scatter_mds(vecs, labels):
    if sp.issparse(vecs):
        reduced = MDS().fit_transform(vecs.todense())
    else:
        reduced = MDS().fit_transform(vecs)

    ids = list(range(0, len(reduced)))

    trace = go.Scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode='markers',
        text=ids,
        marker=dict(
            size=14,
            color=labels
            )
    )

    data = [trace]
    py.iplot(data)

def scatter_svd(vecs, labels):
    reduced = TruncatedSVD().fit_transform(vecs)
    ids = list(range(0, len(reduced)))

    trace = go.Scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode='markers',
        text=ids,
        marker=dict(
            size=14,
            color=labels
            )
    )

    data = [trace]
    py.iplot(data)

def simMatrix(matrix):
    trace = go.Heatmap(
        z=matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, 'rgb(255,255,204)'],
            [0.2, 'rgb(255,255,204)'],

            [0.2, 'rgb(161,218,180)'],
            [0.4, 'rgb(161,218,180)'],

            [0.4, 'rgb(65,182,196)'],
            [0.6, 'rgb(65,182,196)'],

            [0.6, 'rgb(44,127,184)'],
            [0.8, 'rgb(44,127,184)'],

            [0.8, 'rgb(37,52,148)'],
            [1.0, 'rgb(37,52,148)']
        ],
        )
    data=[trace]
    py.iplot(data)

def simMatrixIntersection(matrix, docs):
    hovertext = list()

    for i in range(0, len(matrix)):
        hovertext.append(list())

        for j in range(0, len(matrix)):
            elements = list(l + '<br>' * (n % 10 == 9) for n, l in enumerate(np.intersect1d(docs[i].split(" "), docs[j].split(" "))))
            hovertext[i].append(" ".join(elements))

    trace = go.Heatmap(
        z=matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, 'rgb(255,255,204)'],
            [0.2, 'rgb(255,255,204)'],

            [0.2, 'rgb(161,218,180)'],
            [0.4, 'rgb(161,218,180)'],

            [0.4, 'rgb(65,182,196)'],
            [0.6, 'rgb(65,182,196)'],

            [0.6, 'rgb(44,127,184)'],
            [0.8, 'rgb(44,127,184)'],

            [0.8, 'rgb(37,52,148)'],
            [1.0, 'rgb(37,52,148)']
        ],
        text=hovertext
        )
    data=[trace]
    fig = go.Figure(data=data)
    fig['layout'].update({'width':600, 'height':600,
                             'showlegend':False, 'hovermode': 'closest',
                             })

    py.iplot(fig)

def simMatrixDifference(matrix, docs):
    hovertext = list()

    for i in range(0, len(matrix)):
        hovertext.append(list())

        for j in range(0, len(matrix)):
            elements = list(l + '<br>' * (n % 10 == 9) for n, l in enumerate(np.setdiff1d(docs[i].split(" "), docs[j].split(" "))))
            hovertext[i].append(" ".join(elements))

    trace = go.Heatmap(
        z=matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, 'rgb(255,255,204)'],
            [0.2, 'rgb(255,255,204)'],

            [0.2, 'rgb(161,218,180)'],
            [0.4, 'rgb(161,218,180)'],

            [0.4, 'rgb(65,182,196)'],
            [0.6, 'rgb(65,182,196)'],

            [0.6, 'rgb(44,127,184)'],
            [0.8, 'rgb(44,127,184)'],

            [0.8, 'rgb(37,52,148)'],
            [1.0, 'rgb(37,52,148)']
        ],
        text=hovertext
        )
    data=[trace]

    fig = go.Figure(data=data)
    fig['layout'].update({'width':600, 'height':600,
                             'showlegend':False, 'hovermode': 'closest',
                             })

    py.iplot(fig)

def simMatrixSensitive(matrix):
    trace = go.Heatmap(
        z=matrix,
        zmin=0,
        zmax=1,
        colorscale=[
            [0, 'rgb(180, 180, 180)'],
            [0.1, 'rgb(180, 180, 180)'],

            [0.1, 'rgb(160, 160, 160)'],
            [0.2, 'rgb(160, 160, 160)'],

            [0.2, 'rgb(140, 140, 140)'],
            [0.3, 'rgb(140, 140, 140)'],

            [0.3, 'rgb(120, 120, 120)'],
            [0.4, 'rgb(120, 120, 120)'],

            [0.4, 'rgb(100, 100, 100)'],
            [0.5, 'rgb(100, 100, 100)'],

            [0.5, 'rgb(80, 80, 80)'],
            [0.6, 'rgb(80, 80, 80)'],

            [0.6, 'rgb(60, 60, 60)'],
            [0.7, 'rgb(60, 60, 60)'],

            [0.7, 'rgb(40, 40, 40)'],
            [0.8, 'rgb(40, 40, 40)'],

            [0.8, 'rgb(20, 20, 20)'],
            [0.9, 'rgb(20, 20, 20)'],

            [0.9, 'rgb(0, 0, 0)'],
            [1.0, 'rgb(0, 0, 0)']
        ],
        )
    data=[trace]
    py.iplot(data)
