''' Class which converts raw text into embeddings. '''

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix, hstack


def HAL(docs, window_size=5):
    '''
        Creating HAL space out of the text documents.

        docs = document list
        window_size: window size

        returns: HAL embedding
    '''

    col = []
    data = []
    row = []

    vocabulary = {}

    for text in docs:
        blob = TextBlob(text)

        # Tokenize the text
        toks = blob.words

        for ind in range(0, len(toks)):
            # Add focus word to vocab
            row_index = vocabulary.setdefault(toks[ind], len(vocabulary))

            # looking backwards
            for i in range(-window_size + ind, ind):
                if i >= 0:
                    term = toks[i]
                    value = i - ind + window_size + 1

                    index = vocabulary.setdefault(term, len(vocabulary))

                    col.append(row_index)
                    row.append(index)
                    data.append(value)

            # looking forward
            for i in range(ind + 1, ind + window_size + 1):
                if i < len(toks):
                    term = toks[i]
                    value = abs(i - window_size - ind - 1)

                    index = vocabulary.setdefault(term, len(vocabulary))

                    row.append(row_index)
                    col.append(index)
                    data.append(value)

    # Create sparse matrices
    csr_forward = csr_matrix((data, (row, col)), shape=(
        len(vocabulary), len(vocabulary)), dtype=float)
    csr_backward = csr_forward.transpose()

    # Stack backward and forward matrices
    hal = hstack((csr_backward, csr_forward))

    # Divide by 2 to normalize stacking after stacking
    hal = hal.multiply(0.5)

    # Normalize to [0,1]
    max_value = 1 / hal.max()
    hal = hal.multiply(max_value)

    return hal, vocabulary


def HAL_context_grammar(texts, contexts, nlp, window_size=5):
    '''
        Creating contextual HAL space out of the text documents using POS tags.

        docs: document list
        contexts: Consider only docs containing words from the contexts list
        window_size: window size

        returns: HAL embedding
    '''

    grammer_neighbors = []

    col = []
    data = []
    row = []

    vocabulary = {}

    docs = [nlp(a) for a in texts]

    used_deps = ["acomp", "ccomp", "pcomp", "xcomp", "relcl", "conj"]
    used_tags = ["NN", "NNS", "NNP", "NNPS"]
    used_ent_deps = ["compound", "dobj", "pobj"]

    for doc in docs:
        toks = []

        # Prepare entities
        ents = []

        for e in doc.ents:
            temp = [t.lemma_ for t in e if t.dep_ in used_ent_deps]

            if len(temp) > 1:
                ents.append(" ".join(temp))

        # Get important words around in sentecnes containing context words
        for sent in doc.sents:
            # Check if a context word is within a sentence
            if any(sent.text.lower().find(x) >= 0 for x in contexts):
                for tok in sent:
                    if tok.tag_ in used_tags or tok.dep_ in used_deps:
                        if tok.like_email:
                            toks.append("-EMAIL-")
                        elif tok.like_num:
                            toks.append("-NUMBER-")
                        else:
                            toks.append(tok.lemma_)

        # Fix noun_chunks and entities
        text = " ".join(toks)
        for entity in ents:
            text = text.replace(entity, entity.replace(" ", "-"))

        toks = text.split(" ")

        grammer_neighbors.append(toks)

        # Get surroundings of all context words
        for ind in range(0, len(toks)):
            # Add focus word to vocab
            row_index = vocabulary.setdefault(toks[ind], len(vocabulary))

            # looking backwards
            for i in range(-window_size + ind, ind):
                if i >= 0:
                    term = toks[i]
                    value = i - ind + window_size + 1

                    index = vocabulary.setdefault(term, len(vocabulary))

                    col.append(row_index)
                    row.append(index)
                    data.append(value)

            # looking forward
            for i in range(ind + 1, ind + window_size + 1):
                if i < len(toks):
                    term = toks[i]
                    value = abs(i - window_size - ind - 1)

                    index = vocabulary.setdefault(term, len(vocabulary))

                    row.append(row_index)
                    col.append(index)
                    data.append(value)

    # Create sparse matrices
    csr_forward = csr_matrix(
        (data, (row, col)),
        shape=(len(vocabulary), len(vocabulary)),
        dtype=float)

    csr_backward = csr_forward.transpose()

    # Stack backward and forward matrices
    # hal = hstack((csr_backward, csr_forward))

    # Divide by 2 to normalize stacking after stacking
    # hal = hal.multiply(0.5)

    hal = csr_forward + csr_backward

    # Normalize to [0,1]
    max_value = 1 / hal.max()
    hal = hal.multiply(max_value)

    return hal, vocabulary, grammer_neighbors
