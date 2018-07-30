from nltk.corpus import brown, reuters
import spacy
from spacy.matcher import Matcher
import numpy as np


nlp = spacy.load('en')

def get_dimension_words(tfidf_vectorizer, tfidf_vecs, reduced_vecs):
    axis_words = []

    for dimension in range(0, 10):
        low_indices = np.argsort(tfidf_vecs[np.argmin(reduced_vecs[:,dimension])].todense())[0,-5:].tolist()[0]
        high_indices = np.argsort(tfidf_vecs[np.argmax(reduced_vecs[:,dimension])].todense())[0,-5:].tolist()[0]

        axis_words.append({
            "low": [tfidf_vectorizer.get_feature_names()[index] for index in low_indices],
            "high": [tfidf_vectorizer.get_feature_names()[index] for index in high_indices]
        })

    return axis_words

def get_top_idf_words(data, vectorizer, n_words=5):
    top_words = {}

    for k, v in data.items():
        temp = {}

        if isinstance(v, dict):
            top_words_sub = {}
            for ks, vs in v.items():
                temp_sub = {}

                for ws in vs:
                    idf = vectorizer.idf_[vectorizer.vocabulary_.get(ws)]
                    temp_sub[ws] = idf

                temp_sub_sorted = sorted(temp_sub.items(), key=lambda x: x[1], reverse=True)
                top_words_sub[ks] = temp_sub_sorted[0:n_words]
            temp[k] = top_words_sub
        else:
            for w in v:
                idf = vectorizer.idf_[vectorizer.vocabulary_.get(w)]
                temp[w] = idf

        temp_sorted = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        top_words[k] = temp_sorted[0:n_words]

    return top_words

def load_reuters_data(dataset="reuters"):
    # Raw Texts
    texts = []

    # BROWN CORPUS
    if dataset == "brown":
        # ca is NEWS, cf is LORE, cn ADVENTURE
        brown_ids = ["ca16", "ca12", "ca11", "ca02", "cf21", "cf25", "cf15", "cf06", "cn01", "cn08", "cn13", "cn15"]
        labels = [0,0,0,0,1,1,1,1,2,2,2,2]

        for id in ids:
            texts.append(" ".join(brown.words(fileids=[id])))

    # REUTERS CORPUS
    if dataset == "reuters":
        # 10 documents from "tea" and 10 from "zinc"
        reuters_ids  = ['test/14840', 'test/15198', 'test/15329', 'test/15357', 'test/15540', 'test/15584', 'test/15725', 'test/15737', 'test/16097', 'test/16115', 'test/14882', 'test/16194', 'test/17480', 'test/17486', 'test/17783', 'test/17805', 'test/18337', 'test/18943', 'test/18945', 'test/19431']
        labels = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]

        for id in reuters_ids:
            texts.append(" ".join(reuters.words(fileids=[id])))

    # Reuters corpus many articles
    if dataset == "more":
        rice = reuters.fileids("rice")
        tin = reuters.fileids("tin")
        reuters_ids = rice + tin

        labels = [0 for i in range(0, len(rice))] + [1 for i in range(0, len(tin))]

        for id in reuters_ids:
            texts.append(" ".join(reuters.words(fileids=[id])))

    # texts = [
    #     "Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.",
    #     "I love this sandwich. This is an amazing place! I feel very good about these beers.",
    #     "I can't deal with this. He is my sworn enemy! My boss is horrible."
    #     ]

    # Convert to nlp objects
    docs = [nlp(t) for t in texts]

    # Preprocessing
    clean = []
    used_deps = ["acomp", "ccomp", "pcomp", "xcomp", "relcl", "conj"]
    used_tags = ["NN", "NNS", "NNP", "NNPS"]

    for doc in docs:
        toks = []
        for token in doc:

            # if tok.tag_ in used_tags or tok.dep_ in used_deps:
            #     if tok.like_email:
            #         toks.append("-EMAIL-")
            #     elif tok.like_num:
            #         toks.append("-NUMBER-")
            #     else:
            #         toks.append(tok.lemma_)

            lemma = token.lemma_
            if not token.is_stop and token.is_alpha and lemma != "-PRON-" and len(lemma) > 1:
                toks.append(lemma)

            # if tok.is_alpha:
            #     toks.append(tok.lemma_)

        # Fix noun_chunks and entities
        clean.append(" ".join(toks))

    clean_fancy = []

    matcher = Matcher(nlp.vocab)

    pattern1 = [{'POS': 'ADJ', "OP": "*"}, {'POS': "NOUN", "OP": "+"}, {'POS': "NOUN"}]
    pattern2 = [{'POS': 'ADJ', "OP": "*"}, {'POS': "PROPN", "OP": "+"}, {'POS': "PROPN"}]
    matcher.add('NounChunks', None, pattern1)
    matcher.add('PorperNounChunks', None, pattern2)

    for doc in docs:
        used_tokens = set()
        bag = []

        m = matcher(doc)

        # Go over entities
        for ent in doc.ents:
            # Merging dates
            if ent.label_ in ["DATE", "MONEY"]:
                bag.append(ent.lemma_.replace("-PRON-", "").strip().replace(" ", "-"))

                for token in ent:
                    used_tokens.add(token.i)

        # Go over noun chunks
        # Find noun chunks and add them to the bag of words bag and add indices to used list
        for match_id, start, end in m:
            span = doc[start:end]

            bag.append(span.lemma_.replace("-PRON-", "").strip().replace(" ", "-"))
            for token in span:
                used_tokens.add(token.i)

        # Add the rest of the tokens
        for token in doc:
            if not token.i in used_tokens:
                if token.tag_ in used_tags:
                    lemma = token.lemma_
                    if not token.is_stop and token.is_alpha and lemma != "-PRON-" and len(lemma) > 1:
                        bag.append(lemma)

        clean_fancy.append(" ".join(bag))

    return texts, clean, clean_fancy, labels
