# -*- coding: utf-8 -*-

from nltk import sent_tokenize
from customvectorizer import CustomVectorizer
from customvectorizer import similarity
from nltk.corpus import stopwords
import io

import re

# function to pre process sentences


def preprocessor(document):
    document = re.sub(
        r'([a-zA-ZáàâãéêíóõôúçÁÀÂÃÉÊÍÓÕÔÚÇ, ])\n', r'\1.\n', document)
    document = re.sub(r'[0-9"-,()ºª;$€&]+', '', document)
    document = document.replace('\n', ' ')
    return document

# ------------------------------------------- #
# Summarizing procedure and related functions #
# ------------------------------------------- #


def summarizeDocument(file):
    '''
    Parse the file for sentences
    '''
    doc = io.open(file, 'r', encoding='utf').read()
    doc = preprocessor(doc)
    sentences = sent_tokenize(doc)

    '''
    Compute vector space representations of every sentence.
    It will treat each sentence as a document and so use
    the correct values (sentence frequency).
    !!!
    The tf-idf values computed by this vectorizer are not
    in accordance to what is requested. Documentation states
    that tf is simply the count of each word in each doc/sentence
    (and so, not normalized), and 1 is added to all idf values.
    To meet the requirements, for each term, we would need to subtract
    its tf (as described above) and then divide by the maximal tf in
    that doc/sentence. I think we need to use a CounterVectorizer first
    !!!
    '''
    vectorizer = CustomVectorizer(
        input='content', stopwords=list(stopwords.words('english')))
    vectors = vectorizer.fit(sentences)
    vectors = vectorizer.transform_tfidf(sentences)

    '''
    Transform the document into a single sentence and use
    the vectorizer to model it in the same feature space.
    '''
    docVector = vectorizer.transform_tfidf([doc])

    '''
    For each sentence vector, reduce the document vector
    to the same dimension space, to be able to compute
    the dot product -> similarity
    '''
    sim = []
    for vector in vectors:
        sim.append(similarity(vector, docVector[0]))

    summary = sorted(enumerate(sim), key=lambda s: s[1], reverse=True)[:5]
    summary.sort()

    '''
    Returns the list of selected sentences
    '''
    res = []
    for s in summary:
        res.append(sentences[s[0]])
    return res


# Test
print('Summary:')
for n, sentence in enumerate(summarizeDocument("catalunha.txt"),start=1):
    print(n,sentence)
