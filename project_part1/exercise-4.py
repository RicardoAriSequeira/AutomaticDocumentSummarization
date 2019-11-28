from nltk import sent_tokenize
from os import listdir
from customvectorizer import CustomVectorizer
from customvectorizer import similarity

import re

# gets a list of portugueses stopwords from file
stopwords = open('stopwords.txt').read().splitlines()


# function to pre process sentences
def preprocessor(document):
    document = re.sub(
        r'([a-zA-ZáàâãéêíóõôúçÁÀÂÃÉÊÍÓÕÔÚÇ, ])\n', r'\1.\n', document)
    document = re.sub(r'[0-9"-,()ºª;$€&]+', '', document)
    document = document.replace('\n', ' ')
    return document


def summary(vectorizer, file):

    # gets file sentences
    doc = open(file, 'r', encoding='iso-8859-1').read()
    sentences = sent_tokenize(preprocessor(doc))

    # calculate tfidf vector for document sentences and all document
    vectors = vectorizer.transform_tfidf(sentences)
    docVector = vectorizer.transform_tfidf([doc])

    # calculate similarity for each sentence
    sim = []
    for vector in vectors:
        sim.append(similarity(vector, docVector[0]))

    # calculate MMR value for each sentence
    selected = []
    var = 0.05
    while len(selected) < 5:
        mmr = []
        for s in range(len(sim)):
            mmr_value = (1 - var) * sim[s]
            for sentence in selected:
                mmr_value -= var * similarity(vectors[s], vectors[sentence])
            mmr.append(mmr_value)
        indexOfMax = max(enumerate(mmr), key=lambda x: x[1])[0]
        selected.append(indexOfMax)
        sim[indexOfMax] = 0

    # returns the list of selected sentences
    res = []
    for i in selected:
        res.append(sentences[i])
    return res


def calculateStats(file, summary):

    # gets file sentences
    doc = open(file, 'r', encoding='iso-8859-1').read()
    sentences = sent_tokenize(preprocessor(doc))[1:6]

    # calculates number of true positives
    true_positives = 0
    for sentence in sentences:
        for s in summary:
            if sentence == s:
                true_positives += 1

    # calculates precision, recall and F1 score
    precision = true_positives / len(summary)
    recall = true_positives / 5
    if true_positives == 0:
        F1_score = format(0, '.6f')
    else:
        F1_score = format(2 * (precision * recall) /
                          (precision + recall), '.6f')

    return [format(precision, '.6f'), format(recall, '.6f'), F1_score]


def main():

    # initialize custom vectorizer with all documents collection
    vectorizer = CustomVectorizer(
        input='fromfiles', stopwords=stopwords, encoding='iso-8859-1')
    documents = ['textos-fonte/' + d for d in listdir('textos-fonte')]
    vectorizer.fit(documents)
    vectorizer._input = 'content'

    # print all statistics
    MAP = 0
    print('File\t\t\tPrecision\tRecall\t\tF1 Score')
    for doc in listdir('textos-fonte'):
        path = 'textos-fonte/' + doc
        stats = calculateStats(path, summary(vectorizer, path))
        MAP += float(stats[0])
        print (doc + '\t\t' + stats[0] + '\t' + stats[1] + '\t' + stats[2])
    print('\nMAP Score: ' + str(MAP / 100))


main()
