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


def summary(vectorizer, file, alternativeApproach=False):

    # gets file sentences
    doc = open(file, 'r', encoding='iso-8859-1').read()
    sentences = sent_tokenize(preprocessor(doc))

    # simple approach: use document sentences
    if alternativeApproach == False:
        vectorizer.fit(sentences)

    # calculate tfidf vector for document sentences and all document
    vectors = vectorizer.transform_tfidf(sentences)
    docVector = vectorizer.transform_tfidf([doc])

    # calculate similarity for each sentence
    sim = []
    for vector in vectors:
        sim.append(similarity(vector, docVector[0]))

    # select document summary
    summary = sorted(enumerate(sim), key=lambda s: s[1], reverse=True)[:5]
    summary.sort()

    # returns the list of selected sentences
    res = []
    for i, s in enumerate(summary, start=1):
        res.append(sentences[s[0]])
    return res


def compareApproaches(file, result1, result2):

    # gets automatic extracts from dataset
    extrato = 'extratos/Ext-' + file
    doc = open(extrato, 'r', encoding='iso-8859-1').read()
    sentences = sent_tokenize(preprocessor(doc))

    TP_1 = 0  # True Positives for Simple Approach
    TP_2 = 0  # True Positives for Alternative Approach

    for s in sentences:
        for res in result1:
            if s == res:
                TP_1 += 1
        for res2 in result2:
            if s == res2:
                TP_2 += 1

    P_1 = TP_1 / len(result1)  # Precision for Simple Approach
    P_2 = TP_2 / len(result2)  # Precision for Alternative Approach
    R_1 = TP_1 / len(sentences)  # Recall for Simple Approach
    R_2 = TP_2 / len(sentences)  # Recall for Alternative Approach

    # F1 Score for Simple Approach
    if TP_1 == 0:
        F1_1 = format(0, '.6f')
    else:
        F1_1 = format(2 * (P_1 * R_1) / (P_1 + R_1), '.6f')

    # F1 Score for Alternative Approach
    if TP_2 == 0:
        F1_2 = format(0, '.6f')
    else:
        F1_2 = format(2 * (P_2 * R_2) / (P_2 + R_2), '.6f')

    return [file, format(P_1, '.6f'), format(R_1, '.6f'),
            F1_1, format(P_2, '.6f'), format(R_2, '.6f'), F1_2]


def main():

    # initialize custom vectorizer with all documents collection
    vectorizer1 = CustomVectorizer(
        input='fromfiles', stopwords=stopwords, encoding='iso-8859-1')
    vectorizer2 = CustomVectorizer(
        input='fromfiles', stopwords=stopwords, encoding='iso-8859-1')
    documents = ['textos-fonte/' + d for d in listdir('textos-fonte')]
    vectorizer2.fit(documents)
    vectorizer1._input = 'content'
    vectorizer2._input = 'content'

    # print all statistics
    MAP1 = 0
    MAP2 = 0
    print('File\t\t\tPrecision Simple\tRecall Simple\tF1 Simple\tPrecision Alternative\tRecall Alternative\tF1 Alternative')
    for doc in listdir('textos-fonte'):
        path = 'textos-fonte/' + doc
        stats = compareApproaches(doc, summary(
            vectorizer1, path), summary(vectorizer2, path, True))
        MAP1 += float(stats[1])
        MAP2 += float(stats[4])
        print (stats[0] + '\t\t' + stats[1] + '\t\t' + stats[2] + '\t' +
               stats[3] + '\t' + stats[4] + '\t\t' + stats[5] + '\t\t' + stats[6])
    print('\nMAP Score for Simple Approach: ' + str(MAP1 / 100))
    print('\nMAP Score for Alternative Approach: ' + str(MAP2 / 100))


main()
