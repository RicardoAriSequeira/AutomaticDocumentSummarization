import os

from nltk import sent_tokenize
from customvectorizer import CustomVectorizer, similarity
from numpy import subtract, absolute

from collections import defaultdict

from utilities import pre_process, filter_list

e1 = __import__('exercise-1')

class MAP:
    '''
    Class to store the cumulative average precision
    of each retrieval and then compute the Mean Average
    Precision.
    '''
    def __init__(self):
        self.Q = 0
        self.c_avg_precision = 0

    '''
    Compute the Mean Average Precision, summing and
    storing each summary average precision iteratively.
    '''
    def precision_at(self, summary, target, N):
        return [len([s for s in summary[:i] if s in target])/i for i in range(1, N+1)]

    def accumulate(self, summary, target):
        N = len(summary)
        precision_at_vector = self.precision_at(summary, target, N)

        self.Q += 1
        self.c_avg_precision += sum([precision_at_vector[i] for i, sent in enumerate(summary, start=0) if sent in target])/N

    def result(self):
        return self.c_avg_precision / self.Q


def page_rank_mod_improv(graph, priors, weights, iterations=50, d=0.15, diff=0.001):
    '''
    Given a undirected graph where keys are sentence' ids and each
    corresponding value is a set of weighted similar sentences, iteratively
    computes a rank for all sentences. Bounded by a maximum of 50
    iterations or an absolute difference less then 0.001 between
    iterations, for every sentence.
    '''
    def difference(rank1, rank2):
        r1 = sorted(rank1.values(), key=lambda r: r)
        r2 = sorted(rank2.values(), key=lambda r: r)
        return absolute(subtract(r1, r2))

    N = len(graph.keys())
    priors_sum = sum(priors.values())

    sinks = []  # TODO -> ask how to treat sinks
    rank  = {k : 1/N for k in graph.keys()}  # initial rank as a uniform initial probability distribution

    for i in range(min(iterations, 50)):
        rank_i = rank.copy()
        for k in rank.keys():
            t1 = priors[k] / priors_sum
            t2 = sum([rank_i[j] * weights[k][j] / sum(weights[j].values()) for j in graph[k]])
            rank[k] = d * t1 + (1-d) * t2

        if max(difference(rank, rank_i)) < diff:
            break

    rank = {k : round(rank[k], 6) for k in rank.keys()}
    return rank, i+1


# ----------------------------------------------------- #
# Evaluate both PageRank inspired summarization methods #
# ----------------------------------------------------- #
if __name__ == '__main__':

    print('\nBuild on the adapted PageRank method of the first exercise to improve sentence ranking performance.\n' +
          'Each sentence is given a prior probability based on a non-uniform distribution and the edges weights\n' +
          'of the generated graph are now taken in consideration and depend on pairwise features the sentences\n'
          'may exhibit.')

    print('\nProcessing...')

    TeMario_originals = os.getcwd() + "/TeMário/Textos-fonte/"
    TeMario_summaries = os.getcwd() + "/TeMário/Extractos/"

    files = os.listdir(TeMario_originals)

    rank_MAP = MAP()
    rank_improv_1_MAP = MAP()
    rank_improv_2_MAP = MAP()

    for file in files:

        if file == '.DS_Store':  # Ignore mac default attributes' folder
            continue

        source_file = open(TeMario_originals + file, 'r', encoding='iso-8859-1')
        target_file = open(TeMario_summaries + 'Ext-' + file, 'r', encoding='iso-8859-1')

        source = pre_process(source_file.read())
        target = pre_process(target_file.read())

        source_file.close()
        target_file.close()

        sents = filter_list(sent_tokenize(source))

        vectorizer = CustomVectorizer()

        vectorizer.fit(sents)

        vecs = vectorizer.transform_tfidf(sents)
        source_score = vectorizer.transform_tfidf([source])[0]

        graph = defaultdict(lambda: [])

        weights_tfidf = defaultdict(lambda: {})
        weights_alternative = []  # TODO

        # Build graph
        threshold = 0.1
        for i, v1 in enumerate(vecs):
            for j, v2 in enumerate(vecs[i + 1:], start=i + 1):
                sim = similarity(v1, v2)
                if sim > threshold:
                    graph[i].append(j)
                    graph[j].append(i)

                    weights_tfidf[i][j] = sim
                    weights_tfidf[j][i] = sim

        l = len(graph.keys())
        n = sum(l / (i + 1) for i in range(l))

        # priors must be a probability distribution -> sum equals 1
        priors_pos   = {k: (l / (k + 1)) / n for k in graph.keys()}
        priors_tfidf = {k: similarity(vecs[k], source_score) for k in graph.keys()}
        priors_bayes = []  # TODO

        target_summary = filter_list(sent_tokenize(target))

        rank, i        = e1.page_rank_mod(graph)
        rank_improv_1, i = page_rank_mod_improv(graph, priors_pos, weights_tfidf)
        rank_improv_2, i = page_rank_mod_improv(graph, priors_tfidf, weights_tfidf)

        indices = sorted(rank.keys(), key=lambda k: rank[k], reverse=True)[:5]
        summary = [sents[i] for i in indices]
        rank_MAP.accumulate(summary, target_summary)

        indices = sorted(rank_improv_1.keys(), key=lambda k: rank_improv_1[k], reverse=True)[:5]
        summary = [sents[i] for i in indices]
        rank_improv_1_MAP.accumulate(summary, target_summary)

        indices = sorted(rank_improv_2.keys(), key=lambda k: rank_improv_2[k], reverse=True)[:5]
        summary = [sents[i] for i in indices]
        rank_improv_2_MAP.accumulate(summary, target_summary)


    print('(done)')

    print('\nAdapted PageRank MAP:', round(rank_MAP.result(), 3))
    print('\nImproved algorithm, considering:')

    print('-> position based priors and tfidf weights | MAP:', round(rank_improv_1_MAP.result(), 3))
    print('-> tfidf based priors and tfidf weights    | MAP:', round(rank_improv_2_MAP.result(), 3))
