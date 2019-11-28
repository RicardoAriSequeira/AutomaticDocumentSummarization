from nltk import sent_tokenize
from nltk.corpus import stopwords
from numpy import subtract, absolute

from customvectorizer import CustomVectorizer, similarity
from utilities import pre_process, filter_list

def page_rank_mod(graph, iterations=50, d=0.15, diff=0.001):
    '''
    Given a undirected graph where keys are sentence' ids and each
    corresponding value is a set of similar sentences, iteratively
    computes a rank for all sentences. Bounded by a maximum of 50
    iterations or an absolute difference less then 0.001 between
    iterations, for every sentence.
    '''
    def outlinks(id):
        '''
        If the similarity set is empty, then the sentence acts
        as a sink, and edges to all other pages are considered.
        Follows the same approach of traditional PageRank, by
        distributing its weight evenly.
        '''
        if graph[id]:
            return len(graph[id])
        else:
            sinks.append(id)
            return N-1

    def difference(rank1, rank2):
        r1 = sorted(rank1.values(), key=lambda r: r)
        r2 = sorted(rank2.values(), key=lambda r: r)
        return absolute(subtract(r1, r2))

    N = len(graph.keys())

    sinks = []
    rank  = {k : 1/N for k in graph.keys()}  # initial rank as a uniform initial probability distribution
    links = {k : outlinks(k) for k in graph.keys()}  # pre-compute out-links for efficiency

    for i in range(min(iterations, 50)):  # should enforce max iterations
        rank_i = rank.copy()
        for k in rank.keys():
            rank[k] = d/N + (1-d) * sum([rank_i[j]/links[j] for j in graph[k] + sinks if j != k])

        if max(difference(rank, rank_i)) < diff:
            break

    rank = {k : round(rank[k], 6) for k in rank.keys()}  # the rounding precision influences convergence testing

    return rank, i


if __name__ == '__main__':

    print('\nTesting adapted PageRank algorithm for sentence ranking and consequent text summarization.\n' +
          'A graph is built linking sentences with similarity bigger than a certain threshold.\n' +
          'This method is tested and evaluated on the "catalunha.txt" file, with a 0.1 threshold.\n')

    file = open('catalunha.txt', encoding='utf-8')

    source = pre_process(file.read())
    sents  = filter_list(sent_tokenize(source))

    file.close()

    vectorizer = CustomVectorizer(stopwords=stopwords.words())

    vectorizer.fit(sents)  # -> fit on sentences or on whole text?
    vecs = vectorizer.transform_tfidf(sents)

    graph = {i: [] for i in range(len(vecs))}

    threshold = 0.1
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            if similarity(vecs[i], vecs[j]) > threshold:
                graph[i].append(j)
                graph[j].append(i)

    graph = {k: list(set(graph[k])) for k in graph.keys()}

    rank, i = page_rank_mod(graph)

    summary = sorted(rank.keys(), key=lambda k : rank[k], reverse=True)[:5]
    summary.sort()

    print('Summary:')
    for i in summary:
        print(i, ': ', sents[i], sep='')

    print('\nComputed in', i, 'iterations:')
    print('Rank:', rank)
    print('Sum :', round(sum(rank.values())))


