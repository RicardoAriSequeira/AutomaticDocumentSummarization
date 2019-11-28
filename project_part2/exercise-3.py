import os

import customtagger

from nltk import sent_tokenize, word_tokenize
from sklearn.linear_model import Perceptron

from collections import Counter

from customvectorizer import CustomVectorizer, similarity

from utilities import pre_process, filter_list


e2 = __import__('exercise-2')

def next_r(reset=False):
    if reset:
        next_r.r = 0
    else:
        next_r.r += 1
        return next_r.r

def sent_in_target(sent, target_sents):
    try:
        rank = target_sents.index(sent) + 1
        return rank
    except ValueError:
        return -1


def extract_features(source):

    def count_tags(tags, label):

        tag_counter = Counter(tags)
        counts = 0

        if label == 'nouns':
            for tag in tag_counter.keys():
                if tag in ['N', 'NPROP', 'PROPESS']:
                    counts += tag_counter.get(tag)

        elif label == 'verbs':
            for tag in tag_counter.keys():
                if tag in ['V', 'VAUX', 'ADV', 'ADV-KS', 'ADV-KS-REL']:
                    counts += tag_counter.get(tag)

        elif label == 'adjectives':
            for tag in tag_counter.keys():
                if tag in ['ADJ']:
                    counts += tag_counter.get(tag)

        return counts


    sents = filter_list(sent_tokenize(source))

    feature_pos = [i + 1 for i in range(len(sents))]

    vectorizer = CustomVectorizer()
    vectorizer.fit(sents)  # whole document must be split before fitting TODO -> filter

    tfidf_source = vectorizer.transform_tfidf([source])[0]
    tfidf_sents  = vectorizer.transform_tfidf(sents)

    feature_sim = [similarity(tfidf_sent, tfidf_source) for tfidf_sent in tfidf_sents]

    tagger = customtagger.load_tagger()

    tagged = [tagger.tag(filter_list(word_tokenize(sent))) for sent in sents]

    sent_tags = [list(map(lambda t: t[1], tags)) for tags in tagged]

    feature_nouns = [count_tags(tags, 'nouns') for tags in sent_tags]

    #feature_verbs = [count_tags(tags, 'verbs') for tags in sent_tags]

    #feature_adjectives = [count_tags(tags, 'adjectives') for tags in sent_tags]

    return sents, feature_pos, feature_sim, feature_nouns  #, feature_verbs, feature_adjectives


# Initialize the perceptron for online training
perceptron = Perceptron(tol=1e-3)

rank_classes = [i for i in range(1, 45)]  # [1, 2, 3, 4, 5]

X = []
Y = []

print('\nTraining a perceptron to rank sentences, using the TeMario 2006 dataset.',
      '\nThe following sentence features are considered:\n', \
      '- position in the text\n', \
      '- similarity towards the entire document\n', \
      '- noun words count\n')

print('Gathering data from folder:')

TeMario_originals = os.getcwd() + "/TeMário 2006/Originais/"
TeMario_summaries = os.getcwd() + "/TeMário 2006/SumáriosExtractivos/"
for folder in os.listdir(TeMario_originals):

    if folder == '.DS_Store':  # Ignore mac default attributes' folder
        continue

    print(folder + '')

    for file in os.listdir(TeMario_originals + folder):

        if file == '.DS_Store':  # Ignore mac default attributes' folder
            continue

        next_r(reset=True)

        source_file = open(TeMario_originals + folder + '/' + file, 'r', encoding='iso-8859-1')
        target_file = open(TeMario_summaries + folder + '/' + 'Sum-' + file, 'r', encoding='iso-8859-1')

        source = pre_process(source_file.read())
        target = pre_process(target_file.read())

        sents, feature_pos, feature_sim, feature_nouns = extract_features(source)  # feature_verbs, feature_adjectives

        # TODO -> estimates produced through a model similar to a NB classifier

        # TODO -> graph centrality scores

        target_sents = filter_list(sent_tokenize(target))
        target_score = [sent_in_target(sent, target_sents) for sent in sents]

        # TODO -> collect target classes

        X.extend([[feature_pos[i], feature_sim[i], feature_nouns[i]] for i in range(len(sents))])
        Y.extend(target_score)

        source_file.close()
        target_file.close()


print('\nFitting data...')
perceptron.fit(X, Y)
print('(done)')


# Evaluating the perceptron performance
print('\nEvaluating the perceptron performance on the original TeMário dataset...')

perceptron_MAP = e2.MAP()

TeMario_originals = os.getcwd() + "/TeMário/Textos-fonte/"
TeMario_summaries = os.getcwd() + "/TeMário/Extractos/"

files = os.listdir(TeMario_originals)

for file in files:

    source_file = open(TeMario_originals + file, 'r', encoding='iso-8859-1')
    target_file = open(TeMario_summaries + 'Ext-' + file, 'r', encoding='iso-8859-1')

    source = source_file.read()
    target = target_file.read()

    source_file.close()
    target_file.close()

    sents, feature_pos, feature_sim, feature_nouns = extract_features(source)  # feature_verbs, feature_adjectives

    X = [[feature_pos[i], feature_sim[i], feature_nouns[i]] for i in range(len(sents))]

    rank = perceptron.predict(X)

    rank = {i: rank[i] for i in range(len(rank))}
    indices = sorted(rank.keys(), key=lambda k: rank[k], reverse=True)[:5]
    summary = [sents[i] for i in indices]

    target_summary = filter_list(sent_tokenize(target))

    perceptron_MAP.accumulate(summary, target_summary)


print('Perceptron MAP:', round(perceptron_MAP.result(), 3))