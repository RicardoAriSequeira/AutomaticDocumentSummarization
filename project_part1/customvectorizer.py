from collections import Counter, defaultdict
from nltk import sent_tokenize, word_tokenize

from functools import reduce

from math import log
from numpy import dot
from numpy.linalg import norm

import string, re, customtagger


class CustomVectorizer:
    '''
    Built specifically for the first PRI project, to be able to change
    the score function upon request and better manage feature extraction.
    Currently supports:
        score:
            -> tf-idf: with normalized tf and logarithmically scaled idf;
            -> BM25  : see https://en.wikipedia.org/wiki/Okapi_BM25
        extraction:
            -> stopwords removal, bi-grams and noun phrases extraction
            -> minimum and maximum document frequency (both total and relative)
    '''
    def __init__(self, input='contents', norm=True, mindf=None, maxdf=None, stopwords=[] \
                     , bigrams=False, nphrases=False, encoding='utf-8'):
        self._df        = []
        self._avgdl     = []
        self._input     = input
        self._norm      = norm
        self._encoding  = encoding
        self._mindf     = mindf
        self._maxdf     = maxdf
        self._stopwords = stopwords
        self._bigrams   = bigrams
        self._nphrases  = nphrases


    def fit(self, data):
        '''
        Extracts features and characteristics from the data, used by
        the score function, namely document frequency and average
        document length.
        data: list of individual sentences or document strings if
              input is read from files
        '''

        if self._input == 'fromfiles':
            data = self.read_data_from_file(data)

        df    = []
        avgdl = []
        for d in data :
            features = self.extract_features(d)
            df.extend(list(set(features)))
            avgdl.append(len(features))

        self._nd    = len(data)
        self._df    = Counter(df)
        self._avgdl = reduce(int.__add__, avgdl)/len(avgdl)

        return self._nd, self._df, self._avgdl


    def transform(self, data, score_fn):
        '''
        Returns a vector representation (as a dict) of the provided
        text, given the score function score_fn
        '''
        if self._input == 'fromfiles':
            data = self.read_data_from_file(data)

        vecs = []
        for d in data:

            vec = defaultdict(lambda: 0)
            features = self.extract_features(d)

            freq = Counter(features)

            for feat in features:
                vec[feat] = score_fn(feat, freq)

            if norm:
                self.normalize(vec)

            vecs.append(vec)

        return vecs


    def transform_tfidf(self, text):
        '''
        Returns a vector representation (as a dict) of the provided
        text, considering the tf-idf score function
        '''
        return self.transform(text, self.tfidf)


    def transform_bm25(self, text):
        '''
        Returns a vector representation (as a dict) of the provided
        text, considering the bm25 score function
        '''
        return self.transform(text, self.bm25)


    def tfidf(self, word, freq):
        f = freq[word]
        maxf = max(freq.values())
        return (f/maxf) * log(self._nd/self._df[word]) if self._df[word] != 0 else 0


    def bm25(self, word, freq):
        l = len(freq.keys())
        avg = self._avgdl
        f = freq[word]
        N = self._nd
        n = self._df[word]

        k = 1.2
        b = 0.75

        idf = log((N - n + 0.5) / (n + 0.5))

        bm = f * (k + 1) / (f + k * (1 - b + b * (l/avg)))

        return idf * bm


    def read_data_from_file(self, files):
        data = []

        for f in files:
            d = open(f, 'r', encoding=self._encoding).read().replace('\n', ' ')
            data.append(d)

        return data


    def extract_features(self, data):
        '''
        Extracts features from text data. By default collects
        single words, discarding punctuation symbols. Bi-grams
        and noun-phrases are added as tokens if flags bi-grams
        and nounphr are set to True
        '''
        def build_bigrams(d):

            words = list(filter(lambda w: w not in list(string.punctuation), word_tokenize(d.lower())))
            bigrams = [words[i] + ' ' + words[i + 1] for i in range(len(words) - 1)]

            return bigrams

        def build_noun_phrases(d):
            '''
            Retrieves noun-phrases. Word tokenization keeps all
            information (case and punctuation) to improve tagging.
            '''
            words = word_tokenize(d)

            tagged = tagger.tag(words)

            tags = [t[1] for t in tagged]

            tags_str = ' '.join([t for t in tags])

            pattern  = r'((ADJ )*(((NPROP)|N) )+((PREP)|(KS)) )?(ADJ )*(((NPROP)|N) ?)+'

            matches = re.finditer(pattern, tags_str, flags=0)

            nphrases = []
            for m in matches:
                m_tags = list(filter(lambda t: t != '', m.group(0).split(' ')))
                l = len(m_tags)
                for i in range(len(tags)-len(m_tags)):
                    if tags[i:i+l] == m_tags:
                        nphrases.append(' '.join(words[i:i+l]).strip())

            return nphrases


        features = list(filter(lambda w: w not in list(string.punctuation) + self._stopwords + ['\n'],  word_tokenize(data.lower())))

        if self._bigrams:
            # Extract bi-grams here
            if self._input == 'fromfile':

                data = sent_tokenize(data)

                f = []
                for d in data:
                    f.extend(build_bigrams(d))

                features.extend(f)

            else:
                features.extend(build_bigrams(data))

        if self._nphrases:
            # Extract noun phrases here
            tagger = customtagger.load_tagger()

            if self._input == 'fromfile':
                data = sent_tokenize(data)

                f = []
                for d in data:
                    f.extend(build_noun_phrases(d))

                features.extend(f)

            else:
                features.extend(build_noun_phrases(data))

        return features


    def normalize(self, vec):
        '''
        Normalizes a vector represented as a dict
        '''
        n = norm(list(vec.values()))

        for k in vec.keys():
            vec[k] /= n


    def freq_process(self, freq):
        '''
        Process the frequencies vector to remove features that
        don't meet the specified requirements
        '''

        if self._mindf:
            if type(self._mindf) == int:
                op = lambda v: self._df[v]
            else:
                op = lambda v: self._df[v]/self._nd

            for k in freq.keys():
                if op(k) < self._mindf:
                    freq.pop(k)

        if self._maxdf:
            if type(self._maxdf) == int:
                op = lambda v: self._df[v]
            else:
                op = lambda v: self._df[v]/self._nd

            for k in freq.keys():
                if op(freq[k]) > self._maxdf:
                    freq.pop(k)


    def pre_process(self, data):
        '''
        Pre-process textual data, given a function --> can also use an if where's needed
        passed to the constructor.
        '''
        self._preprocess(data)

# END_OF_CUSTOMVECTOR


def similarity(vec1, vec2, r=4):
    '''
    Computes co-sine similarity given two vectors
    represented as a dict
    '''
    v  = vec1 if len(vec1) < len(vec2) else vec2
    v1 = []
    v2 = []
    for k in v.keys():
        v1.append(vec1[k])
        v2.append(vec2[k])

    return round(abs(float(dot(v1, v2))), r)