from nltk import tag
from nltk.corpus import mac_morpho

from pickle import load, dump

def train_tagger():
    '''
    Train a tagger on the mac_morpho tagged corpus (must be
    available locally) and save it for future use. A total
    of 4 taggers are trained to allow backing of (1) and
    guarantee an assignment to every term.
    (1) 3-gram -> 2-gram -> 1-gram -> N (noun)
    '''
    tagged_sents = mac_morpho.tagged_sents()

    tagger0 = tag.DefaultTagger('N')
    tagger1 = tag.UnigramTagger(tagged_sents,  backoff=tagger0)
    tagger2 = tag.BigramTagger(tagged_sents  , backoff=tagger1)
    tagger3 = tag.NgramTagger(3, tagged_sents, backoff=tagger2)

    save_tagger(tagger3)


def save_tagger(tagger):
    file = open('tagger', 'wb')
    dump(tagger, file, -1)
    file.close()


def load_tagger():
    file = open('tagger', 'rb')
    tagger = load(file)
    file.close()
    return tagger


#train_tagger()