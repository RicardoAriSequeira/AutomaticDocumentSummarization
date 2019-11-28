import os
import webbrowser


import xml.etree.ElementTree as ET
import re

from urllib.request import urlopen
from nltk.corpus import stopwords
from nltk import sent_tokenize

from customvectorizer import CustomVectorizer, similarity

e1 = __import__('exercise-1')

def getContent(url, file):
    n_sentences = 0

    xmlUrl = urlopen(url)
    xmlUrl_string = xmlUrl.read().decode('utf-8')
    xmlUrl_string = xmlUrl_string.replace('&lt;', '<')
    xmlUrl_string = xmlUrl_string.replace('&gt;', '>')
    xmlUrl_string = xmlUrl_string.replace('\n\t  ', '')
    xmlUrl_string = xmlUrl_string.replace('<p>', '')
    xmlUrl_string = xmlUrl_string.replace('</p>', '')
    xmlUrl_string = re.compile(r'<img.*?/>').sub('', xmlUrl_string)
    xmlUrl_string = re.compile(r'<a.*?>').sub('', xmlUrl_string)
    xmlUrl_string = re.compile(r'</a>').sub('', xmlUrl_string)

    root = ET.fromstring(xmlUrl_string)
    for elem in root.findall(".//item"):
        if elem.find("title") is not None and elem.find("title").text is not None:
            n_sentences += 1
            text = elem.find("title").text
            text = text.replace('\t', '')
            text = text.replace('\n', '')
            file.write(text + '.')
        if elem.find("description") is not None and elem.find("description").text is not None:
            n_sentences += 1
            text = elem.find("description").text
            text = text.replace('\t', '')
            text = text.replace('\n', '')
            file.write(text + '.')

    return n_sentences

def generateHTML(sents, summary):

    html = open('exercise-4.html', 'w')

    html.write('<html>\n<head>\n<meta charset="UTF-8">\n<title>Exercise 4 - Summary of World News</title>\n</head>\n')
    html.write('<body>\n<h1>World News Summary</h1>\n')

    sentence = 0

    for i in summary:
        sentence += 1
        if i < index_nytimes:
            html.write('<h2>Sentence ' + str(sentence) + ' <i>(obtained from New York Times)</i>:</h2>\n')
        elif i < index_cnn:
            html.write('<h2>Sentence ' + str(sentence) + ' <i>(obtained from CNN)</i>:</h2>\n')
        elif i < index_washigtonpost:
            html.write('<h2>Sentence ' + str(sentence) + ' <i>(obtained from Washington Post)</i>:</h2>\n')
        elif i < index_latimes:
            html.write('<h2>Sentence ' + str(sentence) + ' <i>(obtained from LA Times)</i>:</h2>\n')
        html.write('<p>' + sents[i]+ '</p>\n')

    html.write('</body>\n</html>')

    html.close()


# Retrieve World news from sources and store them in a file
file = open("worldnews.txt", 'w')

print('\nFetching content from sources...')

index_nytimes = getContent("https://rss.nytimes.com/services/xml/rss/nyt/World.xml", file)
index_cnn = index_nytimes + getContent("http://rss.cnn.com/rss/edition_world.rss", file)
index_washigtonpost = index_cnn + getContent("http://feeds.washingtonpost.com/rss/world", file)
index_latimes= index_washigtonpost + getContent("http://www.latimes.com/world/rss2.0.xml", file)

file.close()

print('(done)')

print('\nSummarizing...')

file = open('worldnews.txt', encoding='utf-8')

text  = file.read()
sents = sent_tokenize(text)

file.close()

vectorizer = CustomVectorizer(stopwords=stopwords.words())

vectorizer.fit(sents)
vecs = vectorizer.transform_tfidf(sents)

graph = {i: [] for i in range(len(vecs))}

threshold = 0.1
for i in range(len(vecs)):
    for j in range(i+1, len(vecs)):
        if similarity(vecs[i], vecs[j]) > threshold:
            graph[i].append(j)
            graph[j].append(i)

graph = {k: list(set(graph[k])) for k in graph.keys()}

rank, i = e1.page_rank_mod(graph)

summary = sorted(rank.keys(), key=lambda k : rank[k], reverse=True)[:5]
summary.sort()

print('(done)')

print('\nBuilding html page...')
generateHTML(sents, summary)
print('(done)')

print('\nOpen in browser? (y/n)')

option = input()

if option in ['y', 'Y']:
    page = os.getcwd() + '/exercise-4.html'
    webbrowser.open(page, 2)

print('\nSummary available in file exercise-4.html.')