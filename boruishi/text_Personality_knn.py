import pandas
import nltk
import numpy
import codecs
import sys
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
from os import listdir
from os.path import isfile, join


df = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
fwrite = codecs.open('boruishi/bigtext.txt','a')
text_line=[]

for i in range(0,9000):
    f = codecs.open('data/TCSS555/Train/Text/'+df['userid'][i]+'.txt', encoding='latin-1')
    text = f.readline()
    text_line.append(text)


english_stopwords = stopwords.words("english")
for i in range (0,len(text_line)):
    texttemp = text_line[i]
    text_line[i] = texttemp.lower()
for i in range (0,len(text_line)):
    text_line[i] = nltk.word_tokenize(text_line[i])

    
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in text_line]
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered_stopwords ]
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
index = similarities.MatrixSimilarity(lsi[corpus])


right = 0.0
open=0.0
for i in range(1,2000):
    ml_course = texts[i]
    ml_bow = dictionary.doc2bow(ml_course)
    ml_lsi = lsi[ml_bow]
    sims = index[ml_lsi]    
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    ope_sum = 0
    for i in range (0,7):
        row = sort_sims[i][0]
        ope = df['ope'][row]
        ope_sum+=ope
    ope_sum = ope_sum/8
    if abs(ope_sum-df['ope'][i])<0.4:
        right+=1
    


