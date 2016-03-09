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
import math
import pickle


class text_ope_knn_test:
    def __init__(self):
        self.df = pandas.read_csv('/home/itadmin/MLProject/data/TCSS555/Train/Profile/Profile.csv')

        cf = open('/home/itadmin/MLProject/boruishi/dictionary.pickle', 'rb')
        self.dictionary = pickle.load(cf)
        cf.close()

        cf = open('/home/itadmin/MLProject/boruishi/lsi.pickle', 'rb')
        self.lsi = pickle.load(cf)
        cf.close()

        cf = open('/home/itadmin/MLProject/boruishi/index.pickle', 'rb')
        self.index = pickle.load(cf)
        cf.close()

        return

    def format(self, txt):
        texts_token = nltk.word_tokenize(txt)
        english_stopwords = stopwords.words("english")
        texts_filtered_stopwords = [word for word in texts_token if not word in english_stopwords]
        english_corm = ","
        texts_filtered_corm = [word for word in texts_filtered_stopwords if not word in english_corm]
        st = LancasterStemmer()
        text_stemmed = [st.stem(word) for word in texts_filtered_stopwords]
        #text = [stem for stem in txt if stem not in text_stemmed ]
        return text_stemmed

    def test(self, texts):
        ml_course = texts
        ml_bow = self.dictionary.doc2bow(ml_course)
        ml_lsi = self.lsi[ml_bow]
        sims = self.index[ml_lsi]    
        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        ope_sum = 0
        for i in range (0,20):
            row = sort_sims[i][0]
            ope = self.df['ope'][row]
            ope_sum+=ope
        ope_sum = ope_sum/20
        return ope_sum
        
    def predict(self, txt): 
        texts = self.format(txt)
        userope = self.test(texts)
        return userope





