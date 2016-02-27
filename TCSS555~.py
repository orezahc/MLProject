import pandas
import nltk
import numpy
import codecs
import sys
import pickle
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

df = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')


# f = open('data/TCSS555/Train/Text/'+df['userid'][0]+'.txt', 'r')
# text = f.readline()
# tokens = nltk.word_tokenize(text[0].replace('\\',''))
# tokens
# tagged = nltk.pos_tag(tokens)
# tagged
# entities = nltk.chunk.ne_chunk(tagged)

all_words = set()
train = []
test_size = 0
wnl = WordNetLemmatizer()

for i in range(0, len(df)-test_size):
	f = codecs.open('data/TCSS555/Train/Text/'+df['userid'][i]+'.txt', encoding='latin-1')
	text = f.readline().replace('\\','').lower()
	text_token = nltk.word_tokenize(text)
	all_words.update(set(word for word in text_token))
	sys.stdout.write("%4d/%04d\r"%(i,len(df)))
	sys.stdout.flush()


for i in range(0, len(df)-test_size):
	f = codecs.open('data/TCSS555/Train/Text/'+df['userid'][i]+'.txt', encoding='latin-1')
	text = f.readline().replace('\\','').lower()
	text_token = nltk.word_tokenize(text)
	# in this moment the all_words doesn't contain all words in train data
	# so text_feature doesn't contain all of data info.
	text_feature = {word : (word in text_token) for word in all_words}
	train.append([text_feature, df['gender'][i]])
	sys.stdout.write("%4d/%04d\r"%(i,len(df)))
	sys.stdout.flush()
sys.stdout.write("\n")

classifier = nltk.NaiveBayesClassifier.train(train)
classifier.show_most_informative_features()

match = 0
for i in range(0, 100):
	test = train[8000+i][0]
	test_gen = train[8000+i][1]
	if test_gen == classifier.classify(test):
		match = match+1
	sys.stdout.write("%4d/%04d\r"%(i,len(df)))
	sys.stdout.flush()



def save_classifier(classifier):
	cf = open('my_classifier.pickle', 'wb')
	pickle.dump(classifier, cf)
	cf.close()
	return

def load_classifier():
	cf = open('my_classifier.pickle', 'rb')
	classifier = pickle.load(cf)
	cf.close()
	return classifier


def save_train(train):
	cf = open('my_train.pickle', 'wb')
	pickle.dump(train, cf)
	cf.close()
	return

def load_train():
	cf = open('my_train.pickle', 'rb')
	train = pickle.load(cf)
	cf.close()
	return train

def save_allwords(allwords):
	cf = open('my_allwords.pickle', 'wb')
	pickle.dump(allwords, cf)
	cf.close()
	return 

def load_allwords():
	cf = open('my_allwords.pickle', 'rb')
	allwords = pickle.load(cf)
	cf.close()
	return allwords


# for i in range(0, len(df)):
# all_words = set(word.lower() for passage in text for word in nltk.word_tokenize(text))
# train = []
# for i in range(0, len(df)):
# 	f = open('data/TCSS555/Train/Text/'+df['userid'][i]+'.txt', 'r')
# 	text = f.readline().replace('\\',''))
# 	text_feature =
# 	train.append((text_feature, df['gender'][i].apply(int)))





# all_words = set(word for word in nltk.word_tokenize(text[0]))
# t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]

# test_sentence = "This is the best band I've ever heard!"
# test_sent_features = {word.lower(): (word in word_tokenize(test_sentence.lower())) for word in all_words}

# sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
# tokens = nltk.word_tokenize(sentence)
# tokens
# tagged = nltk.pos_tag(tokens)
# tagged[0:6]
# entities = nltk.chunk.ne_chunk(tagged)
# entities
# from nltk.corpus import treebank
# t = treebank.parsed_sents('wsj_0001.mrg')[0]
# t.draw()




df = pandas.read_csv('data/TCSS555/Test/Profile/Profile.csv')


# f = open('data/TCSS555/Train/Text/'+df['userid'][0]+'.txt', 'r')
# text = f.readline()
# tokens = nltk.word_tokenize(text[0].replace('\\',''))
# tokens
# tagged = nltk.pos_tag(tokens)
# tagged
# entities = nltk.chunk.ne_chunk(tagged)


Test = []
for i in range(0, len(df)):
	all_words2 = set()
	f = codecs.open('data/TCSS555/Test/Text/'+df['userid'][i]+'.txt', encoding='latin-1')
	text = f.readline().replace('\\','').lower()
	text_token = nltk.word_tokenize(text)
	all_words2.update(set(word for word in text_token))
	# in this moment the all_words doesn't contain all words in train data
	# so text_feature doesn't contain all of data info.
	text_feature = {word : (word in text_token) for word in all_words}
	Test.append([text_feature, df['gender'][i]])
	sys.stdout.write("%4d/%04d\r"%(i,len(df)))
	sys.stdout.flush()
sys.stdout.write("\n")

match = 0
for t in Test:
	if t[1] == classifier.classify(t[0]):
		match = match+1