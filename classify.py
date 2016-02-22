import pandas
import nltk
import numpy
import codecs
import sys
import pickle


if len(sys.argv) < 2:
	print "Invalid filename"
	exit()

filename = sys.argv[1]

df = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')

def load_classifier(filename):
	cf = open(filename, 'rb')
	classifier = pickle.load(cf)
	cf.close()
	return classifier

all_words = set()
test_size = 1000
test = []
for i in range(len(df)-test_size, len(df)):
	f = codecs.open('data/TCSS555/Train/Text/'+df['userid'][i]+'.txt', encoding='latin-1')
	text = f.readline().replace('\\','').lower()
	text_token = nltk.word_tokenize(text)
	all_words.update(set(word for word in text_token))
	# in this moment the all_words doesn't contain all words in train data
	# so text_feature doesn't contain all of data info.
	text_feature = {word : (word in text_token) for word in text_token}
	test.append([text_feature, df['gender'][i]])
	sys.stdout.write("%4d/%04d\r"%(i,len(df)))
	sys.stdout.flush()
sys.stdout.write("\n")


classifier = load_classifier(filename)
classifier.show_most_informative_features()

match = 0
i=0
for test_data in test:
	test = test_data[0]
	test_gen = test_data[1]
	if test_gen == classifier.classify(test):
		match = match+1
	i=i+1
	sys.stdout.write("%4d/%04d\r"%(i,len(test)))
	sys.stdout.flush()

print("===============")
print("accuracy:%d/%d"%(match,test_size))