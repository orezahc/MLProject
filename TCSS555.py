import pandas
import nltk
import numpy
import codecs
import sys
import pickle
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from chchao.like_test import like_test
from chchao.baseline import baseline

if len(sys.argv) < 2:
	print "Invalid arguments!"
	exit()

input_dir = sys.argv[1]
output_dir = sys.argv[2]



try:
	profile = pandas.read_csv(input_dir+'profile/profile.csv')
	print("reading profile.csv successed.")
except:
	print("Error: reading profile.csv failed.")
	exit()

try:
	relation = pandas.read_csv(input_dir+'relation/relation.csv')
	print("reading relation.csv successed.")
except:
	print("Error: reading relation.csv failed.")
	exit()


baseline = baseline()
like_test = like_test()

cf = open('/home/itadmin/MLProject/classifier_like3up_nb_age.pickle', 'rb')
age_predict = pickle.load(cf)



len_profile = len(profile)



for i in range(0, len_profile):
	userid = profile['userid'][i]
	age = age_predict.predict([like_test.lr_formating(relation, userid)])[0]
	print(userid + " age : %d"%age)
	output_dict = baseline.predict()
	print(output_dict)
	output_dict['age'] = like_test.lr_age_get_str(age)
	f = open(output_dir+userid+'.xml', 'w')
	f.write("<user\nId=\"%s\"\nage_group=\"%s\"\ngender=\"%s\"\nextrovert=\"%d\"\nneurotic=\"%d\"\nagreeable=\"%d\"\nconscientious=\"%d\"\nopen=\"%d\"\n/>"%(userid, output_dict['age'], output_dict['gender'], output_dict['ext'], output_dict['neu'], output_dict['agr'], output_dict['con'], output_dict['ope'])
)
	sys.stdout.write("%d/%d\r"%(i,len_profile))
	sys.stdout.flush()
