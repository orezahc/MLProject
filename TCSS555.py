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
from chchao.gender import gender
from boruishi.text_Ope_knn_test import text_ope_knn_test
from boruishi.liwc_lreg import *

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

try:
	oxford_csv = pandas.read_csv(input_dir+'oxford.csv')
	print("reading oxford.csv successed.")
except:
	print("Warning: reading oxford.csv failed.")


baseline = baseline()
like_test = like_test()

cf = open('/home/itadmin/MLProject/clf_age_SGDC_liketext.pickle', 'rb')
age_predict = pickle.load(cf)

# cf = open('/home/itadmin/MLProject/clf_like_mnb_gender.pickle', 'rb')
# g_predict = pickle.load(cf)
g_predict = gender()

len_profile = len(profile)



try:
	ope_p = text_ope_knn_test()
except:
	print('load personalities predictors failed!')


regr_ope, regr_ext, regr_con, regr_agr, regr_neu = train()

cnt_non_like_id = 0
cnt = 0
for i in range(0, len_profile):
	userid = profile['userid'][i]
	like_ids = [like_test.lr_formating(relation, userid)]
	cnt+=1
	if 1 not in like_ids:
		cnt_non_like_id+=1

        text = codecs.open(input_dir+'text/'+userid+'.txt', encoding='latin-1').readline()
        like_text = text+' '+str(relation.query("userid == '%s'"%userid)['like_id'].tolist())[1:-1]
	age = age_predict.predict([like_text])[0]
	gender = g_predict.predict([like_text], like_ids, oxford_csv, userid)[0]
#	print(userid)
#	print("gender "+str(gender))
	output_dict = baseline.predict()

	
	pern = classify(regr_ope, regr_ext, regr_con, regr_agr, regr_neu, input_dir, userid)
	output_dict['ext'] = pern[1]	
	output_dict['arg'] = pern[3]
	output_dict['ope'] = pern[0]
	output_dict['con'] = pern[2]
	output_dict['neu'] = pern[4]
	output_dict['gender'] = like_test.lr_g_get_str(gender)
	output_dict['age'] = like_test.lr_age_get_str(age)
#	print("gender " + str(g_predict.predict_proba(like_ids)))
#	print("age " +str(age_predict.predict_proba(like_ids)))
	f = open(output_dir+userid+'.xml', 'w')
	f.write("<user\nId=\"%s\"\nage_group=\"%s\"\ngender=\"%s\"\nextrovert=\"%f\"\nneurotic=\"%f\"\nagreeable=\"%f\"\nconscientious=\"%f\"\nopen=\"%f\"\n/>"%(userid, output_dict['age'], output_dict['gender'], output_dict['ext'], output_dict['neu'], output_dict['agr'], output_dict['con'], output_dict['ope'])
)
	sys.stdout.write("%d/%d\r"%(i,len_profile))
	sys.stdout.flush()
sys.stdout.write("\n")
import datetime
timestr = datetime.datetime.now().strftime("%Y%m%d%H%M")
try:
	logf = open('/home/itadmin/MLProject/log/log', 'a')
	logf.write(timestr+': total:%d non_like_id:%d\n'%(cnt,cnt_non_like_id))
except:
	print("open log failed!")

