import pandas
import nltk
import numpy
import codecs
import sys
import pickle


profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
n_test = 1000

age_dict = {1 : '18-24', 2 : '25-34', 3 : '35-49', 4 : '50-xx'}

# get one df contains like_id and count
df_like_id_num = pandas.DataFrame({'count' : relation.groupby(['like_id']).size()}).reset_index()
df_like_id_num_10up = df_like_id_num.query('count > 20').sort_values('count', ascending=True).reset_index()

df_like_id_num_10up['m_g'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_g'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_age'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_age'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_ope'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_ope'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_con'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_con'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_ext'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_ext'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_agr'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_agr'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['m_neu'] = pandas.Series(index = df_like_id_num_10up.index)
df_like_id_num_10up['v_neu'] = pandas.Series(index = df_like_id_num_10up.index)

for i in range(0, len(df_like_id_num_10up)):
	userid_list = relation.query("like_id == %d"%df_like_id_num_10up['like_id'][i])['userid'].tolist()
	df_matched_profile = profile.query("userid == '%s'"%userid_list[0])
	for uid in userid_list[1:]:
		entry = profile.query("userid == '%s'"%uid)
		df_matched_profile = df_matched_profile.append(entry)
	mean = df_matched_profile.mean()
	var = df_matched_profile.var()
	df_like_id_num_10up.set_value(i,'m_g', mean['gender'])
	df_like_id_num_10up.set_value(i,'m_age', mean['age'])
	df_like_id_num_10up.set_value(i,'m_ope', mean['ope'])
	df_like_id_num_10up.set_value(i,'m_con', mean['con'])
	df_like_id_num_10up.set_value(i,'m_ext', mean['ext'])
	df_like_id_num_10up.set_value(i,'m_agr', mean['agr'])
	df_like_id_num_10up.set_value(i,'m_neu', mean['neu'])
	df_like_id_num_10up.set_value(i,'v_g', var['gender'])
	df_like_id_num_10up.set_value(i,'v_age', var['age'])
	df_like_id_num_10up.set_value(i,'v_ope', var['ope'])
	df_like_id_num_10up.set_value(i,'v_con', var['con'])
	df_like_id_num_10up.set_value(i,'v_ext', var['ext'])
	df_like_id_num_10up.set_value(i,'v_agr', var['agr'])
	df_like_id_num_10up.set_value(i,'v_neu', var['neu'])
	sys.stdout.write("%4d/%04d\r"%(i,len(df_like_id_num_10up)))
	sys.stdout.flush()

df_like_id_num_10up.to_csv('df_like_id_num_10up.csv')
