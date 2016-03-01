import pandas
import nltk
import numpy
import codecs
import sys
import pickle
from bisect import bisect_left


class like_test:
	def __init__(self):
		profile = pandas.read_csv('/home/itadmin/MLProject/data/TCSS555/Train/Profile/Profile.csv')
		relation = pandas.read_csv('/home/itadmin/MLProject/data/TCSS555/Train/Relation/Relation.csv')
		df_like_id_num = pandas.DataFrame({'count' : relation.groupby(['like_id']).size()}).reset_index()
		df_like_id_num_10up = df_like_id_num.query('count > 2').sort_values('count', ascending=True).reset_index()
		self.mean_value_table = df_like_id_num_10up

		self.like_id_list = self.mean_value_table['like_id'].tolist()
		self.like_id_list.sort()
		self.like_id_list_len = len(self.like_id_list)
		self.age_dict = {0 : 'xx-24', 1 : '25-34', 2 : '35-49', 3 : '50-xx'}
		print(self.age_dict)
		print("length like_id_list : %d"%(self.like_id_list_len))
		print("min : %d max : %d"%(self.mean_value_table['count'][0],self.mean_value_table['count'][len(self.mean_value_table['count'])-1]))

		return

	def info(self):
		print(self.mean_value_table)
		return

	def lr_formating(self, relation, userid):
		like_list = relation.query("userid == '%s'"%userid)['like_id'].tolist()
		like_feature = [0]*self.like_id_list_len
		len_like_list = len(like_list)
		for i in range(0, len_like_list):
			pos = bisect_left(self.like_id_list, like_list[i], 0, self.like_id_list_len)
			if pos < self.like_id_list_len:
				like_feature[pos] = 1

		return like_feature

	def lr_age_format(self, age):
		age_group = [24, 34, 49]
		pos = bisect_left(age_group, age, 0, len(age_group))
		return pos

	def lr_age_get_str(self, agegroup):
		return self.age_dict[agegroup]

	def test(self, relation, userid):
		like_list = relation.query("userid == '%s'"%userid)['like_id'].tolist()

		
		# gender
		n_like_id = 0
		gender = 0
		age = 0
		ope = 0
		con = 0
		ext = 0
		agr = 0
		neu = 0
		# print(like_list)
		for like_id in like_list:
			entry = self.mean_value_table.query("like_id == %d"%like_id).reset_index()
			if len(entry) == 0:
				continue
			n_like_id += 1
			# print(entry['m_g'][0])
			gender += entry['m_g'][0]
			age += entry['m_age'][0]
			ope += entry['m_ope'][0]
			con += entry['m_con'][0]
			ext += entry['m_ext'][0]
			agr += entry['m_agr'][0]
			neu += entry['m_neu'][0]

		output_dict = {}
		if n_like_id == 0:
			output_dict['gender'] = 'female'
			output_dict['age'] = 'xx-24'
			output_dict['ope'] = 2.5
			output_dict['con'] = 2.5
			output_dict['ext'] = 2.5
			output_dict['agr'] = 2.5
			output_dict['neu'] = 2.5
			return output_dict

		if age/n_like_id < 25:
			output_dict['age'] = 'xx-24'
		elif age/n_like_id < 35:
			output_dict['age'] = '25-34'
		elif age/n_like_id < 50:
			output_dict['age'] = '35-49'
		else:
			output_dict['age'] = '50-xx'

		if gender/n_like_id >= 0.5:
			output_dict['gender'] = 'female'
		else: 
			output_dict['gender'] = 'male'
		
		output_dict['ope'] = ope/n_like_id
		output_dict['con'] = con/n_like_id
		output_dict['ext'] = ext/n_like_id
		output_dict['agr'] = agr/n_like_id
		output_dict['neu'] = neu/n_like_id
		return output_dict
		# return "<user\nId=\"%s\"\nage_group=\"%s\"\ngender=\"%s\"\nextrovert=\"%d\"\nneurotic=\"%d\"\nagreeable=\"%d\"\nconscientious=\"%d\"\nopen=\"%d\"\n/>"%(userid, output_dict['age'], output_dict['gender'], output_dict['ext'], output_dict['neu'], output_dict['agr'], output_dict['con'], output_dict['ope']))

if __name__ == '__main__':
	profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
	relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
	x = like_test()
	hit = 0
	for i in range(0, 1000):
		d = x.test(relation, profile['userid'][i])
		result = 0
		if d['gender'] == 'female':
			result = 1
		if profile['gender'][i] == result:
			hit += 1
	print("%d/%d"%(hit, 1000))
	

	# print(x.test(relation, profile['userid'][0]))
