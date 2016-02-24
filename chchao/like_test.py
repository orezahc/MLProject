import pandas
import nltk
import numpy
import codecs
import sys
import pickle


class like_test:
	def __init__(self):
		self.mean_value_table = pandas.read_csv('chchao/df_like_id_num_10up.csv')
		return

	def info(self):
		print(self.mean_value_table)
		return

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
		if age/n_like_id < 25:
			output_dict['age'] = 'xx-24'
		elif age/n_like_id < 35:
			output_dict['age'] = '25-34'
		elif age/n_like_id < 50:
			output_dict['age'] = '35-49'
		else:
			output_dict['age'] = '50-xx'

		if gender/n_like_id >= 0.5:
			return 1 #output_dict['gender'] = 'female'
		else: 
			return 0 #output_dict['gender'] = 'male'
		
		# output_dict['ope'] = ope/n_like_id
		# output_dict['con'] = con/n_like_id
		# output_dict['ext'] = ext/n_like_id
		# output_dict['agr'] = agr/n_like_id
		# output_dict['neu'] = neu/n_like_id
		# return "<user\nId=\"%s\"\nage_group=\"%s\"\ngender=\"%s\"\nextrovert=\"%d\"\nneurotic=\"%d\"\nagreeable=\"%d\"\nconscientious=\"%d\"\nopen=\"%d\"\n/>"%(userid, output_dict['age'], output_dict['gender'], output_dict['ext'], output_dict['neu'], output_dict['agr'], output_dict['con'], output_dict['ope'])


if __name__ == '__main__':
	profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
	relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
	x = like_test()
	hit = 0
	for i in range(0, 1000):
		if profile['gender'][i] == x.test(relation, profile['userid'][0]):
			hit += 1
	print("%d/%d"%(hit, 1000))
	

	# print(x.test(relation, profile['userid'][0]))
