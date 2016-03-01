import pandas
import nltk
import numpy
import codecs
import sys
import pickle

class baseline:
	def __init__(self):
		profile = pandas.read_csv('/home/itadmin/MLProject/data/TCSS555/Train/Profile/Profile.csv')
		mean = profile.mean()
		self.output_dict = {}
		if mean['age'] < 25:
			self.output_dict['age'] = "xx-24"
		elif mean['age'] < 35:
			self.output_dict['age'] = "25-34"
		elif mean['age'] < 50:
			self.output_dict['age'] = "35-49"
		else:
			self.output_dict['age'] = "50-xx"

		if mean['gender'] > 0.5:	
			self.output_dict['gender'] = 1
		else:
			self.output_dict['gender'] = 0

		self.output_dict['ope'] = mean['ope']
		self.output_dict['agr'] = mean['agr']
		self.output_dict['ext'] = mean['ext']
		self.output_dict['con'] = mean['con']
		self.output_dict['neu'] = mean['neu']

	def predict(self):
		return self.output_dict


if __name__ == '__main__':
	profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
	x = baseline()
	hit = 0
	y=0
	for i in range(0, 9500):
		d = x.predict()
		result = profile['gender'][i]
		if d['gender'] == result:
			hit += 1
		y += (d['con']-profile['con'][i])**2
	print("%d/%d"%(hit, 1000))
	print(y/float(9500))
