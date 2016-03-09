import sys
import pickle
import pandas as pd
from oxford import oxford_feature, oxford_gender_classfier

class gender:
	def __init__(self):
		self.clf_knn_gender_oxford = oxford_gender_classfier()
		cf = open('/home/itadmin/MLProject/clf_like_mnb_gender.pickle', 'rb')
		self.clf_mnb_gender_like = pickle.load(cf)

		return

	def predict(self, likes_feature, oxford, userid):
		isContainFace, feature = oxford_feature(oxford, userid)
		if isContainFace:
			gender = self.clf_knn_gender_oxford.predict([feature])
		else:
			gender = self.clf_mnb_gender_like.predict(likes_feature)
		return gender
