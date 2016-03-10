import sys
import pickle
import pandas as pd
from oxford import oxford_feature, oxford_gender_classfier

class gender:
	def __init__(self):
		self.clf_knn_gender_oxford = oxford_gender_classfier()
		cf = open('/home/itadmin/MLProject/clf_gender_SGDC_text_like.pickle', 'rb')
		self.clf_mnb_gender_like = pickle.load(cf)

		return

	def predict(self, like_text, likes_feature, oxford, userid):
		isContainFace, feature = oxford_feature(oxford, userid)
		if isContainFace:
			gender = self.clf_knn_gender_oxford.predict([feature])
			if gender == 1:
				gender = self.clf_mnb_gender_like.predict(like_text)
		else:
			gender = self.clf_mnb_gender_like.predict(like_text)
			#gender = self.clf_mnb_gender_like.predict(likes_feature)
		return gender
