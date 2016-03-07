import pandas as pd
import sys
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import os
import pickle

def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def mid(p0, p1):
	return [(p0[0]+p1[0])/float(2), (p0[1]+p1[1])/float(2)]

def oxford_feature(oxford, userid):
	result = []
	o = oxford.query("userId == '%s'"%userid)
	i = 0
	if len(o) == 0:
		return False, []
	elif len(o) > 1:
		bigface = 0
		for j in range(0, len(o)):
			if int(o.iloc[j]['faceRectangle_width'])*int(o.iloc[j]['faceRectangle_height']) > bigface:
				bigface = o.iloc[j]['faceRectangle_width']
				o.iloc[j]['faceRectangle_height']
				i = j

	# face features vectorization.

	# eye_len = distance([o.iloc[i]['eyeLeftOuter_x'], o.iloc[i]['eyeLeftOuter_y']], [o.iloc[i]['eyeLeftInner_x'], o.iloc[i]['eyeLeftInner_y']])
	# eye_hight = distance([o.iloc[i]['eyeLeftTop_x'], o.iloc[i]['eyeLeftTop_y']], [o.iloc[i]['eyeLeftBottom_x'], o.iloc[i]['eyeLeftBottom_y']])/eye_len
	# eyebrow_len = distance([o.iloc[i]['eyebrowLeftInner_x'], o.iloc[i]['eyebrowLeftInner_y']], [o.iloc[i]['eyebrowLeftOuter_y'], o.iloc[i]['eyebrowLeftOuter_y']])/eye_len
	# outer_eye_eyebrow_len = distance([o.iloc[i]['eyebrowLeftOuter_x'], o.iloc[i]['eyebrowLeftOuter_y']], [o.iloc[i]['eyeLeftOuter_y'], o.iloc[i]['eyeLeftOuter_y']])/eye_len
	# inner_eye_eyebrow_len = distance([o.iloc[i]['eyebrowLeftInner_x'], o.iloc[i]['eyebrowLeftInner_y']], [o.iloc[i]['eyeLeftInner_y'], o.iloc[i]['eyeLeftInner_y']])/eye_len
	# eye_noseroot_len = distance([o.iloc[i]['eyeLeftInner_x'], o.iloc[i]['eyeLeftInner_y']], [o.iloc[i]['noseRootLeft_x'], o.iloc[i]['noseRootLeft_y']])/eye_len
	# noseroot_len = distance([o.iloc[i]['noseRootRight_x'], o.iloc[i]['noseRootRight_y']], [o.iloc[i]['noseRootLeft_x'], o.iloc[i]['noseRootLeft_y']])/eye_len
	# nosealartop_len = distance([o.iloc[i]['noseRightAlarTop_x'], o.iloc[i]['noseRightAlarTop_y']], [o.iloc[i]['noseLeftAlarTop_x'], o.iloc[i]['noseLeftAlarTop_y']])/eye_len
	# nosealarout_len = distance([o.iloc[i]['noseRightAlarOutTip_x'], o.iloc[i]['noseRightAlarOutTip_y']], [o.iloc[i]['noseLeftAlarOutTip_x'], o.iloc[i]['noseLeftAlarOutTip_y']])/eye_len
	# nose_root_alartop_len = distance(mid([o.iloc[i]['noseRootRight_x'], o.iloc[i]['noseRootRight_y']], [o.iloc[i]['noseRootLeft_x'], o.iloc[i]['noseRootLeft_y']]), mid([o.iloc[i]['noseRightAlarTop_x'], o.iloc[i]['noseRightAlarTop_y']], [o.iloc[i]['noseLeftAlarTop_x'], o.iloc[i]['noseLeftAlarTop_y']]))/eye_len
	# nose_alartop_outtip_len = distance(mid([o.iloc[i]['noseRightAlarTop_x'], o.iloc[i]['noseRightAlarTop_y']], [o.iloc[i]['noseLeftAlarTop_x'], o.iloc[i]['noseLeftAlarTop_y']]), mid([o.iloc[i]['noseRightAlarOutTip_x'], o.iloc[i]['noseRightAlarOutTip_y']], [o.iloc[i]['noseLeftAlarOutTip_x'], o.iloc[i]['noseLeftAlarOutTip_y']]))/eye_len

	# result.append(eye_hight)
	# result.append(eyebrow_len)
	# result.append(outer_eye_eyebrow_len)
	# result.append(inner_eye_eyebrow_len)
	# result.append(eye_noseroot_len)

	# result.append(noseroot_len)
	# result.append(nosealartop_len)
	# result.append(nosealarout_len)
	# result.append(nose_root_alartop_len	)
	# result.append(nose_alartop_outtip_len)

	result.append(o.iloc[i]['facialHair_mustache'])
	result.append(o.iloc[i]['facialHair_beard'])
	result.append(o.iloc[i]['facialHair_sideburns'])

	return True, result


def oxford_gender_classfier():
	pickle_path = '/home/itadmin/MLProject//clf_gender_knn_oxford.pickle'

	try:
		cf = open(pickle_path, 'rb')
		knn = pickle.load(cf)
		print(knn)
	except:
		p = pd.read_csv('/home/itadmin/MLProject/data/TCSS555/Train/Profile/Profile.csv')
		o = pd.read_csv('/home/itadmin/MLProject/data/oxford.csv')
		features=[]
		genders=[]
		uid=''
		luid=''
		for i in range(0, len(o)):
			uid = o.iloc[i]['userId']
			if uid == luid:
				continue
			luid = uid 
			isContainFace, feature = oxford_feature(o, uid)
			features.append(feature)
			genders.append(p.query("userid == '%s'"%uid).iloc[0]['gender'])
			sys.stdout.write("preprocessing data : %4d/%04d\r"%(i,len(o)))
			sys.stdout.flush()
		sys.stdout.write("\n")
		print(features)
		print(genders)
		knn = KNeighborsClassifier(n_neighbors=10)
		score = cross_val_score(knn, features, genders, cv=10).mean()
		print("kNN(neighbors=10) using oxford to predict gender:")
		print("10-fold accuracy : %f"%score)

		knn.fit(features, genders)
		print(knn)
		
		print('Saving classifier to '+pickle_path)
		cf = open(pickle_path, 'wb')
		pickle.dump(knn, cf)

	return knn


