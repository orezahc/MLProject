import pandas
import sys
from chchao.like_test import like_test

profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
a = []
b = []
_a = []
_b = []
n_size = 9500
n_train = 8500
knn = 5
x = like_test()

for i in range(0, n_size):
	l = x.lr_formating(relation, profile['userid'][i])
	a.append(l)
	b.append(x.lr_age_format(profile['age'][i]))
	if 1 in a[i]:
		_a.append(a[i])
		_b.append(b[i])
	sys.stdout.write("training : %4d/%04d\r"%(i,n_size))
	sys.stdout.flush()

print("knn : %d"%knn)
print("n_train : %d"%n_train)
print("len(_a):%d len(_b):%d"%(len(_a), len(_b)))

# from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=1000, random_state=42)
# svd.fit(_a[:n_train])
# aa = svd.transform(_a)


# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(aa[:n_train], _b[:n_train]) 
# # print("score : %f"%neigh.score(aa[n_train:], _b[n_train:]))

# yy = neigh.predict(aa[n_train:])

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(_a[:n_train], _b[:n_train])
yy = gnb.predict(_a[n_train:])

cnt = 0


for i in range(0, len(yy)):
	if yy[i] == _b[n_train+i]:
		cnt += 1

print("my score : %d/%d"%(cnt, len(yy)))
print(yy)
print(_b[n_train:])