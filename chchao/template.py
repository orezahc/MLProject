import pandas
import sys
from chchao.like_test import like_test

profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
a = []
b = []
n_size = 9500
n_train = 5000
l = like_test()

for i in range(0, n_size):
	l = x.lr_formating(relation, profile['userid'][i])
	a.append(l)
	b.append(profile['gender'][i])
	sys.stdout.write("training : %4d/%04d\r"%(i,n_size))
	sys.stdout.flush()


from sklearn.random_projection import sparse_random_matrix

svd = TruncatedSVD(n_components=1000, random_state=42)
svd.fit(a[:n_train])
aa = svd.transform(a)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(aa[:n_train], y[:n_train]) 
yy = neigh.predict(aa[n_train:])

