import pandas
import nltk
import numpy
import codecs
import sys
from sklearn import datasets, linear_model
from like_test import like_test

profile = pandas.read_csv('data/TCSS555/Train/Profile/Profile.csv')
relation = pandas.read_csv('data/TCSS555/Train/Relation/Relation.csv')
mean_value_table = pandas.read_csv('chchao/df_like_id_num_10up.csv')

n_profile = len(profile)
like = like_test()
x = []
y = []
n_train = 5000
n_test = 1000

for i in range(0, n_train+n_test):
	feature = like.lr_formating(relation, profile['userid'][i])
	x.append(feature)
	y.append([profile['ope'][i]])
	sys.stdout.write("training : %4d/%04d\r"%(i,n_train+n_test))
	sys.stdout.flush()
sys.stdout.write("\n")


x_train = x[:-1000]
x_test = x[-1000:]

y_train = y[:-1000]
y_test = y[-1000:]

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

