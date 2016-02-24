import pandas
import nltk
import numpy
import codecs
import sys
import pickle


def like_format(relation, userid):
	return relation.query("userid == '%s'"%userid)['like_id'].tolist()