import numpy as np
import pandas as pd

from Evaluate import *

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

import scipy.sparse


# vector hóa train set
def Train(train_X, train_y, test_X, type_train=None):
	# khởi tạo model cho từng loại phân loại
	if type_train == 'LO':
		logist = LogisticRegression()
		logist.fit(train_X, train_y)
		reg_predicted_logist = logist.predict(test_X)
		return reg_predicted_logist
	elif type_train == 'DC':
		DC = DecisionTreeClassifier()
		DC.fit(train_X, train_y)
		reg_predicted_DC = DC.predict(test_X)
		return reg_predicted_DC
	elif type_train == 'NB':
		NB = BernoulliNB()
		NB.fit(train_X.toarray(), train_y)
		reg_predicted_NB = NB.predict(test_X.toarray())
		return reg_predicted_NB
	elif type_train == 'KNN':
		KNN = KNeighborsClassifier(n_neighbors=10, weights='distance')
		KNN.fit(train_X, train_y)
		reg_predicted_KNN = KNN.predict(test_X.toarray())
		return reg_predicted_KNN

def main():

	label = np.load('C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/label.npy')

	data = pd.read_json("C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/Sarcasm_Headlines_Dataset.json", lines = True)
	title = data['headline']

	text_transformer = TfidfVectorizer()

	train_X = text_transformer.fit_transform(title[:20000])
	train_y = label[:20000]
	test_X = text_transformer.transform(title[20000:])
	test_y = label[20000:]

	reg_predicted = Train(train_X, train_y, test_X, 'NB')

	TP = find_TP(reg_predicted, test_y)
	FP = find_FP(reg_predicted, test_y)
	P = find_P(test_y)
	Ac = Accuracy(reg_predicted, test_y)

	# print("TP = ", TP)
	# print("FP = ", FP)
	# print("P  = ", P)

	Precision = TP/(TP+FP)
	Recall = TP/P

	print("Accuracy = ", Ac)

	print("Precision = ", Precision)
	print("Recall = ", Recall)
	 
	print("F1_score:", F1_score(Precision, Recall))

if __name__ == '__main__':
	main()