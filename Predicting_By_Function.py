import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import scipy.sparse


# vector hóa train set
def Train(train_X, train_y, test_X, type_train=None):
	# khởi tạo model cho từng loại phân loại
	if type_train == 'KNN':
		KNN = KNeighborsClassifier()
		KNN.fit(train_X, train_y)
		reg_predicted_KNN = KNN.predict(test_X)
		return reg_predicted_KNN
	elif type_train == 'LO':
		logist = LogisticRegression()
		logist.fit(train_X, train_y)
		reg_predicted_logist = logist.predict(test_X.toarray())
		return reg_predicted_logist
	elif type_train == 'DC':
		DC = DecisionTreeClassifier()
		DC.fit(train_X, train_y)
		reg_predicted_DC = DC.predict(test_X)
		return reg_predicted_DC
	elif type_train == 'NB':
		NB = GaussianNB()
		NB.fit(train_X.toarray(), train_y)
		reg_predicted_NB = NB.predict(test_X.toarray())
		return reg_predicted_NB


# Tìm giá trị TP
def find_TP(predict, test_y):
	_and = np.multiply(test_y, predict)
	return np.count_nonzero(_and)

# Tìm giá trị FP
def find_FP(predict, test):
	test_y = np.array(test)
	count = 0
	for i in range(predict.shape[0]):
		if predict[i] == 1:
			if test_y[i] == 0:
				count += 1
	return count

# Tính số nhãn bằng 1 trong tập
def find_P(test):
	return np.count_nonzero(test)

# Tính F1_score
def F1_score(Precision, Recall):
	return (Precision*Recall)/(Precision+Recall)*2


def main():

	label = np.load('C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/label.npy')
	title = scipy.sparse.load_npz('C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/tfidf.npz')

	train_X = title[:20000]
	train_y = label[:20000]
	test_X = title[20000:]
	test_y = label[20000:]

	reg_predicted = Train(train_X, train_y, test_X, 'LO')

	TP = find_TP(reg_predicted, test_y)
	FP = find_FP(reg_predicted, test_y)
	P = find_P(test_y)

	# print("TP = ", TP)
	# print("FP = ", FP)
	# print("P  = ", P)

	Precision = TP/(TP+FP)
	Recall = TP/P
	print("Precision = ", Precision)
	print("Recall = ", Recall)
	 
	print("F1_score:", F1_score(Precision, Recall))

if __name__ == '__main__':
	main()