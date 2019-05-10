import pandas as pd
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
import string
import re
import nltk
import numpy as np

def ReadData(path):
	# đọc file
	raw_df = pd.read_json(path, lines = True)
	return raw_df

def PreProcessing_Data(cleaned_df, stopwords):
	# loại bỏ cột article_link
	cleaned_df.pop('article_link')

	# loại bỏ những dữ liệu bị thiếu
	cleaned_df.dropna()

	# xử lý cột headline với stopwords
	cleaned_df['headline'] = cleaned_df['headline'].apply(lambda s : ' '.join([re.sub(r'\W+', '', word.lower()) for word in s.split(' ') if word not in stopwords]))

	return cleaned_df


# vector hóa train set
def Train(train_X, train_y, test_X, test_y):
	# Khởi tạo model tf-idf
	tf_idf = TfidfVectorizer()

	# khởi tạo model cho từng loại phân loại
	KNN = KNeighborsClassifier()

	# Pipeline
	reg_test_clf_KNN = Pipeline([('Tf*idf', tf_idf), ('reg', KNN)])

	# train
	reg_test_clf_KNN.fit(train_X, train_y)


	# dự đoán data test
	reg_predicted_KNN = reg_test_clf_KNN.predict(test_X)

	return reg_predicted_KNN

def find_TP(predict, test_y):
	_and = np.multiply(test_y, predict)
	return np.count_nonzero(_and)

def find_FP(predict, test):
	test_y = np.array(test)
	count = 0
	for i in range(predict.shape[0]):
		if predict[i] == 1:
			if test_y[i] == 0:
				count += 1
	return count

def find_P(test):
	return np.count_nonzero(test)

# Tính F1_score
def F1_score(Precision, Recall):
	return (Precision*Recall)/(Precision+Recall)*2

def main():
	# # Tải bộ stopwords từ nltk
	# nltk.download('stopwords')

	# # Tải về bộ model tạo bộ từ điển token
	# nltk.download('punkt')

	path = "C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/Sarcasm_Headlines_Dataset.json"

	cleaned_df = ReadData(path)

	_stopwords = stopwords.words('english') + list(string.punctuation)

	Data = PreProcessing_Data(cleaned_df, _stopwords)

	# chia train và test
	train_X = Data['headline'][ :20000]
	train_y = Data['is_sarcastic'][ :20000]
	test_X = Data['headline'][20000:]
	test_y = Data['is_sarcastic'][20000:]

	reg_predicted_logist = Train(train_X, train_y, test_X, test_y)

	TP = find_TP(reg_predicted_logist, test_y)
	FP = find_FP(reg_predicted_logist, test_y)
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