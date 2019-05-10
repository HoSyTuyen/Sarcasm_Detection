import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spmatrix
from scipy.sparse import coo_matrix

def eliminate_stopwords(Data):
	'''
	Data là dữ liệu ban đầu

	Hàm trả về dữ liệu sau khi đã loại bỏ stopwords và các ký tự đặc biệt
	'''
	# Làm sách dữ liệu
	cleaned_df = Data
	# Loại bỏ cột article_link
	cleaned_df.pop('article_link')
	# Loại bỏ những dữ liệu bị thiếu
	cleaned_df.dropna()
	# Load bộ từ điển stopwords trong tiếng anh
	stop = set(stopwords.words('english'))

	# Xử lý cột headline
	cleaned_df['headline'] = Data['headline'].apply(lambda s : ' '.join([re.sub(r'\W+', '', word.lower()) for word in s.split(' ') if word not in stop]))

	# Loại bỏ ký tự đặc biệt
	processed_data = cleaned_df['headline'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',' ')
	processed_data = processed_data.str.replace(r'[^\w\d\s]', ' ')
	processed_data = processed_data.str.replace(r'\s+', ' ')
	processed_data = processed_data.str.replace(r'^\s+|\s+?$', ' ')
	processed_data = processed_data.str.replace(r'\d+',' ')
	processed_data = processed_data.str.lower()
	return processed_data

def create_list_diction(processed_data):
	'''
	processed_data là dữ liệu đã qua xử lý loại bỏ các stop words

	Hàm trả về 1 dictionary chứa các từ có trong processed_data

	'''
	# danh sách tất cả các token
	diction = []
	# lặp cho toàn bộ dữ liệu
	for headline in processed_data:
		# tách token
	  token = word_tokenize(headline)

	  for w  in token:
	    diction.append(w)

	# khởi tạo diction từ danh sách token
	diction = nltk.FreqDist(diction)
	words = list(diction.keys())

	# chuyển token sang từ điển
	return words

def TF(data, diction, dtype = int):
	'''
	data là dữ liệu đã qua xử lý
	diction là bộ từ điển 

	Hàm trả về ma trận thưa csr
	'''
	row = []
	col = []
	val = []
	for i in range(len(data)):
		# tách token mỗi từ trong mỗi hàng
		line = data[i].split()
		# các từ đã thêm
		proceeded_word = set()
		for word in line:
			if word not in proceeded_word:
				try:
					dict_index = diction.index(word)
				except ValueError:
					continue
				row.append(i)
				col.append(dict_index)
				val.append(line.count(word))
				proceeded_word.update(word)

	# khởi tạo và trả về ma trận csr
	row = np.array(row)
	col = np.array(col)
	val = np.array(val)
	tf = csr_matrix((val, (row, col)), shape = (len(data), len(diction)), dtype = dtype)
	return tf

def Compute_TF(data, diction, dtype = float):
	'''
	data là dữ liệu đã qua xử lý
	diction là bộ từ điển 

	Hàm trả về ma trận thưa sau ma trận thưa tf
	'''
	# khởi tạo ma trận tf
	tf = TF(data, diction, dtype)
	# tính giá trị tf
	for i in range(tf.shape[0]):
		try:
			tf.data[tf.indptr[i]:tf.indptr[i+1]] = tf.data[tf.indptr[i]:tf.indptr[i+1]]/np.max(tf.data[tf.indptr[i]:tf.indptr[i+1]])
		except ValueError:
			pass

	return tf

def Compute_TF_IDF(data, diction):
	'''
	data là dữ liệu đã qua xử lý
	diction là bộ từ điển 

	Hàm trả về ma trận thưa sau khi nhân tf và idf
	'''
	# số từ trong từ điển
	features = len(diction)
	# số điểm dữ liệu
	number_of_datapoints = len(data)
	# tính giá trị tf
	tf = Compute_TF(data, diction)
	# chuyển sang ma trận thưa csc
	matrix_csc = tf.tocsc()
	# tính giá trị IDF
	for i in range(features):
		matrix_csc.data[matrix_csc.indptr[i]:matrix_csc.indptr[i+1]] = np.log(number_of_datapoints/(1+ np.count_nonzero(matrix_csc.data[matrix_csc.indptr[i]:matrix_csc.indptr[i+1]])))
	# nhân 2 ma trận và trả về kết quả 
	return tf.multiply(matrix_csc.tocsr())
def main():
	# đọc file
	path = 'D:/Document/Machine_Learning/CS3/Sarcasm_Headlines_Dataset.json'
	Data = pd.read_json(path, lines = True)

	# loại bỏ stop word
	processed_data = eliminate_stopwords(Data)
	# khởi từ điển
	diction = create_list_diction(processed_data)
	# tính tfidf
	tf_idf = Compute_TF_IDF(processed_data, diction)
	

if __name__ == '__main__':
	main()



