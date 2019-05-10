import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import string
import re

import scipy.sparse


# Đọc file
def ReadData(path):
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

# Chuyển hóa text sang features
def Vectorization(Data):
	# Khởi tạo model tf-idf
	text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=28900)

	# Chuyển hóa từ text sang không gian vector dùng mô hình tf-idf
	title = text_transformer.fit_transform(Data['headline'])

	return title

def main():
	path = "C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/Sarcasm_Headlines_Dataset.json"
	cleaned_df = ReadData(path)

	_stopword = stopwords.words('english') + list(string.punctuation)

	Data = PreProcessing_Data(cleaned_df, _stopword)

	title = Vectorization(Data)

	scipy.sparse.save_npz('C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/tfidf.npz', title)

	np.save('C:/Users/Admin/Downloads/NLTK_is_Sarcastic-master/label.npy', np.array(Data['is_sarcastic']))

if __name__ == '__main__':
	main()