# Sarcasm Detection Structure Folder
-   PreProcessed_Data: Preprocessing data with boolean, tf, idf, tf+idf model (with/without Stopword)
-   final: Reimplement Logistic Regression model.
-   CASE_STUDY_3_REPORT_SARCASM_HEADLINES_DATASETS-17520324_17520828_17521244.rar: Report (in Vietnamese)
-   Evaluate.py: Evaluation model.
-   Predicting_By_Function.py: Predicting using sk-learn (KNN + Decision Tree + Naive Bayes + Logistic Regression).
-   Sarcasm_Headlines_Dataset.json: Raw data
-   label.npy: Encoding Label
-   stopwords.txt: Stopwords.
# Define problem
- Task: Detecting a title is sarcasm or not.
- Input: Title (Dataset includes ~26709 titles from The Onion and The Huff Post).
- Output: Sarcasm or not (binary classification).
- Evaluation: Precision, Recall and F1 score.
# Define method
- B0: Clean data (removing stopword)
- B1: Preprocessing data using boolean/tf/idf/tf+idf model.
- B2: Building model.
- B3: Evaluation.
# Result
- Evaluation using different preprocessing data method:
![alt text](https://github.com/HoSyTuyen/Kaggle-Sarcasm-Detection/blob/master/Result_on_preprocessing.png)
- Evaluation using different model classification on best preprocessing data method:
![alt text](https://github.com/HoSyTuyen/Kaggle-Sarcasm-Detection/blob/master/Final_result.png)
