import numpy as np

# Tìm giá trị TP
def find_TP(y_pred, y_test):
    count = 0
    Positive = np.multiply(y_pred, y_test)
    return np.count_nonzero(Positive)

def Recall(TP, P):
    return 1.0*TP/P

def Precision(TP, FP):
    return 1.0*TP/(TP+FP) 

# tính tổng số điểm dữ liệu dự đoán nhãn là 1 nhưng sai
def find_FP(y_pred, y_test):
    count = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == 1:
            if y_test[i] == 0:
                count += 1
    return count

# tính tống số điểm dữ liệu có nhãn là 1 trong tập test
def find_P(y_test):
    return np.count_nonzero(y_test)

# Tính F1_score
def F1_score(Precision, Recall):
    return (Precision*Recall)/(Precision+Recall)*2

# Tính Accurancy
def Accuracy(y_pred, y_test): 
    count = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == y_test[i]:
            count += 1
    return count/y_pred.shape[0]