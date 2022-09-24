import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

log_reg = LogisticRegression(lr=0.02)
log_reg.fit(X_train,y_train)
predictions = log_reg.predict(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

acc = accuracy(predictions, y_test)
print(f"accuracy {str(acc)}")