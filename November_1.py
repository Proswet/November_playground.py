import re
import glob
import random
import itertools
from sklearn.feature_selection import RFECV
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDClassifier
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import WhiteKernel, Kernel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv("November/Train.csv")
columns = df.columns
Id = df[columns[1]][df.isnull().any(1)]
X = df[columns[3:]]
y = df[columns[2]]

X["mean"] = X.mean(axis=1)
df["mean"] = X["mean"]

df_mean = df[[columns[2], "mean"]]
df_mean.to_csv("November/Train_mean.csv")
print("PL")

X_train_full = df.dropna()["mean"].values.reshape(-1, 1)
y_train_full = df.dropna()[columns[2]].values
X_test_full = X[df.isnull().any(1)]["mean"].values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, random_state=1)

model = GaussianProcessClassifier(kernel=Kernel)
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)
print(mean_squared_error(y_test, pred)**0.5)
print(model.kernel_)