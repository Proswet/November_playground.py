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
from sklearn.gaussian_process.kernels import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

df = pd.read_csv("November/Train_mean.csv")
columns = df.columns

Id = df[columns[0]][df.isnull().any(1)]
X = df[columns[2]]
y = df[columns[1]]

X_train_full = df.dropna()["mean"].values.reshape(-1, 1)
y_train_full = df.dropna()[columns[1]].values
X_test_full = df[df.isnull().any(1)]["mean"].values.reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, random_state=1, train_size=0.02)

kernel = RationalQuadratic(alpha=2.16, length_scale=0.149)
model = GaussianProcessClassifier(kernel=kernel)
model.fit(X_train_full, y_train_full)
pred = model.predict_proba(X_test_full)
df_pred = pd.DataFrame(data=pred, columns=[0, 1])
df_pred["pred"] = 1 - df_pred[0].where(df_pred[0] > 0.5)
df_pred["pred"] = df_pred["pred"].fillna(df_pred[1])


pred = df_pred["pred"].values
#print(mean_squared_error(y_test, pred)**0.5)
print(model.kernel_)

df_res = pd.DataFrame()
df_res["Id"] = Id
df_res["pred"] = pred
df_res.set_index('Id', inplace=True)
df_res.to_csv("November/predict.csv")