import re
import glob
import random
import itertools
from sklearn.feature_selection import RFECV
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDClassifier
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
files = glob.glob("November\submission_files\*.csv")
df = pd.read_csv("November\labels_train.csv")
suf = 0
for file_name in files:
    file = file_name[-16:]
    df_pred = pd.read_csv(f"November\submission_files\{file}")
    df = df.join(df_pred["pred"], on="id", rsuffix=suf, how="outer")
    
    suf += 1
    print(suf)
print(df)
df.to_csv("November/Train.csv")
