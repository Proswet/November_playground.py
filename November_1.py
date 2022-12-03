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
df = pd.read_csv("November/Train.csv")
print(df)