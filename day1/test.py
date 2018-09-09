#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step2 load data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
# X2 = dataset.iloc[:,:-1].values
Y = dataset.iloc[ : , 3].values
print("X------")
print(X)
print("Y------")
print(Y)

# Step3 process lost data,
# define a transform func to process data
#imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = Imputer()
imputer = imputer.fit(X[ : , 1:3])

tempX = imputer.transform(X[ : , 1:3])
X[ : , 1:3] = tempX
print("missing value processed X------")
print(X)

# Step4 process text label
# process text label 2 num (StringIndex), 依照频次排序, 最多的值为0
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print("StringIndex X------")
print(X)

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print("onehotencode X------")
print(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("labelEncode Y------")
print(Y)

# Step5
# split train data & test data
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("X_test------")
print(X_test)
print("Y_test------")
print(Y_test)

# Step6
# 特征标准化/规范化 reference: https://blog.csdn.net/u012102306/article/details/51940147
# (featureValue - mean)/std, 将特征值减去均值，除以标准差
# {方差公式s^2=1/n[(x1-m)^2+(x2-m)^2+...+(xn-m)^2], m为均值, n为特征个数, xn为第n个特征值},
# 标准差描述了样本的波动大小, 值越大波动越大
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
print("X_train feature vector------")
print(X_train)
print("StandardScaler params------")
print(sc_X.mean_)
print(sc_X.std_)
print("StandardScaler params end------")

X_test = sc_X.transform(X_test)
print("X_test feature vector------")
print(X_test)