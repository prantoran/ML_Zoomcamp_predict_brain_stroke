#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


# data preparation

df = pd.read_csv('data/brain_stroke.csv')

df.columns = df.columns.str.lower().str.replace(' ',
 '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.stroke
y_test = df_test.stroke

del df_full_train["stroke"]
del df_test["stroke"]

columns_numerical = ['age', 'avg_glucose_level', 'bmi']

columns_categorical = [
    'gender',
    'ever_married',
    'work_type',
    'residence_type',
    'smoking_status'
]

columns_binary = ['hypertension', 'heart_disease']

# training 

dict_full_train = df_full_train.to_dict(orient="records")
dict_test = df_test.to_dict(orient="records")

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dict_full_train)
X_test = dv.transform(dict_test)


rf = RandomForestClassifier(n_estimators=30, random_state=31)
rf.fit(X_full_train, y_full_train)

y_pred = rf.predict_proba(X_test)[:, 1]
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'test rmse:{rmse:.5f}')

with open ('brain_stroke_model.bin', 'wb') as f:
    pickle.dump((dv,rf), f)

print("Model training completed")