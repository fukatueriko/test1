# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 17:16:57 2014

@author: 
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer

# データを読み込む
df=pd.read_csv("C:\\Users\\fukazu\\Documents\\IPython Notebooks\\deepanAlytics\\train.csv",header=None,nrows=10000)

# データクリーニング
# NaNが一つでも入っているrowを除く
df = df[pd.notnull(df).all(1)]

# 説明変数ｘ、目的変数ｙに分ける
x = df.loc[:, 2:]
y = df[1]

# カテゴリカル変数は文字列に直す
x.loc[:, 4:9] = x.loc[:, 4:9].astype(str)

# カテゴリカル変数を数量化
x = DictVectorizer(sparse=False).fit_transform(x.to_dict('records'))

# SVCで学習
clf = SVC()
clf.fit(x, y)

# 学習スコア
#print clf.score(x, y)
print clf.predict(x[0:0])