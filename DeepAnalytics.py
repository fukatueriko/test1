# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 17:16:57 2014

@author: 
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
    
# SVCで最初の5000個を学習
for p in ['l1', 'l2']:
    for d in [True, False]:
        if p == 'l1' and d == True:
            continue
        for c in [1,10,100,1000,10000]:
            clf = LogisticRegression(penalty=p, dual=d, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
            clf.fit(x[:5000], y[:5000])
            print (p,d,c)
            # 5000番目以降に対する学習スコア
            print clf.score(x[5000:], y[5000:])
