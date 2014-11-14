# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:50:31 2014

@author: 深津恵梨子
"""

import pandas as pd
import MySQLdb

connector = MySQLdb.connect(host="localhost", db="test_db", user=u"root", passwd="Eriko0022")
cursor = connector.cursor()    
cursor.execute("set names utf8")
df=pd.read_sql("select * from customer",connector)

print df
