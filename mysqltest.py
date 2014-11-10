# -*- coding: cp932 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import MySQLdb

if __name__ == "__main__":

    connector = MySQLdb.connect(host="localhost", db="test_db", user=u"root", passwd="XXXXXXX")
    cursor = connector.cursor()
    
    cursor.execute("set names utf8")
    cursor.execute(u"select * from customer")
    
    result = cursor.fetchall()
    for data in result:
        print data[0]
        print data[1]
        print data[2]
    
    cursor.close()
    connector.close()
