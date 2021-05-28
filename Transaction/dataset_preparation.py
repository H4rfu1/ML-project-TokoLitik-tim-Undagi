#!/usr/bin/env python
# coding: utf-8

import os, glob
import json
import requests
import sys
from os import path

URL = str(sys.argv[1])
token = str(sys.argv[2])
my_headers = {'auth-token' : '{}'.format(token)}

response = requests.get(URL, headers=my_headers)

def getData():    
    data = []
    transaction = response.json()["transaction"]
    product = response.json()["product"]

    for i in transaction:
        temp = {"date": i["time"],
                "daily_sales": i["amount"]}
        data.append(temp)


    isDirectoryExist = os.path.isdir("dataset")

    if(isDirectoryExist):
        dir = 'D:/TokoLitik/dataset'
        filelist = glob.glob(os.path.join(dir, "*"))
        for f in filelist:
            os.remove(f)
        os.rmdir(dir)

    os.mkdir("dataset")
    jsonString = json.dumps(data)
    jsonFile = open("dataset/data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(product)
    jsonFile = open("dataset/product.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(transaction)
    jsonFile = open("dataset/transaction.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

if(response.status_code == 200):
    print(response.status_code)    
    getData()

else:
    print(response.status_code)
