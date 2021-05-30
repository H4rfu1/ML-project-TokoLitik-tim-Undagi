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
data = []
transaction = response.json()["transaction"]
product = response.json()["product"]

def getData(dir):
    for i in transaction:
        temp = {"date": i["time"],
                "daily_sales": i["amount"]}
        data.append(temp)

    jsonString = json.dumps(data)
    jsonFile = open("{}\\data.json".format(dir), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(product)
    jsonFile = open("{}\\product.json".format(dir), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def saveData():    
    dataset_ROOT = os.getcwd()+"\\result"
    dataset_DIR = "{}\\productId_{}".format(dataset_ROOT, product["id"])

    isRootExist = os.path.isdir(dataset_ROOT)
    isDirectoryExist = os.path.isdir(dataset_DIR)

    if(isRootExist):
        if(isDirectoryExist):
            filelist = glob.glob(os.path.join(dataset_DIR, "*"))
            for f in filelist:
                os.remove(f)
            getData(dataset_DIR)
        else:
            os.mkdir(dataset_DIR)
            getData(dataset_DIR)
    else:
        os.mkdir(dataset_ROOT)
        os.mkdir(dataset_DIR)
        getData(dataset_DIR)
  

if(response.status_code == 200):
    print(response.status_code)    
    saveData()

else:
    print(response.status_code)
