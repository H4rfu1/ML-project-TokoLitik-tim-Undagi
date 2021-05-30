#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import json
import dateutil
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 1 ----- IMPORT DATASET ----- #
productId = str(sys.argv[1])
dataset_DIR = os.getcwd()+"\\result"+"\\"+'productId_{}'.format(productId)

df = pd.read_json('{}\\data.json'.format(dataset_DIR))
df["date"] = df["date"].dt.date

productFile = open("{}\\product.json".format(dataset_DIR))
product = json.load(productFile)

# 2 ----- FEATURES PREPARATION ----- #

#change date field into datetime format
df["date"] = pd.to_datetime(df["date"])

# change every data in date field as the first date of each month for plotting the monthly sales
df_temp = df.copy()
df_temp["date"] = df_temp["date"].dt.year.astype("str") + '-' + df["date"].dt.month.astype("str") + '-01'
df_temp['date'] = pd.to_datetime(df_temp['date'])

# change daily sales data into monthly sales
df_monthly = df_temp.copy()
df_monthly = df_monthly.groupby('date').daily_sales.sum().reset_index()
df_monthly.rename(columns={'daily_sales':'monthly_sales'}, inplace=True)

# -- Save Monthly Sales Data based on raw dataset -- #
# dataset_dir = "dataset"
# dataset_path = "dataset/monthly_data.json"

# df_monthly_json = df_monthly.to_json(orient="records")
# monthly_parsed = json.loads(df_monthly_json)

# def saveMonthlyData():
#     jsonString = json.dumps(monthly_parsed)
#     jsonFile = open(dataset_path, "w")
#     jsonFile.write(jsonString)
#     jsonFile.close()

# if(os.path.isdir(dataset_dir)):
#     if(os.path.exists(dataset_path)):
#         os.remove(dataset_path)
#     saveMonthlyData()
# else:
#     os.mkdir(image_dir)
#     saveMonthlyData()

# - Plot Monthly Sales Data

# plt.rc("font", size=12)
# fig, ax = plt.subplots(figsize=(10,5))

# # specify how our line should look like
# ax.plot(df_monthly['date'], df_monthly['monthly_sales'], label="monthly_sales")

# # same as above
# ax.set_xlabel("date")
# ax.set_ylabel("monthly_sales")
# ax.set_title("Monthly Sales of {}".format(product["productName"]))
# ax.grid(True)
# ax.legend(loc="upper left")

# save monthly sales graph as image 

# image_dir = "images"
# image_path = "images/monthly_sales.png"

# if(os.path.isdir(image_dir)):
#     if(os.path.exists(image_path)):
#         os.remove(image_path)
#     plt.savefig("images/monthly_sales.png")
# else:
#     os.mkdir(image_dir)
#     plt.savefig("images/monthly_sales.png")



# 3 ----- DATA PREPARATION ----- #

def data_preparation(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)


data = df_monthly["monthly_sales"].values

# reshape dataset for normalization
dataset = data.reshape(-1,1)


# 4 ----- DATA TRANSFORMATION ----- #

x_transformer = MinMaxScaler()
x_transformer.fit(dataset)

# difference transform
x_data_transformed = x_transformer.transform(dataset)
x_data_transformed = list(x_data_transformed.reshape(dataset.shape[0],))

# choose a number of time steps
n_steps_in, n_steps_out = 6, 6
n_features = 1

# split into samples
X, y = data_preparation(x_data_transformed, n_steps_in, n_steps_out)
X = X.reshape((X.shape[0], X.shape[1], n_features))


# 5 ----- LSTM MODEL -----


from tensorflow.keras.layers import *
model = tf.keras.models.Sequential()
model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# 6 ----- FIT MODEL ----- #
model.fit(X, y, epochs=300, verbose=1)


# 7 ---- EXPORTING MODEL ----- #
# saved_model_dir = "D:/TokoLitik/model"
# saved_model_path = "D:/TokoLitik/model/mymodel.h5"

# if(os.path.isdir(saved_model_dir)):
#     if(os.path.exists(saved_model_path)):
#         os.remove(saved_model_path)
#     model.save(saved_model_path)
# else:
#     os.mkdir(saved_model_dir)
#     model.save(saved_model_path)


# 8 ---- TESTING DATA ----- #
# total_data = len(data)
# start_index = total_data - 12
# end_index = total_data - 6

# actual_data = list(data[start_index:end_index])
    
# # check test input actual data by inverted it with MinMaxScaler
# test_data = x_data_transformed[start_index:end_index]
# test_data = np.array(test_data).reshape(-1,1)
# x_inverted = x_transformer.inverse_transform(test_data)
# x_inverted = x_inverted.reshape(6,)
# output_inverted = [math.ceil(i) for i in x_inverted]

# # demonstrate testing
# pred_shape = 6

# x_input = test_data.reshape(pred_shape,)
# x_input = x_input.reshape((1, n_steps_in, n_features))
# yhat = model.predict(x_input, verbose=0)

# # invert difference
# y_hat = np.array(yhat[0]).reshape(-1,1)
# x_inverted = x_transformer.inverse_transform(yhat)
# x_inverted = x_inverted.reshape(pred_shape)
# output_inverted = [math.ceil(i) for i in x_inverted]

# get_months = df_monthly["date"][-6:].values
# create_data = {"date": get_months, "pred": output_inverted, "actual": df_monthly["monthly_sales"][-6:]}
# df_result = pd.DataFrame(create_data)


# plt.rc("font", size=12)
# fig, ax = plt.subplots(figsize=(10,5))

# # specify how our line should look like
# ax.plot(df_result['date'], df_result['actual'], label="actual")
# ax.plot(df_result['date'], df_result['pred'], label="predicted")

# # same as above
# ax.set_xlabel("date")
# ax.set_ylabel("sales")
# ax.set_title("Sales Prediction")
# ax.grid(True)
# ax.legend(loc="upper left")


# 9 ----- PREDICT NEXT 6 MONTHS ----- #

pred_shape = 6 
n_steps_in = 6
n_features = 1

pred_data = np.array(x_data_transformed[-6:])
pred_input = pred_data.reshape(pred_shape,)
pred_input = pred_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(pred_input, verbose=0)

# invert difference
y_hat = np.array(yhat[0]).reshape(-1,1)
x_inverted = x_transformer.inverse_transform(yhat)
x_inverted = x_inverted.reshape(pred_shape)
pred_output = [math.ceil(i) for i in x_inverted]

last_month = df_monthly["date"][-1:].values[0]
last_month = np.datetime64(last_month)
last_month = pd.to_datetime(last_month)
date = last_month.date()

delta = dateutil.relativedelta.relativedelta(months=7)
next_6_months = date + delta

months = np.arange(date, next_6_months, dtype='datetime64[M]')
next_6_months = []

for i in months:
    next_6_months.append(str(i)+'-01')
next_6_months = next_6_months[1:]

get_pred_months = np.array(next_6_months)
create_df_pred = {"date": get_pred_months, "monthly_sales": pred_output}
df_pred_result = pd.DataFrame(create_df_pred)
df_pred_result["date"] = pd.to_datetime(df_pred_result["date"])


plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=(10,5))

# specify how our line should look like
ax.plot(df_pred_result['date'], df_pred_result['monthly_sales'], label="sales")

# same as above
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_title("Sales Prediction (2021/01 - 2021/06)")
ax.grid(True)
ax.legend(loc="upper left")


# 10 -- SAVE PREDICTED DATA AS IMAGE -- #

storeId = product["storeId"]
#result_ROOT1 = os.getcwd()
#result_ROOT2 = "{}\\{}".format(result_ROOT1, storeId)
#result_dir = "{}\\{}".format(result_ROOT2, productId)
dataset_ROOT = os.getcwd()+"\\result"
result_dir = "{}\\productId_{}".format(dataset_ROOT, productId)

pred_image_path = "{}\\next_6_months.png".format(result_dir)

#isRoot1Exist = os.path.isdir(result_ROOT1)
#isRoot2Exist = os.path.isdir(result_ROOT2)
#isDirExist = os.path.isdir(result_dir)
isPredImgExist = os.path.exists(pred_image_path)

if(isPredImgExist):
	os.remove(pred_image_path)
	plt.savefig(pred_image_path)
else:
	plt.savefig(pred_image_path)

#if(isRoot1Exist):
#    if(isRoot2Exist):
#        if(isDirExist):
#            if(isPredImgExist):
#                os.remove(pred_image_path)
#            plt.savefig(pred_image_path)
#        else:
#            os.mkdir(result_dir)
#            plt.savefig(pred_image_path)
#    else:
#        os.mkdir(result_ROOT2)
#        os.mkdir(result_dir)
#        plt.savefig(pred_image_path)
#else:
#    os.mkdir(result_ROOT1)
#    os.mkdir(result_ROOT2)
#    os.mkdir(result_dir)
#    plt.savefig(pred_image_path)


# 11 -- MERGE DATASET WITH PREDICTED DATA -- #
mydataframe = df_monthly.append(df_pred_result, ignore_index=True)

df_result_json = mydataframe.to_json(orient="records")
result_parsed = json.loads(df_result_json)


plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=(10,5))

# specify how our line should look like
ax.plot(mydataframe['date'], mydataframe['monthly_sales'], label="sales")

# same as above
ax.set_xlabel("date")
ax.set_ylabel("sales")
ax.set_title("Sales Prediction (2021/01 - 2021/06)")
ax.grid(True)
ax.legend(loc="upper left")



# 12 ----- STOCK PREDICT ----- #

product_composition = product["composition_details"]
compositions = []

for i in product_composition:
    temp = {"name": i['composition']["compositionName"], "amount": i["amount"]}
    compositions.append(temp)


compositions_pred = []
count = 0

get_pred_months = np.array(next_6_months)

for i in get_pred_months: 
    temp = {
        "date" : get_pred_months[count],
        "monthly_expenses" : {}
    }
    
    for i in compositions:
        temp["monthly_expenses"][i["name"]] = i["amount"]*pred_output[count]
    count=count+1
    compositions_pred.append(temp)



# 13 ------ Save Json and Image Files ----- #

image_path = "{}\\start_next_6_months.png".format(result_dir)
composition_path = "{}\\compositions_pred.json".format(result_dir)

isImgPathExist = os.path.exists(image_path)
isCompoPathExist = os.path.exists(composition_path)

def savePredictJson():    
    jsonString = json.dumps(compositions_pred)
    jsonFile = open(composition_path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

if(isImgPathExist and isCompoPathExist):
    os.remove(image_path)
    os.remove(composition_path)
savePredictJson()
plt.savefig(image_path)
