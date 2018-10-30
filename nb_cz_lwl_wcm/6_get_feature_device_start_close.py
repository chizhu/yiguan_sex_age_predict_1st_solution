# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing

train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', header=None)
test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', header=None)

data_all = pd.concat([train, test], axis=0)
data_all = data_all.rename({0:'id'}, axis=1)
del data_all[1],data_all[2]

start_close_time = pd.read_csv('Demo/deviceid_package_start_close.tsv', sep='\t', header=None)
start_close_time = start_close_time.rename({0:'id', 1:'app_name', 2:'start_time', 3:'close_time'}, axis=1)


start_close_time['diff_time'] = (start_close_time['close_time'] - start_close_time['start_time'])/1000

print('开始转换时间')
import time
start_close_time['close_time'] = start_close_time['close_time'].apply(lambda row: int(time.localtime(row/1000).tm_hour))
start_close_time['start_time'] = start_close_time['start_time'].apply(lambda row: int(time.localtime(row/1000).tm_hour))

# 一个表里面的总次数
print('一个表的总次数')
feature = pd.DataFrame()
feature['start_close_count'] = pd.merge(data_all, start_close_time.groupby('id').size().reset_index(), on='id', how='left')[0]

# 0 - 5 点的使用次数
temp = start_close_time[(start_close_time['close_time'] >=0)&(start_close_time['close_time'] <=5)]
temp = temp.groupby('id').size().reset_index()
feature['zero_five_count'] = pd.merge(data_all, temp, on='id', how='left').fillna(0)[0]

# 玩的时间最长的app的名字编码
def get_max_label(row):
    row_name = list(row['app_name'])
    row_diff_time = list(row['diff_time'])
    return row_name[np.argmax(row_diff_time)]

start_close_max_name = start_close_time.groupby('id').apply(lambda row:get_max_label(row)).reset_index()
label_encoder = preprocessing.LabelEncoder()
feature['start_close_max_name'] = label_encoder.fit_transform(pd.merge(data_all, start_close_max_name, on='id', how='left').fillna(0)[0])

feature.to_csv('feature/feature_start_close.csv', index=False)