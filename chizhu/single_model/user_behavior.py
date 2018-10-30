
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# %matplotlib inline
from config import path
#add
import gc

packtime = pd.read_table(path+'deviceid_package_start_close.tsv',
                         names=['device_id', 'app', 'start', 'close'], low_memory=True)
# packtime.head()
packtime['peroid'] = (packtime['close'] - packtime['start'])/1000
packtime['start'] = pd.to_datetime(packtime['start'], unit='ms')
#packtime['closetime'] = pd.to_datetime(packtime['close'], unit='ms')
del packtime['close']
gc.collect()

#packtime['day'] = packtime['start'].dt.day
#packtime['month'] = packtime['start'].dt.month
packtime['hour'] = packtime['start'].dt.hour
packtime['date'] = packtime['start'].dt.date
packtime['dayofweek'] = packtime['start'].dt.dayofweek
#packtime['hour'] = pd.cut(packtime['hour'], bins=4).cat.codes

#平均每天使用设备时间
dtime = packtime.groupby(['device_id', 'date'])['peroid'].agg('sum')
#不同时间段占比
qtime = packtime.groupby(['device_id', 'hour'])['peroid'].agg('sum')
wtime = packtime.groupby(['device_id', 'dayofweek'])['peroid'].agg('sum')
atime = packtime.groupby(['device_id', 'app'])['peroid'].agg('sum')


dapp = packtime[['device_id', 'date', 'app']].drop_duplicates().groupby(
    ['device_id', 'date'])['app'].agg(' '.join)
dapp = dapp.reset_index()
dapp['app_len'] = dapp['app'].apply(lambda x: x.split(' ')).apply(len)
dapp_stat = dapp.groupby('device_id')['app_len'].agg(
    {'std': 'std', 'mean': 'mean', 'max': 'max'})
dapp_stat = dapp_stat.reset_index()
dapp_stat.columns = ['device_id', 'app_len_std', 'app_len_mean', 'app_len_max']
# dapp_stat.head()

dtime = dtime.reset_index()
dtime_stat = dtime.groupby(['device_id'])['peroid'].agg(
    {'sum': 'sum', 'mean': 'mean', 'std': 'std', 'max': 'max'}).reset_index()
dtime_stat.columns = ['device_id', 'date_sum',
                      'date_mean', 'date_std', 'date_max']
# dtime_stat.head()

qtime = qtime.reset_index()
ftime = qtime.pivot(index='device_id', columns='hour',
                    values='peroid').fillna(0)
ftime.columns = ['h%s' % i for i in range(24)]
ftime.reset_index(inplace=True)
# ftime.head()

wtime = wtime.reset_index()
weektime = wtime.pivot(
    index='device_id', columns='dayofweek', values='peroid').fillna(0)
weektime.columns = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
weektime.reset_index(inplace=True)
# weektime.head()

atime = atime.reset_index()
app = atime.groupby(['device_id'])['peroid'].idxmax()

#dapp_stat.shape, dtime_stat.shape, ftime.shape, weektime.shape, app.shape

user = pd.merge(dapp_stat, dtime_stat, on='device_id', how='left')
user = pd.merge(user, ftime, on='device_id', how='left')
user = pd.merge(user, weektime, on='device_id', how='left')
user = pd.merge(user, atime.iloc[app], on='device_id', how='left')

app_cat = pd.read_table(path+'package_label.tsv',
                        names=['app', 'category', 'app_name'])

cat_enc = pd.DataFrame(app_cat['category'].value_counts())
cat_enc['idx'] = range(45)

app_cat['cat_enc'] = app_cat['category'].map(cat_enc['idx'])
app_cat.set_index(['app'], inplace=True)

atime['app_cat_enc'] = atime['app'].map(app_cat['cat_enc']).fillna(45)

cat_num = atime.groupby(['device_id', 'app_cat_enc'])[
    'app'].agg('count').reset_index()
cat_time = atime.groupby(['device_id', 'app_cat_enc'])[
    'peroid'].agg('sum').reset_index()

app_cat_num = cat_num.pivot(
    index='device_id', columns='app_cat_enc', values='app').fillna(0)
app_cat_num.columns = ['cat%s' % i for i in range(46)]
app_cat_time = cat_time.pivot(
    index='device_id', columns='app_cat_enc', values='peroid').fillna(0)
app_cat_time.columns = ['time%s' % i for i in range(46)]

user = pd.merge(user, app_cat_num, on='device_id', how='left')
user = pd.merge(user, app_cat_time, on='device_id', how='left')
user.to_csv('data/user_behavior.csv', index=False)


