#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime,timedelta  
import matplotlib.pyplot as plt
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gc
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Categorical, Real, Log10
from skopt.utils import use_named_args
from skopt import gp_minimize
from gensim.models import Word2Vec, FastText
import gensim 
import re


# 获取app开关表中使用总时间Top100的app使用时长统计特征
def get_top100_statis_feat(start_close):
	start_close['period'] = (start_close['close'] - start_close['start'])/1000
	start_close['start'] = pd.to_datetime(start_close['start'], unit='ms')
	app_use_time = start_close.groupby(['app'])['period'].agg('sum').reset_index()
	app_use_top100 = app_use_time.sort_values(by='period', ascending=False)[:100]['app']
	device_app_use_time = start_close.groupby(['device_id', 'app'])['period'].agg('sum').reset_index()
	use_time_top100_statis = device_app_use_time.set_index('app').loc[list(app_use_top100)].reset_index()
	top100_statis = use_time_top100_statis.pivot(index='device_id', columns='app', values='period').reset_index()
	top100_statis = top100_statis.fillna(0)
	top100_statis.columns = ['device_id'] + ['top100_statis_' + str(i) for i in range(0, 100)]
	print('top100_statis_feat done')
	return top100_statis

# 获得手机品牌特征
def get_brand_feat(brand):
	# 手机品牌预处理
	brand['vendor'] = brand['vendor'].astype(str).apply(lambda x : x.split(' ')[0].upper())
	brand['ph_ver'] = brand['vendor'] + '_' + brand['version']
	ph_ver = brand['ph_ver'].value_counts()
	ph_ver_cnt = pd.DataFrame(ph_ver).reset_index()
	ph_ver_cnt.columns = ['ph_ver', 'ph_ver_cnt']
	brand = pd.merge(left=brand, right=ph_ver_cnt,on='ph_ver')
	# 针对长尾分布做的一点处理
	mask = (brand.ph_ver_cnt < 100)
	brand.loc[mask, 'ph_ver'] = 'other'
	ph_ver_le = preprocessing.LabelEncoder()
	brand['ph_ver'] = ph_ver_le.fit_transform(brand['ph_ver'].astype(str))
	print('brand_feat done')
	return brand[['device_id', 'ph_ver']]

# 获取app开关表的tfidf特征，不是df格式
def get_start_close_tfidf_feat(data_all, start_close):
	# 每个app的总使用次数统计
	app_num = start_close['app'].value_counts().reset_index()
	app_num.columns = ['app', 'app_num']
	start_close = pd.merge(left=start_close, right=app_num, on='app')
	# 同样的，针对长尾分布做些处理（尝试过不做处理，或换其他阈值，这个100的阈值最高）
	start_close.loc[start_close.app_num < 100, 'app'] = 'other'
	df_app = start_close[['device_id', 'app']]
	apps = df_app.drop_duplicates().groupby(['device_id'])['app'].apply(' '.join).reset_index()
	apps['app_length'] = apps['app'].apply(lambda x:len(x.split(' ')))
	data_all = pd.merge(data_all, apps, on='device_id', how='left')
	# 获取每台设备所安装的apps的tfidf
	tfidf = CountVectorizer()
	apps['app'] = tfidf.fit_transform(apps['app'])
	# 转换
	start_close_tfidf = tfidf.transform(list(data_all['app']))
	print('start_close_tfidf_feat done')
	return start_close_tfidf

# 利用word2vec得到每台设备所安装app的embedding表示
def get_packages_w2c_feat(packages):
	packages['apps'] = packages['apps'].apply(lambda x:x.split(','))
	packages['app_length'] = packages['apps'].apply(lambda x:len(x))
	embed_size = 128
	fastmodel = Word2Vec(list(packages['apps']), size=embed_size, window=4, min_count=3, negative=2,
	                 sg=1, sample=0.002, hs=1, workers=4)  

	embedding_fast = pd.DataFrame([fastmodel[word] for word in (fastmodel.wv.vocab)])
	embedding_fast['app'] = list(fastmodel.wv.vocab)
	embedding_fast.columns= ["fdim_%s" % str(i) for i in range(embed_size)]+["app"]

	id_list = []
	for i in range(packages.shape[0]):
	    id_list += [list(packages['device_id'])[i]]*packages['app_length'].iloc[i]
	app_list = [word for item in packages['apps'] for word in item]
	app_vect = pd.DataFrame({'device_id':id_list})        
	app_vect['app'] = app_list

	app_vect = app_vect.merge(embedding_fast, on='app', how='left')
	app_vect = app_vect.drop('app', axis=1)

	seqfeature = app_vect.groupby(['device_id']).agg('mean')
	seqfeature.reset_index(inplace=True)
	print('packages_w2c_feat done')
	return seqfeature

# 用户一周七天玩手机的时长情况的统计特征
def get_week_statis_feat(start_close):
	start_close['dayofweek'] = start_close['start'].dt.dayofweek
	start_close['hour'] = start_close['start'].dt.hour
	app_use_time = start_close.groupby(['device_id', 'dayofweek'])['period'].agg('sum').reset_index()
	week_app_use = app_use_time.pivot_table(values='period', columns='dayofweek', index='device_id').reset_index()
	week_app_use = week_app_use.fillna(0)
	week_app_use.columns = ['device_id'] + ['week_day_' + str(i) for i in range(0, 7)]

	week_app_use['week_max'] = week_app_use.max(axis=1)
	week_app_use['week_min'] = week_app_use.min(axis=1)
	week_app_use['week_sum'] = week_app_use.sum(axis=1)
	week_app_use['week_std'] = week_app_use.std(axis=1)
	print('week_statis_feat done')
	return week_app_use


def get_user_behaviour_feat(start_close):
	# start_close['peroid'] = (start_close['close'] - start_close['start'])/1000
	# start_close['start'] = pd.to_datetime(start_close['start'], unit='ms')
	#start_close['closetime'] = pd.to_datetime(start_close['close'], unit='ms')
	# del start_close['close']
	# gc.collect();
	start_close['hour'] = start_close['start'].dt.hour
	start_close['date'] = start_close['start'].dt.date
	start_close['dayofweek'] = start_close['start'].dt.dayofweek
	#平均每天使用设备时间
	dtime = start_close.groupby(['device_id', 'date'])['period'].agg('sum')
	#不同时间段占比
	qtime = start_close.groupby(['device_id', 'hour'])['period'].agg('sum')
	wtime = start_close.groupby(['device_id', 'dayofweek'])['period'].agg('sum')
	atime = start_close.groupby(['device_id', 'app'])['period'].agg('sum')
	dapp = start_close[['device_id', 'date', 'app']].drop_duplicates().groupby(['device_id', 'date'])['app'].agg(' '.join)
	dapp = dapp.reset_index()
	dapp['app_len'] = dapp['app'].apply(lambda x:x.split(' ')).apply(len)
	dapp_stat = dapp.groupby('device_id')['app_len'].agg({'std':'std', 'mean':'mean', 'max':'max'})
	dapp_stat = dapp_stat.reset_index()
	dapp_stat.columns = ['device_id', 'app_len_std', 'app_len_mean', 'app_len_max']
	dtime = dtime.reset_index()
	dtime_stat = dtime.groupby(['device_id'])['period'].agg({'sum':'sum', 'mean':'mean', 'std':'std', 'max':'max'}).reset_index()
	dtime_stat.columns = ['device_id', 'date_sum', 'date_mean', 'date_std', 'date_max']
	qtime = qtime.reset_index()
	ftime = qtime.pivot(index='device_id', columns='hour', values='period').fillna(0)
	ftime.columns = ['h%s'%i for i in range(24)]
	ftime.reset_index(inplace=True)
	wtime = wtime.reset_index()
	weektime = wtime.pivot(index='device_id', columns='dayofweek', values='period').fillna(0)
	weektime.columns = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
	weektime.reset_index(inplace=True)
	atime = atime.reset_index()
	app = atime.groupby(['device_id'])['period'].idxmax()
	user = pd.merge(dapp_stat, dtime_stat, on='device_id', how='left')
	user = pd.merge(user, ftime, on='device_id', how='left')
	user = pd.merge(user, weektime, on='device_id', how='left')
	user = pd.merge(user, atime.iloc[app], on='device_id', how='left')
	app_cat = pd.read_table('Demo/package_label.tsv', names=['app', 'category', 'app_name'])

	cat_enc = pd.DataFrame(app_cat['category'].value_counts())
	cat_enc['idx'] = range(45)

	app_cat['cat_enc'] = app_cat['category'].map(cat_enc['idx'])
	app_cat.set_index(['app'], inplace=True)
	atime['app_cat_enc'] = atime['app'].map(app_cat['cat_enc']).fillna(45)

	cat_num = atime.groupby(['device_id', 'app_cat_enc'])['app'].agg('count').reset_index()
	cat_time = atime.groupby(['device_id', 'app_cat_enc'])['period'].agg('sum').reset_index()
	app_cat_num = cat_num.pivot(index='device_id', columns='app_cat_enc', values='app').fillna(0)
	app_cat_num.columns = ['cat%s'%i for i in range(46)]
	app_cat_time = cat_time.pivot(index='device_id', columns='app_cat_enc', values='period').fillna(0)
	app_cat_time.columns = ['time%s'%i for i in range(46)]
	user = pd.merge(user, app_cat_num, on='device_id', how='left')
	user = pd.merge(user, app_cat_time, on='device_id', how='left')
	del user['app']
	print('user_behaviour_feat done')
	return user






if __name__ == '__main__':
	test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', names=['device_id'])
	train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', names=['device_id', 'sex', 'age'])
	brand = pd.read_table('Demo/deviceid_brand.tsv', names=['device_id', 'vendor', 'version'])
	start_close = pd.read_table('Demo/deviceid_package_start_close.tsv', 
	                         names=['device_id', 'app', 'start', 'close'])
	packages = pd.read_csv('Demo/deviceid_packages.tsv', sep='\t', names=['device_id', 'apps'])
	data_all = pd.concat([train, test], axis=0, ignore_index=True)
	print('data done')

	top100_statis_feat = get_top100_statis_feat(start_close)
	brand_feat = get_brand_feat(brand)
	# start_close_tfidf_feat = get_start_close_tfidf_feat(data_all, start_close)
	packages_w2c_feat = get_packages_w2c_feat(packages)
	week_statis_feat = get_week_statis_feat(start_close)
	user_behaviour_feat = get_user_behaviour_feat(start_close)
	print('feats done')

	data_all = pd.merge(data_all, top100_statis_feat, on='device_id', how='left')
	data_all = pd.merge(data_all, brand_feat, on='device_id', how='left')
	data_all = pd.merge(data_all, packages_w2c_feat, on='device_id', how='left')
	data_all = pd.merge(data_all, week_statis_feat, on='device_id', how='left')
	data_all = pd.merge(data_all, user_behaviour_feat, on='device_id', how='left')
	print('merge done')
	# 删掉标签
	del data_all['age'], data_all['sex']
	data_all.to_csv('feature/feat_lwl.csv', index=None)



