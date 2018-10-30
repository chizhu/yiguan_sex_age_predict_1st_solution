
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datetime import datetime,timedelta  
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import gc
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


# In[2]:

print ('8.device_quchong_start_app_w2c.py')
path='input/'
data=pd.DataFrame()
#sex_age=pd.read_excel('./data/性别年龄对照表.xlsx')


# In[3]:


deviceid_packages=pd.read_csv(path+'deviceid_packages.tsv',sep='\t',names=['device_id','apps'])
deviceid_test=pd.read_csv(path+'deviceid_test.tsv',sep='\t',names=['device_id'])
deviceid_train=pd.read_csv(path+'deviceid_train.tsv',sep='\t',names=['device_id','sex','age'])
deviceid_brand = pd.read_csv(path+'deviceid_brand.tsv',sep='\t', names=['device_id','device_brand', 'device_type'])
deviceid_package_start_close = pd.read_csv(path+'deviceid_package_start_close.tsv',sep='\t', names=['device_id','app_id','start_time','close_time'])
package_label = pd.read_csv(path+'package_label.tsv',sep='\t',names=['app_id','app_parent_type', 'app_child_type'])


deviceid_brand['device_brand'] = deviceid_brand.device_brand.apply(lambda x : str(x).split(' ')[0])

df_temp = deviceid_brand.groupby('device_brand')['device_id'].count().reset_index().rename(columns={'device_id':'brand_counts'})
one_time_brand = df_temp[df_temp.brand_counts == 1].device_brand.values
deviceid_brand['device_brand'] = deviceid_brand.device_brand.apply(lambda x : 'other' if x in one_time_brand else x)

df_temp = deviceid_brand.groupby('device_brand')['device_id'].count().reset_index().rename(columns={'device_id':'brand_counts'})
one_time_brand = df_temp[df_temp.brand_counts == 2].device_brand.values
deviceid_brand['device_brand'] = deviceid_brand.device_brand.apply(lambda x : 'other_2' if x in one_time_brand else x)

df_temp = deviceid_brand.groupby('device_brand')['device_id'].count().reset_index().rename(columns={'device_id':'brand_counts'})
one_time_brand = df_temp[df_temp.brand_counts == 3].device_brand.values
deviceid_brand['device_brand'] = deviceid_brand.device_brand.apply(lambda x : 'other_3' if x in one_time_brand else x)


#转换成对应的数字
lbl = LabelEncoder()
lbl.fit(list(deviceid_brand.device_brand.values))
deviceid_brand['device_brand'] = lbl.transform(list(deviceid_brand.device_brand.values))

lbl = LabelEncoder()
lbl.fit(list(deviceid_brand.device_type.values))
deviceid_brand['device_type'] = lbl.transform(list(deviceid_brand.device_type.values))

#转换成对应的数字
lbl = LabelEncoder()
lbl.fit(list(package_label.app_parent_type.values))
package_label['app_parent_type'] = lbl.transform(list(package_label.app_parent_type.values))

lbl = LabelEncoder()
lbl.fit(list(package_label.app_child_type.values))
package_label['app_child_type'] = lbl.transform(list(package_label.app_child_type.values))


# In[4]:


import time

# 输入毫秒级的时间，转出正常格式的时间
def timeStamp(timeNum):
    timeStamp = float(timeNum/1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

#解析出具体的时间
deviceid_package_start_close['start_date'] = pd.to_datetime(deviceid_package_start_close.start_time.apply(timeStamp))
deviceid_package_start_close['end_date'] = pd.to_datetime(deviceid_package_start_close.close_time.apply(timeStamp))
deviceid_package_start_close['start_hour'] = deviceid_package_start_close.start_date.dt.hour
deviceid_package_start_close['end_hour'] = deviceid_package_start_close.end_date.dt.hour
deviceid_package_start_close['time_gap'] = (deviceid_package_start_close['end_date'] - deviceid_package_start_close['start_date']).astype('timedelta64[s]')

deviceid_package_start_close = deviceid_package_start_close.merge(package_label, on='app_id', how='left')
deviceid_package_start_close.app_parent_type.fillna(-1, inplace=True)
deviceid_package_start_close.app_child_type.fillna(-1, inplace=True)
deviceid_package_start_close['start_year'] = deviceid_package_start_close.start_date.dt.year
deviceid_package_start_close['end_year'] = deviceid_package_start_close.end_date.dt.year
deviceid_package_start_close['year_gap'] = deviceid_package_start_close['end_year'] - deviceid_package_start_close['start_year']


# In[9]:


df_temp = deviceid_package_start_close.groupby(['device_id', 'app_id'])['start_hour'].mean().reset_index()
df_temp


# In[10]:


df_sorted = df_temp.sort_values(by='start_hour')


# In[13]:


df_device_start_app_list = df_sorted.groupby('device_id').apply(lambda x : list(x.app_id)).reset_index().rename(columns = {0 : 'app_list'})


# In[17]:


app_list = list(df_device_start_app_list.app_list.values)


# In[35]:


model = Word2Vec(app_list, size=10, window=4, min_count=2, workers=4)
model.save("word2vec.model")


# In[37]:


vocab = list(model.wv.vocab.keys())

w2c_arr = []

for v in vocab :
    w2c_arr.append(list(model.wv[v]))


# In[38]:


df_w2c_start = pd.DataFrame()
df_w2c_start['app_id'] = vocab
df_w2c_start = pd.concat([df_w2c_start, pd.DataFrame(w2c_arr)], axis=1)
df_w2c_start.columns = ['app_id'] + ['w2c_start_app_' + str(i) for i in range(10)]


# In[47]:


df_sorted = df_sorted.merge(df_w2c_start, on='app_id', how='left')
df_sorted


# In[48]:


w2c_nums = 10
agg = {}
for l in ['w2c_start_app_' + str(i) for i in range(w2c_nums)] :
    agg[l] = ['mean', 'std', 'max', 'min']


# In[50]:


df_agg = df_sorted.groupby('device_id').agg(agg)
df_agg.columns = pd.Index(['device_quchong' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
df_agg = df_agg.reset_index()


# In[52]:


df_agg.to_csv('device_quchong_start_app_w2c.csv', index=None)

