
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



# In[2]:

print ('7.start_close_sex_prob_oof.py')
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

deviceid_train=pd.concat([deviceid_train,deviceid_test])


# In[6]:


deviceid_packages['apps']=deviceid_packages['apps'].apply(lambda x:x.split(','))
deviceid_packages['app_lenghth']=deviceid_packages['apps'].apply(lambda x:len(x))


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


agg_func = {
    'start_hour' : ['min', 'max', 'mean', 'std', 'count'], 
    'end_hour' : ['min', 'max', 'mean', 'std'], 
    'time_gap' : ['min', 'max', 'mean', 'std']
}
df_agg = deviceid_package_start_close.groupby(['device_id', 'app_id']).agg(agg_func)
df_agg.columns = pd.Index(['device_app_grouped' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
df_agg = df_agg.reset_index()
df_agg = df_agg.merge(package_label, on='app_id', how='left')


# In[11]:


#device在每个时段打开app的次数
df_temp = deviceid_package_start_close.groupby(['device_id', 'app_id', 'start_hour'])['start_time'].count().reset_index()
df_temp = pd.pivot_table(df_temp, index=['device_id', 'app_id'], columns='start_hour', values='start_time').reset_index()
df_temp.columns = ['device_id', 'app_id'] + ['device_app_start_counts'+str(i) + '_hour' for i in range(0,24)]
df_temp.fillna(0, inplace=True)


# In[13]:


df_agg = df_agg.merge(df_temp, on=['device_id', 'app_id'], how='left')


# In[15]:


apps=deviceid_packages['apps'].apply(lambda x:' '.join(x)).tolist()
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
cntTf = vectorizer.fit_transform(apps)
tfidf=transformer.fit_transform(cntTf)
word=vectorizer.get_feature_names()
weight=tfidf.toarray()
df_weight=pd.DataFrame(weight)
feature=df_weight.columns
df_weight['sum']=0
for f in tqdm(feature):
    df_weight['sum']+=df_weight[f]
deviceid_packages['tfidf_sum']=df_weight['sum']

lda = LatentDirichletAllocation(n_topics=5,
                                learning_offset=50.,
                                random_state=666)
docres = lda.fit_transform(cntTf)

deviceid_packages = pd.concat([deviceid_packages,pd.DataFrame(docres)],axis=1)

del deviceid_packages['apps']
deviceid_packages.columns = ['device_id', 'app_lenghth', 'tfidf_sum', 'LDA_0', 'LDA_1', 'LDA_2', 'LDA_3', 'LDA_4']


# In[207]:


df_temp = df_agg.merge(deviceid_packages, on='device_id', how='left')
df_w2c_start = pd.read_csv('device_start_app_w2c.csv')
df_w2c_close = pd.read_csv('device_close_app_w2c.csv')
df_w2c_all = pd.read_csv('device_all_app_w2c.csv')
df_sex_prob_oof = pd.read_csv('device_sex_prob_oof.csv')
df_age_prob_oof = pd.read_csv('device_age_prob_oof.csv')


df_temp = df_temp.merge(df_w2c_start, on='device_id', how='left')
df_temp = df_temp.merge(df_w2c_close, on='device_id', how='left')
df_temp = df_temp.merge(df_w2c_all, on='device_id', how='left')
df_temp = df_temp.merge(df_sex_prob_oof, on='device_id', how='left')
df_temp = df_temp.merge(df_age_prob_oof, on='device_id', how='left')


# In[224]:


agg_func = {
    'device_id' : ['count'], 
    'app_lenghth' : ['min', 'mean', 'std', 'max'], 
    'tfidf_sum' : ['min', 'mean', 'std', 'max'], 
    'LDA_1' : ['min', 'mean', 'std', 'max'], 
    'LDA_2' : ['min', 'mean', 'std', 'max'], 
    'LDA_3' : ['min', 'mean', 'std', 'max'], 
    'LDA_4' : ['min', 'mean', 'std', 'max'], 
}

for j in [i for i in df_age_prob_oof.columns.values if i != 'device_id'] :
    agg_func[j] = ['min', 'mean', 'std', 'max']

for j in [i for i in df_sex_prob_oof.columns.values if i != 'device_id'] :
    agg_func[j] = ['min', 'mean', 'std', 'max']    
    
for j in [i for i in df_w2c_all.columns.values if i != 'device_id'] :
    agg_func[j] = ['mean']   
    
for j in [i for i in df_w2c_start.columns.values if i != 'device_id'] :
    agg_func[j] = ['mean']  
    
for j in [i for i in df_w2c_close.columns.values if i != 'device_id'] :
    agg_func[j] = ['mean']      


# In[226]:


df_app_temp = df_temp.groupby('app_id').agg(agg_func)
df_app_temp.columns = pd.Index(['app_grouped' + e[0] + "_" + e[1].upper() for e in df_app_temp.columns.tolist()])
df_app_temp = df_app_temp.reset_index()
df_train = df_agg.merge(df_app_temp, on='app_id', how='left')


# In[228]:


df_train = df_train.merge(deviceid_train, on='device_id', how='left')


# In[235]:


train = df_train[df_train['sex'].notnull()]
test = df_train[df_train['sex'].isnull()]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

X = train.drop(['sex','age', 'app_id', 'device_id'],axis=1)
Y = train['sex']
Y_CAT = pd.Categorical(Y)
Y = pd.Series(Y_CAT.codes)


# In[237]:


from sklearn.model_selection import KFold, StratifiedKFold

seed = 2018
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)

sub_list = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])

cate_feat = ['device_type','device_brand']

params = {
    'boosting_type': 'gbdt',
    'learning_rate' : 0.02,
    #'max_depth':5,
    'num_leaves' : 2 ** 5,
    'metric': {'binary_logloss'},
    'objective' : 'binary',
    'random_state' : 6666,
    'bagging_freq' : 5,
    'feature_fraction' : 0.7,
    'bagging_fraction' : 0.7,
    'min_split_gain' : 0.0970905919552776,
    'min_child_weight' : 9.42012323936088,  
}

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, Y)):
    train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
    valid_x, valid_y = X.iloc[valid_idx], Y.iloc[valid_idx] 
    
    lgb_train=lgb.Dataset(train_x,label=train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
    
    gbm = lgb.train(params, lgb_train, num_boost_round=2100, valid_sets=[lgb_train, lgb_eval], 
                    verbose_eval=100)  
    
    oof_preds[valid_idx] = gbm.predict(valid_x[X.columns.values])

    
train['sex_prob_oof'] = oof_preds    


# In[239]:


#用全部的train来预测test
lgb_train = lgb.Dataset(X,label=Y)

gbm = lgb.train(params, lgb_train, num_boost_round=2100, valid_sets=lgb_train, verbose_eval=100)  

test['sex_prob_oof'] = gbm.predict(test[X.columns.values])


# In[240]:


df_sex_prob_oof = pd.concat([train[['device_id', 'sex_prob_oof']], test[['device_id', 'sex_prob_oof']]])
df_sex_prob_oof.columns = ['device_id', 'start_close_sex_prob_oof']


agg_func = {
    'start_close_sex_prob_oof' : ['min', 'max', 'mean', 'std']
}

df_sex_prob_oof = df_sex_prob_oof.groupby('device_id').agg(agg_func)
df_sex_prob_oof.columns = pd.Index(['device_app_grouped' + e[0] + "_" + e[1].upper() for e in df_sex_prob_oof.columns.tolist()])
df_sex_prob_oof = df_sex_prob_oof.reset_index()


# In[242]:


df_sex_prob_oof.to_csv('start_close_sex_prob_oof.csv', index=None)

