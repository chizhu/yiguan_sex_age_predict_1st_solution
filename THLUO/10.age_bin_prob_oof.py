
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

print ('10.age_bin_prob_oof.py')
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


# In[5]:


deviceid_train=pd.concat([deviceid_train,deviceid_test])


# In[6]:


deviceid_packages['apps']=deviceid_packages['apps'].apply(lambda x:x.split(','))
deviceid_packages['app_lenghth']=deviceid_packages['apps'].apply(lambda x:len(x))




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


# In[10]:


lda = LatentDirichletAllocation(n_topics=5,
                                learning_offset=50.,
                                random_state=666)
docres = lda.fit_transform(cntTf)


# In[11]:


deviceid_packages = pd.concat([deviceid_packages,pd.DataFrame(docres)],axis=1)


# In[12]:


temp=deviceid_packages.drop('apps',axis=1)
deviceid_train=pd.merge(deviceid_train,temp,on='device_id',how='left')


# In[13]:


#解析出所有的device_app_pair
device_id_arr = []
app_arr = []
df_device_app_pair = pd.DataFrame()
for row in deviceid_packages.values :
    device_id = row[0]
    app_list = row[1]
    for app in app_list :
        device_id_arr.append(device_id)
        app_arr.append(app)
#生成pair        
df_device_app_pair['device_id'] = device_id_arr
df_device_app_pair['app_id'] = app_arr    

df_device_app_pair = df_device_app_pair.merge(package_label, how='left', on='app_id')

#特征工程
def open_app_timegap_in_hour() :
    df_temp = deviceid_package_start_close.groupby(['device_id', 'start_hour'])['time_gap'].mean().reset_index().rename(columns = {'time_gap': 'mean_time_gap'})
    df_mean_temp = pd.pivot_table(df_temp, index='device_id', columns='start_hour', values='mean_time_gap').reset_index()
    df_mean_temp.columns = ['device_id'] + ['open_app_timegap_in_'+str(i) + '_mean_hour' for i in range(0,24)]
    df_mean_temp.fillna(0, inplace=True)


    
    return df_mean_temp


# In[8]:


def device_start_end_app_timegap() :
    #用户打开，关闭app的时间间隔
    df_ = deviceid_package_start_close.sort_values(by=['device_id', 'start_date'], ascending=False)
    df_['prev_start_date'] = df_.groupby('device_id')['start_date'].shift(-1)
    df_['start_date_gap'] = (df_['start_date'] - df_['prev_start_date']).astype('timedelta64[s]')
    agg_dic = {'start_date_gap' : ['min', 'max', 'mean', 'median', 'std']}
    df_start_gap_agg = df_.groupby('device_id').agg(agg_dic)
    df_start_gap_agg.columns = pd.Index(['device_' + e[0] + "_" + e[1].upper() for e in df_start_gap_agg.columns.tolist()])
    df_start_gap_agg = df_start_gap_agg.reset_index()
    #del df_
    gc.collect()
    #关闭时间间隔
    df_ = deviceid_package_start_close.sort_values(by=['device_id', 'end_date'], ascending=False)
    df_['prev_end_date'] = df_.groupby('device_id')['end_date'].shift(-1)
    df_['end_date_gap'] = (df_['end_date'] - df_['prev_end_date']).astype('timedelta64[s]')
    agg_dic = {'end_date_gap' : ['min', 'max', 'mean', 'median', 'std']}
    df_end_gap_agg = df_.groupby('device_id').agg(agg_dic)
    df_end_gap_agg.columns = pd.Index(['device_' + e[0] + "_" + e[1].upper() for e in df_end_gap_agg.columns.tolist()])
    df_end_gap_agg = df_end_gap_agg.reset_index()
    #del df_
    gc.collect()



    df_agg = df_start_gap_agg.merge(df_end_gap_agg, on='device_id', how='left')
    #df_agg = df_agg.merge(df_app_start_gap_agg, on='device_id', how='left')
    #df_agg = df_agg.merge(df_app_end_gap_agg, on='device_id', how='left')
    return df_agg

def open_app_counts_in_hour() :
    df_temp = deviceid_package_start_close.groupby(['device_id', 'start_hour'])['app_id'].count().reset_index().rename(columns = {'app_id': 'app_counts'})
    df_temp = pd.pivot_table(df_temp, index='device_id', columns='start_hour', values='app_counts').reset_index()
    df_temp.columns = ['device_id'] + ['open_app_counts_in'+str(i) + '_hour' for i in range(0,24)]
    df_temp.fillna(0, inplace=True)
    return df_temp

def close_app_counts_in_hour() :
    df_temp = deviceid_package_start_close.groupby(['device_id', 'end_hour'])['app_id'].count().reset_index().rename(columns = {'app_id': 'app_counts'})
    df_temp = pd.pivot_table(df_temp, index='device_id', columns='end_hour', values='app_counts').reset_index()
    df_temp.columns = ['device_id'] + ['close_app_counts_in'+str(i) + '_hour' for i in range(0,24)]
    df_temp.fillna(0, inplace=True)
    return df_temp

def app_type_mean_time_gap_one_hot () :
    df_temp = deviceid_package_start_close.groupby(['device_id', 'app_parent_type'])['time_gap'].mean().reset_index()
    df_temp = pd.pivot_table(df_temp, index='device_id', columns='app_parent_type', values='time_gap').reset_index()
    df_temp.columns = ['device_id'] + ['app_parent_type_mean_time_gap'+str(i) for i in range(-1,45)]
    df_temp.fillna(-1, inplace=True)
    return df_temp  

def device_active_hour() :
    aggregations = {
        'start_hour' : ['std','mean','max','min'],
        'end_hour' : ['std','mean','max','min']
    }
    df_agg = deviceid_package_start_close.groupby('device_id').agg(aggregations)
    df_agg.columns = pd.Index(['device_grouped_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()   
    
    return df_agg


def device_brand_encoding() :
    df_temp = deviceid_brand.merge(deviceid_train[['device_id', 'age', 'sex']], on='device_id', how='left')

    aggregations = {
        'age' : ['std','mean'],
        'sex' : ['mean'],
    }

    df_device_brand = df_temp.groupby('device_brand').agg(aggregations)
    df_device_brand.columns = pd.Index(['device_brand_' + e[0] + "_" + e[1].upper() for e in df_device_brand.columns.tolist()])
    df_device_brand = df_device_brand.reset_index()

    df_device_type = df_temp.groupby('device_type').agg(aggregations)
    df_device_type.columns = pd.Index(['device_type_' + e[0] + "_" + e[1].upper() for e in df_device_type.columns.tolist()])
    df_device_type = df_device_type.reset_index()

    df_temp = df_temp.merge(df_device_brand, on='device_brand', how='left')
    df_temp = df_temp.merge(df_device_type, on='device_type', how='left')

    aggregations = {
        'device_brand_age_STD' : ['mean'],
        'device_brand_age_MEAN' : ['mean'],
        'device_brand_sex_MEAN' : ['mean'],
        #'device_type_age_STD' : ['mean'],
        #'device_type_age_MEAN' : ['mean'],
        #'device_type_sex_MEAN' : ['mean']
    }

    df_agg = df_temp.groupby('device_id').agg(aggregations)
    df_agg.columns = pd.Index(['device_grouped_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()
    return df_agg


#统计device运行app的情况
def device_active_time_time_stat() :
    #device开启app的时间统计信息
    deviceid_package_start_close['active_time'] = deviceid_package_start_close['close_time'] - deviceid_package_start_close['start_time']

    #device开启了多少次app
    #device开启了多少个app
    aggregations = {
        'app_id' : ['count', 'nunique'],
        'active_time' : ['mean', 'std', 'max', 'min'],
    }
    df_agg = deviceid_package_start_close.groupby('device_id').agg(aggregations)
    df_agg.columns = pd.Index(['device_grouped_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()

    aggregations = {
        'active_time' : ['mean', 'std', 'max', 'min', 'count'],
    }
    df_da_agg = deviceid_package_start_close.groupby(['device_id', 'app_id']).agg(aggregations)
    df_da_agg.columns = pd.Index(['device_app_grouped_' + e[0] + "_" + e[1].upper() for e in df_da_agg.columns.tolist()])
    df_da_agg = df_da_agg.reset_index()

    #device开启app的平均时间
    aggregations = {
        'device_app_grouped_active_time_MEAN' : ['mean', 'std', 'max', 'min'],
        'device_app_grouped_active_time_STD' : ['mean', 'std', 'max', 'min'],
        'device_app_grouped_active_time_MAX' : ['mean', 'std', 'max', 'min'],
        'device_app_grouped_active_time_MIN' : ['mean', 'std', 'max', 'min'],
        'device_app_grouped_active_time_COUNT' : ['mean', 'std', 'max', 'min'],
    }
    df_temp = df_da_agg.groupby(['device_id']).agg(aggregations)
    df_temp.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_temp.columns.tolist()])
    df_temp = df_temp.reset_index()

    df_agg = df_agg.merge(df_temp, on='device_id', how='left')
    return df_agg


def app_type_encoding() :
    df_temp = df_device_app_pair.merge(deviceid_train[['device_id', 'age', 'sex']], on='device_id', how='left')

    aggregations = {
        'age' : ['std','mean'],
        'sex' : ['mean'],
    }

    df_agg_app_parent_type = df_temp.groupby('app_parent_type').agg(aggregations)
    df_agg_app_parent_type.columns = pd.Index(['app_parent_type_' + e[0] + "_" + e[1].upper() for e in df_agg_app_parent_type.columns.tolist()])
    df_agg_app_parent_type = df_agg_app_parent_type.reset_index()

    df_agg_app_child_type = df_temp.groupby('app_child_type').agg(aggregations)
    df_agg_app_child_type.columns = pd.Index(['app_child_type_' + e[0] + "_" + e[1].upper() for e in df_agg_app_child_type.columns.tolist()])
    df_agg_app_child_type = df_agg_app_child_type.reset_index()

    df_temp = df_temp.merge(df_agg_app_parent_type, on='app_parent_type', how='left')
    df_temp = df_temp.merge(df_agg_app_child_type, on='app_child_type', how='left')

    aggregations = {
        'app_parent_type_age_STD' : ['mean'],
        'app_parent_type_age_MEAN' : ['mean'],
        'app_parent_type_sex_MEAN' : ['mean'],
        'app_child_type_age_STD' : ['mean'],
        'app_child_type_age_MEAN' : ['mean'],
        'app_child_type_sex_MEAN' : ['mean']
    }
    
    df_agg = df_temp.groupby('device_id').agg(aggregations)
    df_agg.columns = pd.Index(['device_grouped_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
    df_agg = df_agg.reset_index()
    return df_agg

#每个device对应的app_parent_type计数
def app_type_onehot_in_device(df) :
    df_copy = df.fillna(-1)
    df_temp = df_copy.groupby(['device_id', 'app_parent_type'])['app_id'].size().reset_index()
    df_temp.rename(columns = {'app_id' : 'app_parent_type_counts'}, inplace=True)
    df_temp = pd.pivot_table(df_temp, index='device_id', columns='app_parent_type', values='app_parent_type_counts').reset_index()
    df_temp.columns = ['device_id'] + ['app_parent_type'+str(i) for i in range(-1,45)]
    df_temp.fillna(0, inplace=True)
    return df_temp



# In[15]:


#提取特征
df_train = deviceid_train.merge(device_active_time_time_stat(), on='device_id', how='left')
df_train = df_train.merge(deviceid_brand, on='device_id', how='left')
df_train = df_train.merge(app_type_onehot_in_device(df_device_app_pair), on='device_id', how='left')
df_train = df_train.merge(app_type_encoding(), on='device_id', how='left')
df_train = df_train.merge(device_active_hour(), on='device_id', how='left')
df_train = df_train.merge(app_type_mean_time_gap_one_hot(), on='device_id', how='left')
df_train = df_train.merge(open_app_counts_in_hour(), on='device_id', how='left')
df_train = df_train.merge(close_app_counts_in_hour(), on='device_id', how='left')
df_train = df_train.merge(device_brand_encoding(), on='device_id', how='left')
df_train = df_train.merge(device_start_end_app_timegap(), on='device_id', how='left')
df_train = df_train.merge(open_app_timegap_in_hour(), on='device_id', how='left')


# In[16]:


df_w2c_start = pd.read_csv('device_start_app_w2c.csv')
df_w2c_close = pd.read_csv('device_close_app_w2c.csv')
df_w2c_all = pd.read_csv('device_all_app_w2c.csv')
df_device_quchong_start_app_w2c = pd.read_csv('device_quchong_start_app_w2c.csv')
df_device_app_unique_start_app_w2c = pd.read_csv('device_app_unique_start_app_w2c.csv')
df_device_app_unique_close_app_w2c = pd.read_csv('device_app_unique_close_app_w2c.csv')
df_device_app_unique_all_app_w2c = pd.read_csv('device_app_unique_all_app_w2c.csv')


# In[17]:


df_train_w2v = df_train.merge(df_w2c_start, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_w2c_close, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_w2c_all, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_quchong_start_app_w2c, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_app_unique_start_app_w2c, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_app_unique_close_app_w2c, on='device_id', how='left')
df_train_w2v = df_train_w2v.merge(df_device_app_unique_all_app_w2c, on='device_id', how='left')


# In[19]:


df_train_w2v['sex'] = df_train_w2v['sex'].apply(lambda x:str(x))
df_train_w2v['age'] = df_train_w2v['age'].apply(lambda x:str(x))
def tool(x):
    if x=='nan':
        return x
    else:
        return str(int(float(x)))
df_train_w2v['sex']=df_train_w2v['sex'].apply(tool)
df_train_w2v['age']=df_train_w2v['age'].apply(tool)
df_train_w2v['sex_age']=df_train_w2v['sex']+'-'+df_train_w2v['age']

df_train_w2v = df_train_w2v.replace({'nan':np.NaN,'nan-nan':np.NaN})


# In[31]:


train = df_train_w2v[df_train_w2v['sex_age'].notnull()]
test = df_train_w2v[df_train_w2v['sex_age'].isnull()]
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[32]:


Y = train['age']
train['label'] = Y


# In[35]:


from sklearn.model_selection import KFold, StratifiedKFold
label_set = train.label.unique()
lgb_round = {'3': 363,
             '5': 273,
             '4': 328,
             '7': 228,
             '6': 361,
             '9': 181,
             '10': 338,
             '2': 312,
             '8': 234,
             '1': 220,
             '0': 200}
for sex_age in label_set :
    print (sex_age)
    X = train.drop(['sex', 'age', 'sex_age', 'label', 'device_id'],axis=1)
    Y = train.label.apply(lambda x : 1 if x == sex_age else 0)
    print (Y.value_counts())
    seed = 2018
    num_folds = 5
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)

    sub_list = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])

    params = {
        'boosting_type': 'gbdt',
        'learning_rate' : 0.02,
        #'max_depth':5,
        'num_leaves' : 2 ** 5,
        'metric': {'binary_logloss'},
        #'num_class' : 22,
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

        gbm = lgb.train(params, lgb_train, num_boost_round=lgb_round[sex_age], valid_sets=[lgb_train, lgb_eval], verbose_eval=50)  

        oof_preds[valid_idx] = gbm.predict(valid_x[X.columns.values])


    train['age_bin_prob_oof_' + str(sex_age)] = oof_preds      
    
    
    #用全部的train来预测test
    lgb_train = lgb.Dataset(X,label=Y)

    gbm = lgb.train(params, lgb_train, num_boost_round=lgb_round[sex_age], valid_sets=lgb_train, verbose_eval=50)  

    test['age_bin_prob_oof_' + str(sex_age)] = gbm.predict(test[X.columns.values])


# In[36]:


columns = ['device_id'] + ['age_bin_prob_oof_' + str(i) for i in range(11)]


# In[38]:


pd.concat([train[columns], test[columns]]).to_csv('age_bin_prob_oof.csv', index=None)

