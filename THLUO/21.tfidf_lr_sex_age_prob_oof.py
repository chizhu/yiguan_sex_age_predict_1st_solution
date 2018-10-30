
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss



# In[2]:

print('21.tfidf_lr.py')
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

deviceid_train = pd.concat([deviceid_train, deviceid_test])


# In[4]:


deviceid_package_start = deviceid_package_start_close[['device_id', 'app_id', 'start_time']]
deviceid_package_start.columns = ['device_id', 'app_id', 'all_time']
deviceid_package_close = deviceid_package_start_close[['device_id', 'app_id', 'close_time']]
deviceid_package_close.columns = ['device_id', 'app_id', 'all_time']
deviceid_package_all = pd.concat([deviceid_package_start, deviceid_package_close])
deviceid_package_all = deviceid_package_all.sort_values(by='all_time')
#deviceid_package_all = deviceid_package_all.merge(deviceid_train, on='device_id', how='left')


# In[5]:


df = deviceid_package_all.groupby('device_id').apply(lambda x : list(x.app_id)).reset_index().rename(columns = {0 : 'app_list'})


# In[6]:


df_sex_prob_oof = pd.read_csv('device_sex_prob_oof.csv')
df_age_prob_oof = pd.read_csv('device_age_prob_oof.csv')
df_start_close_sex_prob_oof = pd.read_csv('start_close_sex_prob_oof.csv')
df_start_close_age_prob_oof = pd.read_csv('start_close_age_prob_oof.csv')
df_start_close_sex_age_prob_oof = pd.read_csv('start_close_sex_age_prob_oof.csv')


gc.collect()
df = df.merge(df_sex_prob_oof, on='device_id', how='left')
df = df.merge(df_age_prob_oof, on='device_id', how='left')
df = df.merge(df_start_close_sex_prob_oof, on='device_id', how='left')
df = df.merge(df_start_close_age_prob_oof, on='device_id', how='left')
df = df.merge(df_start_close_sex_age_prob_oof, on='device_id', how='left')
df.fillna(0, inplace=True)
apps = df['app_list'].apply(lambda x:' '.join(x)).tolist()
del df['app_list']


df = df.merge(deviceid_train, on='device_id', how='left')


# In[8]:


vectorizer=CountVectorizer()
transformer=TfidfTransformer()
cntTf = vectorizer.fit_transform(apps)
tfidf=transformer.fit_transform(cntTf)
word=vectorizer.get_feature_names()
weight=tfidf.toarray()
df_weight=pd.DataFrame(weight)
feature=df_weight.columns


# In[9]:


for i in df.columns.values:
    df_weight[i] = df[i]
    df_weight[i] = df[i]


# In[11]:


df_weight['sex'] = df_weight['sex'].apply(lambda x:str(x))
df_weight['age'] = df_weight['age'].apply(lambda x:str(x))
def tool(x):
    if x == 'nan':
        return x
    else:
        return str(int(float(x)))
df_weight['sex'] = df_weight['sex'].apply(tool)
df_weight['age'] = df_weight['age'].apply(tool)
df_weight['sex_age'] = df_weight['sex']+'-'+df_weight['age']
df_weight['sex_age'] = df_weight.sex_age.replace({'nan':np.NaN,'nan-nan':np.NaN})


# In[12]:


train = df_weight[df_weight.sex_age.notnull()]
train.reset_index(drop=True, inplace=True)
test = df_weight[df_weight.sex_age.isnull()]
test.reset_index(drop=True, inplace=True)
gc.collect()


# In[16]:


X = train.drop(['sex','age','sex_age','device_id'],axis=1)
Y = train['sex_age']
Y_CAT = pd.Categorical(Y)
Y = pd.Series(Y_CAT.codes)


# In[18]:


from sklearn.model_selection import KFold, StratifiedKFold
gc.collect()
seed = 666
num_folds = 5
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)

oof_preds = np.zeros([train.shape[0], 22])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, Y)):
    train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
    valid_x, valid_y = X.iloc[valid_idx], Y.iloc[valid_idx] 
    

    clf = LogisticRegression(C=4)
    clf.fit(train_x, train_y)
    valid_preds=clf.predict_proba(valid_x)
    train_preds=clf.predict_proba(train_x)
    
    oof_preds[valid_idx] = valid_preds
    
    print (log_loss(train_y.values, train_preds), log_loss(valid_y.values, valid_preds))
    
    
oof_train = pd.DataFrame(oof_preds)
oof_train.columns = ['tfidf_lr_sex_age_prob_oof_' + str(i)  for i in range(22)] 
train_temp = pd.concat([train[['device_id']], oof_train], axis=1)    


# In[20]:


#用全部的数据预测
clf = LogisticRegression(C=4)
clf.fit(X, Y)
train_preds=clf.predict_proba(X)
test_preds=clf.predict_proba(test[X.columns])
print (log_loss(Y.values, train_preds))

oof_test = pd.DataFrame(test_preds)
oof_test.columns = ['tfidf_lr_sex_age_prob_oof_' + str(i)  for i in range(22)] 


# In[24]:


oof_test


# In[25]:


test_temp = pd.concat([test[['device_id']], oof_test], axis=1)    
test_temp


# In[26]:


sex_age_oof = pd.concat([train_temp, test_temp])
sex_age_oof


# In[29]:


sex_age_oof.to_csv('tfidf_lr_sex_age_prob_oof.csv', index=None)

