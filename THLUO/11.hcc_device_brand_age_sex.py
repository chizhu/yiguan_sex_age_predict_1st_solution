
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
from sklearn.model_selection import StratifiedKFold



# In[2]:

print ('11.hcc_device_brand_age_sex.py')
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


#deviceid_brand['device_brand'] = deviceid_brand.device_brand.apply(lambda x : str(x).split(' ')[0])


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


df_train = deviceid_train.merge(deviceid_brand, how='left', on='device_id')
df_train.fillna(-1, inplace=True)
df_test = deviceid_test.merge(deviceid_brand, how='left', on='device_id')
df_test.fillna(-1, inplace=True)


# In[5]:


df_train['sex'] = df_train.sex.apply(lambda x : 1 if x == 1 else 0)
df_train = df_train.join(pd.get_dummies(df_train["age"], prefix="age").astype(int))
df_train['sex_age'] = df_train['sex'].map(str) + '_' + df_train['age'].map(str)
Y = df_train['sex_age']
Y_CAT = pd.Categorical(Y)
df_train['sex_age'] = pd.Series(Y_CAT.codes)
df_train = df_train.join(pd.get_dummies(df_train["sex_age"], prefix="sex_age").astype(int))


# In[6]:


sex_age_columns = ['sex_age_' + str(i) for i in range(22)]
sex_age_prior_set = df_train[sex_age_columns].mean().values
age_columns = ['age_' + str(i) for i in range(11)]
age_prior_set = df_train[age_columns].mean().values
sex_prior_prob= df_train.sex.mean()
sex_prior_prob


# In[7]:


def hcc_encode(train_df, test_df, variable, target, prior_prob, k=5, f=1, g=1, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return


# In[8]:


#拟合年龄
#拟合测试集
# High-Cardinality Categorical encoding
skf = StratifiedKFold(5)
nums = 11
for variable in ['device_brand', 'device_type'] : 
    for i in range(nums) :
        target = age_columns[i]
        age_prior_prob = age_prior_set[i]
        print (variable, target, age_prior_prob)
        hcc_encode(df_train, df_test, variable, target, age_prior_prob, k=5, f=1, g=1, update_df=None)
        #拟合验证集
        for train, test in skf.split(np.zeros(len(df_train)), df_train['age']):
            hcc_encode(df_train.iloc[train], df_train.iloc[test], variable, target, age_prior_prob, k=5, update_df=df_train)        


# In[9]:


#拟合性别
#拟合测试集
# High-Cardinality Categorical encoding
skf = StratifiedKFold(5)
for variable in ['device_brand', 'device_type'] : 
    target = 'sex'
    print (variable, target, sex_prior_prob)
    hcc_encode(df_train, df_test, variable, target, sex_prior_prob, k=5, f=1, g=1, update_df=None)
    #拟合验证集
    for train, test in skf.split(np.zeros(len(df_train)), df_train['age']):
        hcc_encode(df_train.iloc[train], df_train.iloc[test], variable, target, sex_prior_prob, k=5, f=1, g=1, update_df=df_train)        


# In[10]:


#拟合性别年龄
#拟合测试集
# High-Cardinality Categorical encoding
skf = StratifiedKFold(5)
nums = 22
for variable in ['device_brand', 'device_type'] : 
    for i in range(nums) :
        target = sex_age_columns[i]
        sex_age_prior_prob = sex_age_prior_set[i]
        print (variable, target, sex_age_prior_prob)
        hcc_encode(df_train, df_test, variable, target, sex_age_prior_prob, k=5, f=1, g=1, update_df=None)
        #拟合验证集
        for train, test in skf.split(np.zeros(len(df_train)), df_train['sex_age']):
            hcc_encode(df_train.iloc[train], df_train.iloc[test], variable, target, sex_age_prior_prob, k=5, update_df=df_train)        


# In[14]:


hcc_columns = ['device_id'] + ['hcc_device_brand_age_' + str(i) for i in range(11)] + ['hcc_device_brand_sex'] + ['hcc_device_type_age_' + str(i) for i in range(11)] + ['hcc_device_type_sex'] + ['hcc_device_type_sex_age_' + str(i) for i in range(22)]  
df_total = pd.concat([df_train[hcc_columns], df_test[hcc_columns]])


# In[15]:


df_total.to_csv('hcc_device_brand_age_sex.csv', index=None)

