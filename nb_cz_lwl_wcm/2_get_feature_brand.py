# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing

train = pd.read_csv('Demo/deviceid_train.tsv', sep='\t', header=None)
test = pd.read_csv('Demo/deviceid_test.tsv', sep='\t', header=None)

data_all = pd.concat([train, test], axis=0)
data_all = data_all.rename({0:'id'}, axis=1)
del data_all[1],data_all[2]
deviced_brand = pd.read_csv('Demo/deviceid_brand.tsv', sep='\t', header=None)
deviced_brand = deviced_brand.rename({0: 'id'}, axis=1)
data_all = pd.merge(data_all, deviced_brand, on='id', how='left')
print(data_all)
# 直接做类别编码特征

feature = pd.DataFrame()
label_encoder = preprocessing.LabelEncoder()
feature['phone_type'] = label_encoder.fit_transform(data_all[1])
feature['phone_type_detail'] = label_encoder.fit_transform(data_all[2])
feature.to_csv('feature/deviceid_brand_feature.csv', index=False)