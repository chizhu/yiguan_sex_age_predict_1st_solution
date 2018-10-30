# -*- coding:utf-8 -*-

import pandas as pd

df_brand = pd.read_csv('feature/deviceid_brand_feature.csv')
df_lr = pd.read_csv('feature/tfidf_lr_error_single_classfiy.csv')
df_pac = pd.read_csv('feature/tfidf_pac_error_single_classfiy.csv')
df_sgd = pd.read_csv('feature/tfidf_sgd_error_single_classfiy.csv')
df_ridge = pd.read_csv('feature/tfidf_ridge_error_single_classfiy.csv')
df_bnb = pd.read_csv('feature/tfidf_bnb_error_single_classfiy.csv')
df_mnb = pd.read_csv('feature/tfidf_mnb_error_single_classfiy.csv')
df_lsvc = pd.read_csv('feature/tfidf_lsvc_error_single_classfiy.csv')
df_lr_2 = pd.read_csv('feature/tfidf_lr_1_3_error_single_classfiy.csv')
df_pac_2 = pd.read_csv('feature/tfidf_pac_1_3_error_single_classfiy.csv')
df_sgd_2 = pd.read_csv('feature/tfidf_sgd_1_3_error_single_classfiy.csv')
df_ridge_2 = pd.read_csv('feature/tfidf_ridge_1_3_error_single_classfiy.csv')
df_bnb_2 = pd.read_csv('feature/tfidf_bnb_1_3_error_single_classfiy.csv')
df_mnb_2 = pd.read_csv('feature/tfidf_mnb_1_3_error_single_classfiy.csv')
df_lsvc_2 = pd.read_csv('feature/tfidf_lsvc_2_error_single_classfiy.csv')
df_kmeans_2 = pd.read_csv('feature/cluster_2_tfidf_feature.csv')
df_start_close = pd.read_csv('feature/feature_start_close.csv')
df_ling_reg = pd.read_csv('feature/tfidf_ling_reg.csv')
df_par_reg = pd.read_csv('feature/tfidf_par_reg.csv')
df_svr_reg = pd.read_csv('feature/tfidf_svr_reg.csv')
df_w2v = pd.read_csv('feature/w2v_avg.csv')
del df_w2v['DeviceID']
df_best_nn = pd.read_csv('feature/yg_best_nn.csv')
del df_best_nn['DeviceID']
df_chizhu_lgb = pd.read_csv('feature/lgb_feat_chizhu.csv')
del df_chizhu_lgb['DeviceID']
df_chizhu_nn = pd.read_csv('feature/nn_feat.csv')
del df_chizhu_nn['DeviceID']
df_lwl_lgb = pd.read_csv('feature/feat_lwl.csv')
del df_lwl_lgb['DeviceID']
df_feature = pd.concat([
                        df_brand,
                        df_lr, df_pac, df_sgd,
                        df_ridge, df_bnb, df_mnb, df_lsvc,
                        df_start_close, df_ling_reg, df_par_reg,df_svr_reg,
                        df_lr_2, df_pac_2, df_sgd_2, df_ridge_2, df_bnb_2, df_mnb_2,
                        df_lsvc_2, df_kmeans_2, df_w2v, df_best_nn, df_chizhu_lgb, df_chizhu_nn
                        df_lwl_lgb
                        ], axis=1)

df_feature.to_csv('feature/feature_one.csv', encoding='utf8', index=None)

