import os
import pandas as pd
path = "/Users/chizhu/data/competition_data/易观/"
os.listdir(path)

train = pd.read_csv(path+"deviceid_train.tsv", sep="\t",
                    names=["id", "sex", "age"])
test = pd.read_csv(path+"deviceid_test.tsv", sep="\t", names=['DeviceID'])
pred = pd.read_csv(path+"nn_feat_v6.csv")

lgb1 = pd.read_csv(path+"th_results_ems_22_nb_5400.csv")  # 576
lgb1 = pd.merge(test, lgb1, on="DeviceID", how="left")
submit = lgb1.copy()

nn1 = pd.read_csv(path+"xgb_and_nurbs.csv")  # 573
nn1 = pd.merge(test, nn1, on="DeviceID", how="left")

# nn2=pd.read_csv(path+"th_results_ems_2547.csv")##574
# nn2=pd.merge(test,nn2,on="DeviceID",how="left")

# lgb2=pd.read_csv(path+"th_results_ems_2.549.csv")##570
# lgb2=pd.merge(test,lgb2,on="DeviceID",how="left")

# lgb3=pd.read_csv(path+"th_results_ems_2547.csv")##547
# lgb3=pd.merge(test,lgb3,on="DeviceID",how="left")


for i in['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6',
         '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4',
         '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']:
    #     submit[i]=(lgb1[i]+lgb2[i]+nn1[i]+nn2[i])/4.0
    submit[i] = 0.75*lgb1[i]+0.25*nn1[i]
#     submit[i]=0.1*lgb1[i]+0.1*nn1[i]+0.2*nn2[i]+0.2*lgb2[i]+0.4*lgb3[i]

submit.to_csv(path+"th_nurbs_7525.csv", index=False)
