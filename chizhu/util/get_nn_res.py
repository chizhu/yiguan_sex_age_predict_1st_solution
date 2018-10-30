import pandas as pd 
path = "/Users/chizhu/data/competition_data/易观/"
res1 = pd.read_csv(path+"res1.csv")
res2_1 = pd.read_csv(path+"res2_1.csv")
res2_2 = pd.read_csv(path+"res2_2.csv")
res1.index = range(len(res1))
res2_1.index = range(len(res2_1))
res2_2.index = range(len(res2_2))
final_1 = res2_1.copy()
final_2 = res2_2.copy()
for i in range(11):
    final_1[str(i)] = res1['sex1']*res2_1[str(i)]
    final_2[str(i)] = res1['sex2']*res2_2[str(i)]
id_list = pred['DeviceID']
final = id_list
final.index = range(len(final))
final.columns = ['DeviceID']
final_pred = pd.concat([final_1, final_2], 1)
final = pd.concat([final, final_pred], 1)
final.columns = ['DeviceID', '1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6',
                 '1-7', '1-8', '1-9', '1-10', '2-0', '2-1', '2-2', '2-3', '2-4',
                 '2-5', '2-6', '2-7', '2-8', '2-9', '2-10']

final.to_csv(path+'nn_feat_v12.csv', index=False)

train = pd.read_csv(path+"deviceid_train.tsv", sep="\t",
                    names=["id", "sex", "age"])
test = pd.read_csv(path+"deviceid_test.tsv", sep="\t", names=['DeviceID'])

pred = pd.read_csv(path+"nn_feat_v6.csv")
sub = pd.merge(test, pred, on="DeviceID", how="left")

sub.to_csv(path+"nn_v6.csv", index=False)


