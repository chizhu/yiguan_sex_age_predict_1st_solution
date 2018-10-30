import pandas as pd
import numpy as np

def weights_ensemble(results, weights):
	'''
	针对此次比赛的按权重进行模型融合的函数脚本
	results: list，存放所有需要融合的结果路径
	weights: list, 存放各个结果的权重
	return: 可以直接to_csv提交的结果
	'''
    for i in range(len(results)):
        if i == 0:
            sub = pd.read_csv(results[0])
            final_cols = list(sub.columns)
            cols = list(sub.columns)
            cols[1:]  = [col + '_0' for col in cols[1:]]
            sub.columns = cols
        else:
            result = pd.read_csv(results[i])
            cols = list(result.columns)
            cols[1:]  = [col + '_' + str(i) for col in cols[1:]]
            result.columns = cols
            sub = pd.merge(left=sub, right=result, on='DeviceID')
    for i in range(len(weights)):
        for col in final_cols[1:]:
            if col not in sub.columns:
                sub[col] = weights[i] * sub[col + '_' + str(i)]
            else:
                sub[col] = sub[col] +  weights[i] * sub[col + '_' + str(i)]
    sub = sub[final_cols]
    return sub

def result_corr(path1, path2):
	'''
	根据此次比赛写的评测不同提交结果相关性文件
	path1: 结果1的路径
	path2: 结果2的路径
	return： 返回不同提交结果的相关性
	'''
	result_1 = pd.read_csv(path1)
	result_2 = pd.read_csv(path2)
	result = pd.merge(left=result_1, right=result_2, on='DeviceID', suffixes=('_x', '_y'))
	cols = result_1.columns[1:]
	col_list = []
	for col in cols:
	    col_pair = [col + '_x', col + '_y']
	    col_list.append(result[col_pair].corr().loc[col + '_x', col + '_y'])

	return np.mean(col_list)