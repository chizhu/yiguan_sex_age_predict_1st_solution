# yiguan_sex_age_predict_1st_solution 
易观性别年龄预测第一名解决方案

##### [比赛链接](https://www.tinymind.cn/competitions/43)
--------

团队是分别个人做然后再合并，所以团队中特征文件有所交叉，主要用到的方案是stacking不同模型，因为数据产出的维度较高，通过不同模型stacking可以达到不会损失过量信息下达到降维的目的。

以下是运行代码的顺序：

* 1.产出特征文件 

> 按照nb_cz_lwl_wcm文件夹运行说明分别运行 nb_cz_lwl_wcm文件夹下的所有文件产出特征文件 feature_one.csv
> 按照thluo 文件夹下运行说明分别运行 thluo 文件夹下的代码生成 thluo_train_best_feat.csv

* 2.模型加权
注：模型所得到的结果在linwangli文件夹下

> 运行完thluo文件夹下面的所有代码会生成thluo_prob
> 用linwangli/code文件夹下面的模型以及上面所求得的特征文件可跑出对应概率文件，相关概率文件加权方案看 linwangli文件夹下面的融合思路ppt

<br>
<br>

CONTRIBUTORS:[THLUO](https://github.com/THLUO)   [WangliLin](https://github.com/WangliLin)   [Puck Wang](https://github.com/PuckWong)   [chizhu](https://github.com/chizhu) [NURBS](https://github.com/suncostanx)






