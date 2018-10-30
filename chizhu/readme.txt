|-single_model/
    |-data/ 处理后的特征和数据存放位置
    |-model/ 模型文件
    |-submit 模型概率文件，可用作stacking材料
    |-config.py 配置原始文件路径
    |-user_behavior.py 得到user_behavior特征集
    |-get_nn_feat.py 获得nn 的统计特征输入
    |-lgb.py
    |-xgb.py
    |-xgb_nb.py 条件概率
    |-cnn.py
    |-deepnn.py 
    |-yg_best_nn.py 
|-stacking/
    |-all_feat/ 使用全部概率文件的xgb的条件概率
    |-nurbs_feat/ 使用rurbs概率文件的xgb的22分类以及条件概率
        |-xgb_nurbs_nb.py 条件概率
        |-xgb_22.py 22分类
|-util/
    |-bagging.py  加权融合脚本
    |-get_nn_res.py 获得nn概率文件和可提交的结果


使用说明:
single_model:1)先配置config.py 里的文件路径
             2)运行user_behavior.py 
             3)运行get_nn_feat.py 
             4)然后可以逐个运行nn或者tree模型，得到的概率文件在submit/

stacking：这里直接运行是不行的 因为需要概率文件，大小在2G左右，没有附上，之后可以找我们要
util:加权用，这里需要的是stacking/nurbs_feat下的xgb_22.py和_xgbnb.py产生的结果取均值得到一份结果,xgb_22_nb.csv

