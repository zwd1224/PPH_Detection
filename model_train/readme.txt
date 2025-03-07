pip_list：库函数安装版本，有需要再安装
PPH产后出血数据集：只用前面588（294+294）个样本，共48个特征：孕周	第一产程时间	第二产程时间	第三产程时间	总产程时间	羊水量	年龄	孕次	产次	临产方式	胎膜破裂方式	前羊水性质	前羊水量	胎儿娩出方式	胎盘娩出方式	胎盘完整	胎盘重量	胎盘长	胎盘宽	胎盘厚	胎膜完整	脐带长	脐带情况	脐带扭转	胎盘附着	宫颈情况	宫颈外缝针数	宫颈内埋缝合	会阴	阴道壁裂伤	缩宫素	益母草	欣母沛	卡贝	新生儿性别	出生体重	头围	身长	胸围	出生时情况	呼吸	转归	身高	体重	BMI	学历	复方氯化钠	阴道检查次数

其中patientID（病号，每个病人只有一个）、visitNo（住院号，主码）为病人的序号，其中产时出血、产后两小时内出血、总出血、出血量为结果特征
【腾讯文档】数据集补充说明（没有找到的特征描述，可以查看之前历史版本）
https://docs.qq.com/sheet/DRmdHcHJ5VnZMWHhs?tab=BB08J2
其他数据集，在main函数里面进行解释
restore.py：使用训练好的模型进行预测
main.py：主函数，决策路径等功能在这里面
modes.py：base_model模进行修改
base_model：原始模型设计
utils.py：使用到的一些函数
参考：https://github.com/AlCorreia/Human-in-the-loop-Feature-Selection，不同的地方就是修改的地方
不同数据集的超参选择：
产后出血PPH:
{'strength': 300, 'model': 'RL', 'optimizer': 'Adam', 'learning_rate': 0.05, 'epochs': 5, 'batch_size': 64, 'num_cat': 2, 'dir': './Base_model', 'arch': [128], 'num_filters': 256, 'num_features': 48, 'num_samples': 8, 'nonlinearity': 'tf.nn.relu', 'frame_size': 60, 'feedback_distance': 'mse', 'second_conv': False, 'dropout': 0.5, 'initial_tau': 10.0, 'tau_decay': True, 'reg': 1, 'pre_train': False, 'number_patches': 10}
术后恶心呕吐PONV:
{'strength': 300, 'model': 'RL', 'optimizer': 'Adam', 'learning_rate': 0.001, 'epochs': 15, 'batch_size': 64, 'num_cat': 2, 'dir': './Base_model', 'arch': [256], 'num_filters': 128, 'num_features': 31, 'num_samples': 8, 'nonlinearity': 'tf.nn.relu', 'frame_size': 60, 'feedback_distance': 'cosine', 'second_conv': False, 'dropout': 1, 'initial_tau': 10.0, 'tau_decay': True, 'reg': 1.0, 'pre_train': False, 'number_patches': 10}
产后抑郁PPD：
{'strength': 300, 'model': 'RL', 'optimizer': 'Adam', 'learning_rate': 0.05, 'epochs': 21, 'batch_size': 64, 'num_cat': 2, 'dir': './Base_model', 'arch': [512], 'num_filters': 512, 'num_features': 9, 'num_samples': 8, 'nonlinearity': 'tf.nn.relu', 'frame_size': 60, 'feedback_distance': 'cosine', 'second_conv': False, 'dropout': 1, 'initial_tau': 10.0, 'tau_decay': True, 'reg': 1.0, 'pre_train': False, 'number_patches': 10}
妊娠期糖尿病GDM：
{'strength': 300, 'model': 'RL', 'optimizer': 'Adam', 'learning_rate': 0.05, 'epochs': 12, 'batch_size': 128, 'num_cat': 2, 'dir': './Base_model', 'arch': [512], 'num_filters': 512, 'num_features': 15, 'num_samples': 8, 'nonlinearity': 'tf.nn.relu', 'frame_size': 60, 'feedback_distance': 'cosine', 'second_conv': False, 'dropout': 1, 'initial_tau': 10.0, 'tau_decay': True, 'reg': 1.0, 'pre_train': False, 'number_patches': 10}






