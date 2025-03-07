# 修改模型内的参数
import datetime
import argparse
import time
import math
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm
from models import Baseline, Bernoulli_Net
from utils import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from base_model import average_path_length,fit_tree,avg_path_length
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, fbeta_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score
import tensorflow as tf
# 指定要使用的两个GPU设备
import os
import joblib


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
start_time = time.time()

def str2bool(v):
    """ Converts a string to a boolean value. """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(self):
    # Define the deep learning model

    if FLAGS.model == 'Base':
        pre_training = False
        kernlen = int(FLAGS.frame_size/2)
        net = Baseline(directory=FLAGS.dir, optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate, layer_sizes=FLAGS.arch, num_features=FLAGS.num_features,
                       num_filters=FLAGS.num_filters, frame_size=FLAGS.frame_size)
    elif FLAGS.model == 'RL':
        kernlen = int(FLAGS.frame_size/2)
        net = Bernoulli_Net(layer_sizes=FLAGS.arch, optimizer=FLAGS.optimizer, num_filters=FLAGS.num_filters,
        num_features=FLAGS.num_features, num_samples=FLAGS.num_samples, frame_size=FLAGS.frame_size,
                            learning_rate=FLAGS.learning_rate, feedback_distance=FLAGS.feedback_distance,
                            directory=FLAGS.dir, second_conv=FLAGS.second_conv,strength=FLAGS.strength)

    batch_size = FLAGS.batch_size

    def model_pre_training(X_train, y_train, X_test, y_test):
        print("Pre_Training")
        for epoch in tqdm(range(FLAGS.epochs)):
            _x, _y = input_fn(X_test, y_test, batch_size=batch_size)
            net.evaluate(X_test, y_test, pre_trainining=True)
            net.save()
            X_train, y_train = shuffle_in_unison(X_train, y_train)
            for i in range(0, len(X_train), batch_size):
                _x, _y = X_train.iloc[i:i + batch_size], y_train.iloc[i:i + batch_size]
                net.pre_train(_x, _y,dropout=FLAGS.dropout)
    # cnn
    def model_training(X_train, y_train, X_test, y_test):
        print("Training")

        for epoch in tqdm(range(FLAGS.epochs)):
            # 模型测试
            _x, _y = input_fn(X_test, y_test, batch_size=len(X_test))
            net.evaluate(_x, _y)
            net.save()
            # 打乱训练集顺序（保证每次epoch，训练集顺序不相同）
            X_train, y_train= shuffle_in_unison(X_train, y_train)
            for i in range(0, len(X_train), batch_size):
                # 按照batch_size大小进行训练
                _x, _y = X_train[i:i + batch_size], y_train[i:i + batch_size]
                net.train(_x, _y, dropout=FLAGS.dropout)
    # cnn+tree
    def model_cnnTree_training(X_train, y_train, X_test, y_test):
        print("cnnTree Training")
        # 初始化mlp，根据网络weight预测，路径长度apl
        # warm_start=True,在之前的模型基础上进行训练
        mlp = MLPRegressor(warm_start=True)
        for epoch in tqdm(range(FLAGS.epochs)):
            timeft1 = time.time()
            # # 模型测试
            _x, _y = input_fn(X_test, y_test, batch_size=len(X_test))
            net.evaluate(_x, _y)
            net.save()
            # 参数初始化
            weights = []
            apls = []
            # 打乱训练集顺序
            X_train, y_train = shuffle_in_unison(X_train, y_train)
            for i in range(0, len(X_train), batch_size):
                # 按照batch_size大小进行训练
                _x, _y= X_train[i:i + batch_size],y_train[i:i + batch_size]
                net.cnn_tree_train(_x, _y, dropout=FLAGS.dropout)
                #计算apl
                y_hat = net.predict(X_train, flag=True)
                y_pred = np.argmax(y_hat, axis=1)
                weight = net.saved_weight
                weights.append(weight)
                apl,tree = average_path_length(X_train, y_pred)
                apls.append(apl)
            # _apls是预测出来的,apls是计算得出来的
            apls3 = np.round(apls, 3)
            # mlp训练
            mlp.fit(weights, apls)
            # 查看apls情况
            print('')
            print('计算-apls :', apls3)
            print('-apls均值 :', np.round(np.mean(apls3), 3))
            # 预测_apls3
            _apls3 = mlp.predict(weights)
            _apls3 = np.round(_apls3, 3).tolist()
            _apls_mean = np.round(np.mean(_apls3), 3)
            # 传入到目标函数
            # print('传入的apl', _apls_mean)
            net.get_apl(_apls_mean)
            timeft2 = time.time()
            # print(f"反馈一个eopch耗时: {timeft2 - timeft1:.2f} seconds.")
        # get_tree(tree)
        X_test_original = scaler.inverse_transform(X_test)
        # 保存决策树模型
        joblib.dump(tree, 'decision_tree_model.joblib')
        print("Model saved as decision_tree_model.joblib")
        # 加载决策树模型
        tree1 = joblib.load('decision_tree_model.joblib')
        get_decision_path(tree1, X_test, X_test_original, visitNo_test)
    def model_feedback_training(X_train, y_train, X_test, y_test, train_coords):
        print("Feedback Training")
        for epoch in tqdm(range(FLAGS.epochs)):
            timef1 = time.time()
            _x, _y = input_fn(X_test, y_test, batch_size=len(X_test))
            net.evaluate(_x, _y)
            net.save()
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y, _train_coords = X_train[i:i + batch_size],\
                y_train[i:i + batch_size], train_coords[i:i + batch_size]
                net.feedback_train(_x, _y, _train_coords, dropout=FLAGS.dropout)
            timef2 = time.time()
            # print('')
            # print(f"反馈一个eopch耗时: {timef2 - timef1:.2f} seconds.")

    # cnn+feedback+tree
    def model_feedbackTree_training(X_train, y_train, X_test, y_test, train_coords):
        print("FeedbackTree Training")
        mlp = MLPRegressor(warm_start=True)
        for epoch in tqdm(range(FLAGS.epochs*3)):
            timeft1 = time.time()
            _x, _y = input_fn(X_test, y_test, batch_size=len(X_test))
            net.evaluate(_x, _y)
            net.save()
            weights = []
            apls = []
            _apls = []
            X_train, y_train, train_coords = shuffle_in_unison(X_train, y_train, train_coords)
            for i in range(0, len(X_train), batch_size):
                _x, _y, _train_coords = X_train[i:i + batch_size], \
                    y_train[i:i + batch_size], train_coords[i:i + batch_size]
                net.feedback_tree_train(_x, _y, _train_coords, dropout=FLAGS.dropout)
                y_hat = net.predict(X_train, flag=True)
                y_pred = np.argmax(y_hat, axis=1)
                weight = net.saved_weight
                weights.append(weight)
                apl, tree = average_path_length(X_train, y_pred)
                # print('apl:',apl)
                apls.append(apl)
            # _apls是预测出来的,apls是计算得出来的
            # mlp训练
            apls3 = np.round(apls, 3)
            mlp.fit(weights, apls)
            print('')
            print('计算-apls :', apls3)
            print('-apls均值 :', np.round(np.mean(apls3), 3))
            _apls3 = mlp.predict(weights)
            _apls3 = np.round(_apls3, 3).tolist()
            _apls_mean = np.round(np.mean(_apls3), 3)
            print('传入的apl', _apls_mean)
            net.get_apl(_apls_mean)
            timeft2 = time.time()
            # print(f"反馈一个eopch耗时: {timeft2 - timeft1:.2f} seconds.")
        get_tree(tree)

    def get_tree(tree):
        print('决策树')
        target_name = ['0', '1']
        # feature_names = [f'X[{i}]' for i in range(len(columns))]
        # PPD
        columns=["年龄", "感到悲伤或流泪", "对婴儿和伴侣易怒",
                       "睡眠障碍", "注意力不集中", "过度进食或食欲丧失",
                       "感到内疚", "与婴儿建立亲密关系困难", "自杀倾向"]
        feature_names = ["Age","SadCry","IBP","SleepDist","Inatten","OELA",
                         "Guilt","DiffBond","SuiTen"]
        # GDM
        # columns = ["年龄", "孕次", "前次怀孕妊娠周期", "体重指数",
        #                  "高密度脂蛋白", "家族病史", "产前流产",
        #                  "巨大儿或出生缺陷", "多囊卵巢综合症", "收缩压",
        #                  "舒张压", "口服葡萄糖耐量试验", "血红蛋白",
        #                  "久坐生活方式", "糖尿病前期"]
        # feature_names = ["Age","Grav","PPDur","BMI","HDL","FamHistD","PrenAb",
        #                  "MBD","PCOS","SBP","DBP","OGTT","Hb","Sedent","Prediab"]
        # PONV
        # feature_names = ["Age","BMI","HMS","SH","PHPONV","OS","Lap","Hys","SP",
        #                  "ASA","Mid","Eto","Sev","Roc","Neo","Suf","Fen","Dez",
        #                  "ETI","LM","DoA","DoS","TIFI","Met","Tro","Dex","Pit",
        #                  "Oxy","PCIA","PPS","POS"]
        # PPH
        # feature_names =["GAge", "D1St", "D2St", "D3St", "TLD", "AFV", "Age",
        #                 "NPreg", "NDel", "LOM", "MRM", "AAN", "AAFV", "FDM",
        #                 "PDM", "PI", "PWt", "PL", "PW", "PT", "MI", "UCL",
        #                 "UCC", "UCT", "PA", "CC", "CSN", "CBS", "Perineum",
        #                 "VWL", "OU", "MU", "XU", "CU", "NBG", "BW", "HC",
        #                 "BL", "CChest", "BC", "Respiration", "MNW", "Height",
        #                 "Weight", "BMI", "EL", "CSCU", "VE"]

        dot_data1 = export_graphviz(tree, out_file=None,
                                   class_names=target_name, filled=True, rounded=True,
                                   feature_names=columns)
        dot_data1 = dot_data1.replace('helvetica', 'MicrosoftYaHei')
        graph1 = pydotplus.graph_from_dot_data(dot_data1)
        graph1.write_pdf('PPD_中.pdf')
        dot_data = export_graphviz(tree, out_file=None,
                                   class_names=target_name, filled=True, rounded=True,
                                   feature_names=feature_names)
        # 使用 pydotplus 解析 DOT 格式的字符串，并生成 PDF 文件
        dot_data=dot_data.replace('helvetica','MicrosoftYaHei')
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf('PPD_EN.pdf')

    def get_result(X_test, y_test):
        y_pred = net.predict(X_test, flag=True)
        auc = roc_auc_score(y_test, y_pred)
        cm = net.confusion_matrix(X_test,y_test, flag=True)
        # 计算指标
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        # beta = 2  # 设置beta值为2
        # f2 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        return auc,accuracy,precision,recall,cm,f1

    def get_decision_path(tree_model, features_scaled, features_original, visitNo):
        feature_names = columns
        def get_decision_path_original(tree_model, features_scaled, features_original, visitNo):
            decision_paths_original = tree_model.decision_path(features_scaled)
            decision_paths_str_original = []
            for sample, original_sample, visit_no, path in zip(features_scaled, features_original, visitNo,
                                                               decision_paths_original):
                decision_path_str = f"visit_no: {visit_no} || "
                predicted_probabilities =tree_model.predict_proba([sample])[0]
                for node in path.indices:
                    feature_idx = tree_model.tree_.feature[node]
                    threshold = tree_model.tree_.threshold[node]
                    if feature_idx != -2:
                        original_threshold = original_sample[feature_idx]
                        feature_name = feature_names[feature_idx]
                        decision_path_str += f"{feature_name} = {original_threshold:.1f} ∧ "
                    else:
                        class_probabilities = tree_model.tree_.value[node][0]
                        predicted_class_idx = class_probabilities.argmax()
                        predicted_class = ['健康', '患病'][predicted_class_idx]
                        decision_path_str += f"预测情况: {predicted_class} (良性概率: {predicted_probabilities[0]:.3f}, 患病概率: {predicted_probabilities[1]:.3f})"
                decision_paths_str_original.append(decision_path_str)
            return decision_paths_str_original
        # 决策路径2：使用标准化后的数值
        def get_decision_path_scaled(tree_model, features_scaled, visitNo):
            feature_names = columns
            decision_paths_scaled = tree_model.decision_path(features_scaled)
            decision_paths_str_scaled = []
            for sample, visit_no, path in zip(features_scaled, visitNo, decision_paths_scaled):
                decision_path_str = f"样本唯一标识: {visit_no} || "
                predicted_probabilities = tree_model.predict_proba([sample])[0]
                for node in path.indices:
                    feature_idx = tree_model.tree_.feature[node]
                    threshold = tree_model.tree_.threshold[node]
                    if feature_idx != -2:
                        feature_name = feature_names[feature_idx]
                        if sample[feature_idx] <= threshold:
                            decision_path_str += f"{feature_name} <= {threshold:.3f} ∧ "
                        else:
                            decision_path_str += f"{feature_name} > {threshold:.3f} ∧ "
                    else:
                        class_probabilities = tree_model.tree_.value[node][0]
                        predicted_class_idx = class_probabilities.argmax()
                        predicted_class = ['健康', '患病'][predicted_class_idx]
                        decision_path_str += f"预测情况: {predicted_class} (良性概率: {predicted_probabilities[0]:.3f}, 患病概率: {predicted_probabilities[1]:.3f})"

                decision_paths_str_scaled.append(decision_path_str)
            return decision_paths_str_scaled

        # 获取测试集的决策路径
        decision_paths_original = get_decision_path_original(tree_model, features_scaled, features_original, visitNo)
        decision_paths_scaled = get_decision_path_scaled(tree_model, features_scaled, visitNo)

        # 打印决策路径
        print("使用标准化前的数值的决策路径：")
        for i, path in enumerate(decision_paths_original):
            print(f"样本 {i + 1}: {path}")

        print("\n使用标准化后的数值的决策路径：")
        for i, path in enumerate(decision_paths_scaled):
            print(f"样本 {i + 1}: {path}")

    # 数据处理 产后出血PPH 48 visitNo
    data = pd.read_csv('D:\\pphdata\\320_294_1_x_48.csv', encoding='gbk')
    columns = list(data.columns[2:-5])
    # 294
    X = data[columns][:588]
    y = data['出血症'][:588]
    visitNo = data['visitNo'][:588]

    # 产后抑郁PPD 9
    # data = pd.read_csv('D:\\pphdata\\PublicDataset\\PPD\\2.csv', encoding='gbk')
    # columns = list(data.columns[1:-2])
    # X = data[columns][:750]
    # y = data['Feeling anxious'][:750]
    # visitNo = data['ID'][:750]
    # print('PPD')

    # 妊娠期糖尿病 15
    # data = pd.read_csv(
    #     'D:\\pphdata\\Publicdataset\\Gestational Diabetes Mellitus (GDM Data Set)\\Gestational Diabetic Dat Set.csv',
    #     encoding='gbk')
    # columns = list(data.columns[1:-1])
    # X = data[columns]
    # y = data['Class Label(GDM /Non GDM)']
    # visitNo = data['Case Number']


    # 术后呕吐 30
    # ID
    # data = pd.read_csv('D:\\pphdata\\PONV.csv',encoding='gbk')
    # columns = list(data.columns[2:-1])
    # X = data[columns][:606]
    # y = data['PONV'][:606]
    # visitNo = data['ID'][:606]
    # print('PONV')


    y = np.eye(2)[y]
    y = pd.DataFrame(y)

    #  反馈数据处理
    feedback_data = pd.read_csv('D:\\pphdata\\new_feedback_48.csv',encoding='gbk')
    feedback_columns = list(feedback_data.columns[2:-5])
    y1 = feedback_data['出血症'][:588]
    # y1 = np.eye(2)[y1]
    X_coords = feedback_data[feedback_columns][:588]

    X=X.values
    y=y.values
    num=42
    X_train, X_test, y_train, y_test, visitNo_train, visitNo_test = train_test_split(X, y, visitNo,
                                                                            test_size=0.3, random_state=num)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=num)
    train_coords, test_coords, y1_train, y1_test = train_test_split(X_coords, y1, test_size=0.3, random_state=num)
    print(num)
    print('数据集大小:',len(X))
    print('标记数据集大小:',len(X_coords))
    # print(y_train)
    train_coords = train_coords.values
    train_one=0
    test_one=0
    for i in range(len(y_train)):
        if y_train[i][1]==1:
            train_one+=1
    for i in range(len(y_test)):
        if y_test[i][1]==1:
            test_one+=1

    print('训练集正样本数量:',train_one)
    print('测试集正样本数量:',test_one)

    scaler = StandardScaler()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # CNN
    model_training(X_train, y_train, X_test, y_test)
    auc, accuracy, precision, recall, cm,f1 = get_result(X_test, y_test)
    print('\n')
    print('CNN:')
    print('Auc:', np.around(auc, 3))
    print('Accuracy:', np.around(accuracy, 3))
    print('Precision  :', np.around(precision, 3))
    print('Recall  :', np.around(recall, 3))
    print('F1  :', np.around(f1, 3))
    # print('CM  :', cm)


    # CNN+Tree
    model_cnnTree_training(X_train, y_train, X_test, y_test)
    auc, accuracy, precision, recall, cm,f1 = get_result(X_test, y_test)
    print('cnnTree:')
    print('Auc:', np.around(auc, 3))
    print('Accuracy:', np.around(accuracy, 3))
    print('Precision  :', np.around(precision, 3))
    print('Recall  :', np.around(recall, 3))
    print('F1  :', np.around(f1, 3))
    # print('CM  :', cm)

    # CNN+feedback
    # model_feedback_training(X_train, y_train, X_test, y_test, train_coords)
    # auc, accuracy, precision, recall, cm,f1 = get_result(X_test, y_test)
    # print('Feedback:')
    # print('Auc:', np.around(auc, 3))
    # print('Accuracy:', np.around(accuracy, 3))
    # print('Precision  :', np.around(precision, 3))
    # print('Recall  :', np.around(recall, 3))
    # print('F1  :', np.around(f1, 3))
    # print('CM  :', cm)

    # CNN+Tree+feedback
    # model_feedbackTree_training(X_train, y_train, X_test, y_test, train_coords)
    # auc, accuracy, precision, recall, cm,f1 = get_result(X_test, y_test)
    #
    # print('FeedbackTree:')
    # print('Auc:', np.around(auc, 3))
    # print('Accuracy:', np.around(accuracy, 3))
    # print('Precision  :', np.around(precision, 3))
    # print('Recall  :', np.around(recall, 3))
    # print('F1  :', np.around(f1, 3))
    # print('CM  :', cm)

    end_time = time.time()

    # 计算时间差
    elapsed_time = end_time - start_time

    # 打印时间差
    print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print(f"Time elapsed: {elapsed_time/60.0:.2f} minutes.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strength', '-st',
        type=str,
        default=300,
        help='''Tree regularization intensity '''
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='RL',
        help='''Categorical variables representation:
                binary for binary encoding, embedded for word embedding. '''
    )

    parser.add_argument(
        '--optimizer', '-o',
        type=str,
        default='Adam',
        help='Optimizer.'
    #     Adagrad, Adam, Ftrl, Momentum, RMSProp, SGD
    )

    parser.add_argument(
        '--learning_rate', '-l',
        type=float,
        default=0.05,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='The number of iterations to run'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=64,
        help='The batch size'
    )

    parser.add_argument(
        '--num_cat', '-nc',
        type=int,
        default=2,
        help='The number of categories for categorical distributions. '
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='./PPH_10',
        help='''Categorical variables representation:
                binary for binary encoding, embedded for word embedding.'''
    )

    parser.add_argument(
        '--arch', '-A',
        type=str,
        default='128',
        help=''' The number of neurons in each layer of the neural net classifier. '''
    )

    parser.add_argument(
        '--num_filters', '-fi',
        type=int,
        default=256,
        help=''' The number of filters in the conv net. '''
    )

    parser.add_argument(
        '--num_features', '-fe',
        type=int,
        default=48,
        help=''' The number of features used by the agent. '''
    )

    parser.add_argument(
        '--num_samples', '-sa',
        type=int,
        default=8,
        help=''' The number of samples per example. '''
    )

    parser.add_argument(
        '--nonlinearity', '-f',
        type=str,
        default='tf.nn.relu',
        help=''' The neural net activation function. '''
    )

    parser.add_argument(
        '--frame_size', '-fr',
        type=int,
        default=60,
        help=''' The size of the cluttered MNIST image. '''
    )

    parser.add_argument(
        '--feedback_distance', '-fd',
        type=str,
        default='mse',
        help=''' The dissimilarity measure used to evaluate the feedback. '''
    )

    parser.add_argument(
        '--second_conv', '-sc',
        type=str2bool,
        default='f',
        help=''' Wheter to add a second convolutional layer. '''
    )


    parser.add_argument(
        '--dropout', '-dr',
        type=float,
        default=0.5,
        help=''' The probability of keeping a neuron active. '''
    )

    parser.add_argument(
        '--initial_tau', '-t',
        type=float,
        default=10.0,
        help=''' The initial temperature vaule for the PD model. '''
    )

    parser.add_argument(
        '--tau_decay', '-td',
        type=str2bool,
        default='true',
        help=''' Whether to decay tau or keep it constant. '''
    )

    parser.add_argument(
        '--reg', '-r',
        type=float,
        default=1,
        help=''' Trade-off between classification and regularization cost. '''
    )

    parser.add_argument(
        '--pre_train', '-p',
        type=str2bool,
        default='f',
        help=''' Whether the model should be pre trained or not. '''
    )

    parser.add_argument(
        '--number_patches', '-np',
        type=int,
        default=10,
        help=''' The number of noise patches to be added to the model. '''
    )

    # 打印当前时间

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.arch = [int(item) for item in FLAGS.arch.split(',')]
    if not os.path.exists(FLAGS.dir):
        os.makedirs(FLAGS.dir)
    else:
        pass
        # raise ValueError('This model\'s name has already been used.')
    with open(FLAGS.dir + '/config', 'w') as f:
        f.write(str(vars(FLAGS)))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



