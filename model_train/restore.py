from utils import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
# 数据集路径
data = pd.read_csv('D:\\pphdata\\320_294_1_x_48.csv', encoding='gbk')
columns = list(data.columns[2:-5])
# 294
X = data[columns][:588]
y = data['出血症'][:588]
y = np.eye(2)[y]
y = pd.DataFrame(y)
visitNo = data['visitNo'][:588]

# 随机分配训练测试
X_train, X_test, y_train, y_test, visitNo_train, visitNo_test = \
    train_test_split(X, y, visitNo, test_size=0.3, random_state=42)
# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_test y_test visitNo_test 一样对应，visitNo_test可做检查是否正确，与数据集对照
# # pph预测模型路径
directory='C:\\Users\\LENOVO\\PPH\\Myproject\\newmy\\PPH_10'
with tf.Session() as sess:
    # 模型加载
    model_path = directory + '\\model.ckpt-63'
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    graph = tf.get_default_graph()
    input_tensor_name = 'X:0'
    output_tensor_name = 'y:0'
    # 使用加载的模型进行预测
    feed_dict = {}
    feed_dict['phase:0'] = 1
    # 可以检测一个或者多个病人,数字保持一致
    # 预测10个病人X_test[:10], np.ones([10,48])
    # 预测10个病人X_test[:1], np.ones([1, 48])
    # X_test[:1]为需要预测的病人，经过了标准化后的
    feed_dict['X:0'] = X_test[:1]
    feed_dict['a:0'] = np.ones([1,48])
    # 获取默认计算图的 GraphDef
    graph_def = tf.get_default_graph().as_graph_def()
    prediction=graph.get_tensor_by_name('dnn/prediction:0')
    pre=sess.run(prediction, feed_dict=feed_dict)
    print('预测结果：')
    print('阴性概率            阳性概率')
    print(pre)
    result = np.where(pre > 0.5, 1, 0)
    print(result)
    for i in range(len(result)):
        if result[i][0]==0:
            print('第',i+1,'位病人预测为阳性，产后出血')
        else:
            print('第', i+1, '位病人预测为阳性，非产后出血')
# 检查是否预测正确
# print(X_test[:1])
print(y_test[:1])
print(visitNo_test[:1])

