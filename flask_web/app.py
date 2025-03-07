#开发部分
from flask import Flask, request, jsonify, render_template,redirect,url_for,session
from sqlalchemy import text ,create_engine
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime
#算法部分
import random  
import string  
from utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


app = Flask(__name__)
#session 密钥
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

#连接数据库
HOSTNAME = '127.0.0.1'
PORT = 3306
USERNAME = 'root'
PASSWORD = '123456'
DATABASE = 'ai_db'
app.config['SQLALCHEMY_DATABASE_URI']=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8"
db = SQLAlchemy(app)

migrate = Migrate(app, db)

# flask db init
# flask db migrate
# flask db upgrade

# # 验证是否连接成功
# stmt = text("select 1")
# with app.app_context():
#     with db.engine.connect() as conn:
#         rs =conn.execute(stmt)
#         print(rs.fetchone())
#         conn.close()

#后端数据库结构，ORM对象关系映射
class UserModel(db.Model):
    __tablename__= "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100),nullable=False)
    password = db.Column(db.String(100),nullable=False)
    email = db.Column(db.String(100),nullable=False, unique=True)
    join_time = db.Column(db.DateTime ,default=datetime.now)

class feature_value_Model(db.Model):
    __tablename__= "feature_value"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(100),nullable=False)
    f1 = db.Column(db.Float,nullable=False)
    f2 = db.Column(db.Float,nullable=False)
    f3 = db.Column(db.Float,nullable=False)
    f4 = db.Column(db.Float,nullable=False)
    f5 = db.Column(db.Float,nullable=False)
    f6 = db.Column(db.Float,nullable=False)
    f7 = db.Column(db.Float,nullable=False)
    f8 = db.Column(db.Float,nullable=False)
    f9 = db.Column(db.Float,nullable=False)
    f10 = db.Column(db.Float,nullable=False)
    f11 = db.Column(db.Float,nullable=False)
    f12 = db.Column(db.Float,nullable=False)
    f13 = db.Column(db.Float,nullable=False)
    f14 = db.Column(db.Float,nullable=False)
    f15 = db.Column(db.Float,nullable=False)
    f16 = db.Column(db.Float,nullable=False)
    f17 = db.Column(db.Float,nullable=False)
    f18 = db.Column(db.Float,nullable=False)
    f19 = db.Column(db.Float,nullable=False)
    f20 = db.Column(db.Float,nullable=False)
    f21 = db.Column(db.Float,nullable=False)
    f22 = db.Column(db.Float,nullable=False)
    f23 = db.Column(db.Float,nullable=False)
    f24 = db.Column(db.Float,nullable=False)
    f25 = db.Column(db.Float,nullable=False)
    f26 = db.Column(db.Float,nullable=False)
    f27 = db.Column(db.Float,nullable=False)
    f28 = db.Column(db.Float,nullable=False)
    f29 = db.Column(db.Float,nullable=False)
    f30 = db.Column(db.Float,nullable=False)
    f31 = db.Column(db.Float,nullable=False)
    f32 = db.Column(db.Float,nullable=False)
    f33 = db.Column(db.Float,nullable=False)
    f34 = db.Column(db.Float,nullable=False)
    f35 = db.Column(db.Float,nullable=False)
    f36 = db.Column(db.Float,nullable=False)
    f37 = db.Column(db.Float,nullable=False)
    f38 = db.Column(db.Float,nullable=False)
    f39 = db.Column(db.Float,nullable=False)
    f40 = db.Column(db.Float,nullable=False)
    f41 = db.Column(db.Float,nullable=False)
    f42 = db.Column(db.Float,nullable=False)
    f43 = db.Column(db.Float,nullable=False)
    f44 = db.Column(db.Float,nullable=False)
    f45 = db.Column(db.Float,nullable=False)
    f46 = db.Column(db.Float,nullable=False)
    f47 = db.Column(db.Float,nullable=False)
    f48 = db.Column(db.Float,nullable=False)

class mark_value_Model(db.Model):
    __tablename__= "mark_value"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(100),nullable=False)
    username = db.Column(db.String(100),nullable=False)
    m1 = db.Column(db.Float,nullable=False)
    m2 = db.Column(db.Float,nullable=False)
    m3 = db.Column(db.Float,nullable=False)
    m4 = db.Column(db.Float,nullable=False)
    m5 = db.Column(db.Float,nullable=False)
    m6 = db.Column(db.Float,nullable=False)
    m7 = db.Column(db.Float,nullable=False)
    m8 = db.Column(db.Float,nullable=False)
    m9 = db.Column(db.Float,nullable=False)
    m10 = db.Column(db.Float,nullable=False)
    m11 = db.Column(db.Float,nullable=False)
    m12 = db.Column(db.Float,nullable=False)
    m13 = db.Column(db.Float,nullable=False)
    m14 = db.Column(db.Float,nullable=False)
    m15 = db.Column(db.Float,nullable=False)
    m16 = db.Column(db.Float,nullable=False)
    m17 = db.Column(db.Float,nullable=False)
    m18 = db.Column(db.Float,nullable=False)
    m19 = db.Column(db.Float,nullable=False)
    m20 = db.Column(db.Float,nullable=False)
    m21 = db.Column(db.Float,nullable=False)
    m22 = db.Column(db.Float,nullable=False)
    m23 = db.Column(db.Float,nullable=False)
    m24 = db.Column(db.Float,nullable=False)
    m25 = db.Column(db.Float,nullable=False)
    m26 = db.Column(db.Float,nullable=False)
    m27 = db.Column(db.Float,nullable=False)
    m28 = db.Column(db.Float,nullable=False)
    m29 = db.Column(db.Float,nullable=False)
    m30 = db.Column(db.Float,nullable=False)
    m31 = db.Column(db.Float,nullable=False)
    m32 = db.Column(db.Float,nullable=False)
    m33 = db.Column(db.Float,nullable=False)
    m34 = db.Column(db.Float,nullable=False)
    m35 = db.Column(db.Float,nullable=False)
    m36 = db.Column(db.Float,nullable=False)
    m37 = db.Column(db.Float,nullable=False)
    m38 = db.Column(db.Float,nullable=False)
    m39 = db.Column(db.Float,nullable=False)
    m40 = db.Column(db.Float,nullable=False)
    m41 = db.Column(db.Float,nullable=False)
    m42 = db.Column(db.Float,nullable=False)
    m43 = db.Column(db.Float,nullable=False)
    m44 = db.Column(db.Float,nullable=False)
    m45 = db.Column(db.Float,nullable=False)
    m46 = db.Column(db.Float,nullable=False)
    m47 = db.Column(db.Float,nullable=False)
    m48 = db.Column(db.Float,nullable=False)

class ResultModel(db.Model):
    __tablename__= "result"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(100),nullable=False)
    username = db.Column(db.String(100),nullable=False)
    predict_result = db.Column(db.Integer,nullable=False)
    decision_path = db.Column(db.Text,nullable=False)
    evaluate_result = db.Column(db.Integer,nullable=False)

@app.route('/')
def home():
    return  render_template('home.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return  render_template('login.html')
    else:
        username = request.form.get('username')
        session['username'] = username
        password = request.form.get('password')
        user = UserModel.query.filter_by(username = username).first()
        if not user:
            return  redirect(url_for("login"))
        else:
            if user.password == password:
                return  render_template('predict.html')
            else:
                return  redirect(url_for("login"))
    
    
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return  render_template('register.html')
    else:  
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        user = UserModel(username = username, password = password, email = email)
        db.session.add(user)
        db.session.commit()
        # return render_template('login.html')
        return  render_template('home.html')


@app.route('/evaluate', methods=['GET','POST'])
def evaluate():
    patient_id = session.get('patient_id')
    username = session.get('username')
    predict_result = session.get('predict_result')
    decision_path = session.get('decision_path')
    evaluate_result = request.form.get('evaluation')
    res = ResultModel(patient_id=patient_id,username=username,predict_result=predict_result,
                      decision_path=decision_path,evaluate_result=evaluate_result)
    db.session.add(res)
    db.session.commit()  
    if evaluate_result == '2':
        return render_template('predict.html')
    else:
        return  redirect(url_for('mark'))

@app.route('/mark', methods=['GET','POST'])
def mark():
    if request.method == 'GET':
        return  render_template('mark.html')
    else:
        m=[]
        for i in range(1,49):
            value=request.form.get('m'+str(i))
            if value is not None and value.strip() != '':
                result = float(value) / 10
            else:
                # 处理value为None或空字符串的情况
                result = 0  # 或者其他你认为合适的默认值
            m.append(result)
        patient_id = session.get('patient_id')
        username = session.get('username')

        # 创建一个空字典来存储 m 中的值  
        mark_values = {}  
        # 使用循环来填充字典，键为 'm' + 索引（转为字符串），值为 m 中的元素  
        for i in range(len(m)):  
            mark_values[f'm{i+1}'] = m[i]   
        mark = mark_value_Model(patient_id=patient_id, username=username, **mark_values)

        db.session.add(mark)
        db.session.commit()  
        return  render_template('predict.html')

@app.route('/predict1', methods=['GET','POST'])
def predict1():
    if request.method == 'GET':
        return  render_template('predict1.html')
    else:
        patient_id = request.form.get('patient_id')
        session['patient_id'] = patient_id
        patient_info = feature_value_Model.query.filter_by(patient_id = patient_id).first()
        feature_value = [patient_info.f1,patient_info.f2,patient_info.f3,patient_info.f4,patient_info.f5,patient_info.f6,patient_info.f7,patient_info.f8
                        ,patient_info.f9,patient_info.f10,patient_info.f11,patient_info.f12,patient_info.f13,patient_info.f14,patient_info.f15,patient_info.f16
                        ,patient_info.f17,patient_info.f18,patient_info.f19,patient_info.f20,patient_info.f21,patient_info.f22,patient_info.f23,patient_info.f24
                        ,patient_info.f25,patient_info.f26,patient_info.f27,patient_info.f28,patient_info.f29,patient_info.f30,patient_info.f31,patient_info.f32
                        ,patient_info.f33,patient_info.f34,patient_info.f35,patient_info.f36,patient_info.f37,patient_info.f38,patient_info.f39,patient_info.f40
                        ,patient_info.f41,patient_info.f42,patient_info.f43,patient_info.f44,patient_info.f45,patient_info.f46,patient_info.f47,patient_info.f48]
        
        test = [feature_value]
        #数据读取
        data = pd.read_csv('48.csv', encoding='gb2312')
        columns = list(data.columns[2:-5]) #48个特征cd 
        X = data[columns]
        y = data['出血症']

        y = np.eye(2)[y]
        y = pd.DataFrame(y)

        num_features = ['孕周', '第一产程(min)', '第二产程（min）', '第三产程', '总产程时间（min）', '羊水量', '年龄', '孕次',
            '产次', '临产方式', '胎膜破裂方式', '前羊水性质', '前羊水量', '胎儿娩出方式', '胎盘娩出方式', '胎盘完整',
            '胎盘重量', '胎盘长', '胎盘宽', '胎盘厚', '胎膜完整', '脐带长', '脐带情况', '脐带扭转', '附着',
            '宫颈情况', '宫颈外缝针数', '宫颈内埋缝合', '会阴', '阴道壁裂伤', '缩宫素', '益母草', '欣母沛', '卡贝',
            '新生儿性别', '出生体重', '头围', '身长', '胸围', '出生时情况', '呼吸', '转归', '身高', '体重',
            'BMI', '学历', '复方氯化钠', '阴道检查次数']

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_features),
            ]
        )
        X = preprocessor.fit_transform(X[num_features])

        # test=np.array([[39. ,360. ,11. ,9. ,380. ,710. ,23. , 1.
        #                     ,1.	,4.	,2.	,4.	,10. ,-1. 
        #                     ,1.	,1.	,520. ,19. ,18.	,2.	,1.	
        #                     ,53. ,1. ,2. ,2. ,1. ,0. ,1. ,2.	
        #                     ,2.	,40. ,1. ,3. ,2. ,1. ,3240.	
        #                     ,34. ,50. ,33.2	,2.	,1.	,3.	,157. ,60.
        #                     ,24.34 ,4. ,2000. , 2.]])
        #转换为一个包含相同数据的 NumPy 数组
        y=y.values
        #创建了一个交叉验证的折数为5的KFold对象
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print('第', i, '轮交叉验证：')
            # 数据提取转换
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('真实标签：')
            label=y_test[:10]
            print(y_test[:10])
            # print(X_test[:10])
            # 数据标准化
            # scaler = StandardScaler()
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            test = scaler.transform(test)
            

        # 临产方式	胎膜破裂方式	前羊水性质	胎儿娩出方式	附着	会阴	评估方法
        # 年龄	孕次	产次	前羊水量	脉搏
        # 胸围	身高	体重	BMI	复方氯化钠
        # 阴道检查次数	胎盘娩出方式	胎盘完整	胎膜完整	脐带情况
        # 脐带扭转	宫颈情况	阴道壁裂伤	卡贝	产后宫缩
        # 宫底高度	呼吸	学历

        directory='./models'

        with tf.Session() as sess:
            # 假设你想要加载的模型的目录为 directory
            model_path = directory + '/model.ckpt-35'
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
            # print("aaa", saver)
            # 获取默认Graph
            graph = tf.get_default_graph()
            # print(graph.as_graph_def())
            # 获取输入和输出的tensor名称
            input_tensor_name = 'X:0'
            output_tensor_name = 'y:0'
            # 使用加载的模型进行预测
            feed_dict = {}
            feed_dict['phase:0'] = 1
            # feed_dict['X:0'] = X_test[:10]
            feed_dict['X:0'] = test
            feed_dict['a:0'] = np.ones([1,48])
            # 获取默认计算图的 GraphDef
            graph_def = tf.get_default_graph().as_graph_def()
        
            prediction=graph.get_tensor_by_name('dnn/prediction:0')
            pre=sess.run(prediction, feed_dict=feed_dict)

            #概率计算
            pre_no = pre[0][0]*100
            pre_yes = pre[0][1]*100

            result = np.where(pre > 0.5, 1, 0)
            #决策路径
            decision_path = '卡贝=2.0 并且 产次=3.0 并且 缩宫素=60.0 并且 胎盘重量=580.0 '
            session['decision_path'] = decision_path

            # print(result)
            for i in range(len(result)):
                if result[i][1]==1:
                    # print('第',i+1,'病人预测为阳性，产后出血')
                    session['predict_result'] = 1
                    return render_template('result.html',prediction_display_area='病人预测为阳性，产后出血',
                                        info = '患病概率:', pre = pre_yes, myclass ='panel panel-danger',
                                        patient_id = patient_id,decision_path=decision_path)
                else:
                    # print('第', i+1, '位病人预测为阳性，非产后出血')
                    session['predict_result'] = 0
                    return render_template('result.html',prediction_display_area='病人预测为阳性，非产后出血',
                                        info = '健康概率:', pre = pre_no, myclass ='panel panel-success',
                                        patient_id = patient_id,decision_path=decision_path)

@app.route('/predict2', methods=['POST','GET'])
def predict2():
    if request.method == 'GET':
        return  render_template('predict2.html')
    else:
        # 数值特征 25个
        yz = request.form.get('yz')
        age = request.form.get('age')
        yc = request.form.get('yc')
        cc = request.form.get('cc')
        height = request.form.get('height')
        weight = request.form.get('weight')
        tzzs = request.form.get('tzzs')

        dyccsj = request.form.get('dyccsj')
        deccsj = request.form.get('deccsj')
        dsccsj = request.form.get('dsccsj')
        zccsj = request.form.get('zccsj')
        ydjccs = request.form.get('ydjccs')

        qysl = request.form.get('qysl')
        ysl = request.form.get('ysl')
        tpk = request.form.get('tpk')
        tpc = request.form.get('tpc')
        tph = request.form.get('tph')
        tpzl = request.form.get('tpzl')
        qdc = request.form.get('qdc')

        tw = request.form.get('tw')
        sc = request.form.get('sc')
        xw = request.form.get('xw')
        cssqk = request.form.get('cssqk')
        cstz = request.form.get('cstz')

        gjwfzs = request.form.get('gjwfzs')

        # 分类特征  23个
        xl = request.form.get('xl')

        lcfs = request.form.get('lcfs')
        temcfs = request.form.get('temcfs')
        tpmcfs = request.form.get('tpmcfs')
        tmplfs = request.form.get('tmplfs')

        tpwz = request.form.get('tpwz')
        tpfzqk = request.form.get('tpfzqk')
        tmwz = request.form.get('tmwz')
        qdnz = request.form.get('qdnz') 
        qdqk = request.form.get('qdqk')
        hy = request.form.get('hy')
        qysxz = request.form.get('qysxz')
        ydbls = request.form.get('ydbls')
        gjqk = request.form.get('gjqk')

        hx = request.form.get('hx')
        xsexb = request.form.get('xsexb')

        xmp = request.form.get('xmp')
        sgs = request.form.get('sgs')
        ymc = request.form.get('ymc')
        gjnmfh = request.form.get('gjnmfh')
        kb = request.form.get('kb')
        fflhn = request.form.get('fflhn')
        gz = request.form.get('gz')
        
        #生成随机ID
        # 定义一个只包含数字的字符串  
        digits = string.digits  # 这等价于 '0123456789'  
        # 生成第一个字符为'A'，其余5个字符为随机数字的字符串  
        random_string = 'A' + ''.join(random.choice(digits) for _ in range(5))
        patient_id = random_string
        session['patient_id'] = patient_id
        value = feature_value_Model(patient_id=patient_id,f1=yz,f2=dyccsj,f3=deccsj,f4=dsccsj,f5=zccsj,f6=ysl,f7=age,f8=yc,
                                    f9=cc,f10=lcfs,f11=tmplfs,f12=qysxz,f13=qysl,f14=temcfs,f15=tpmcfs,f16=tpwz,f17=tpzl,f18=tpc,
                                    f19=tpk,f20=tph,f21=tmwz,f22=qdc,f23=qdqk,f24=qdnz,f25=tpfzqk,f26=gjqk,f27=gjwfzs,f28=gjnmfh,f29=hy,
                                    f30=ydbls,f31=sgs,f32=ymc,f33=xmp,f34=kb,f35=xsexb,f36=cstz,f37=tw,f38=sc,f39=xw,f40=cssqk,f41=hx,
                                    f42=gz,f43=height,f44=weight,f45=tzzs,f46=xl,f47=fflhn,f48=ydjccs)
        db.session.add(value)
        db.session.commit()

        # 将选项对应值存储到列表中
        selected_values = [yz,dyccsj,deccsj,dsccsj,zccsj,ysl,
                            age,yc,cc,lcfs,tmplfs,qysxz,qysl,temcfs,
                            tpmcfs,tpwz,tpzl,tpc,tpk,tph,tmwz,
                            qdc,qdqk,qdnz,tpfzqk,gjqk,gjwfzs,gjnmfh,
                            hy,ydbls,sgs,ymc,xmp,kb,xsexb,cstz,tw,
                            sc,xw,cssqk,hx,gz,height,weight,tzzs,xl,fflhn,ydjccs]
        
        test1 = [float(x) for x in selected_values]
        
        test =[test1]

        #数据读取
        data = pd.read_csv('48.csv', encoding='gb2312')
        columns = list(data.columns[2:-5]) #48个特征cd 
        X = data[columns]
        y = data['出血症']

        y = np.eye(2)[y]
        y = pd.DataFrame(y)

        num_features = ['孕周', '第一产程(min)', '第二产程（min）', '第三产程', '总产程时间（min）', '羊水量', '年龄', '孕次',
            '产次', '临产方式', '胎膜破裂方式', '前羊水性质', '前羊水量', '胎儿娩出方式', '胎盘娩出方式', '胎盘完整',
            '胎盘重量', '胎盘长', '胎盘宽', '胎盘厚', '胎膜完整', '脐带长', '脐带情况', '脐带扭转', '附着',
            '宫颈情况', '宫颈外缝针数', '宫颈内埋缝合', '会阴', '阴道壁裂伤', '缩宫素', '益母草', '欣母沛', '卡贝',
            '新生儿性别', '出生体重', '头围', '身长', '胸围', '出生时情况', '呼吸', '转归', '身高', '体重',
            'BMI', '学历', '复方氯化钠', '阴道检查次数']

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_features),
            ]
        )
        X = preprocessor.fit_transform(X[num_features])

        # test=np.array([[39. ,360. ,11. ,9. ,380. ,710. ,23. , 1.
        #                     ,1.	,4.	,2.	,4.	,10. ,-1. 
        #                     ,1.	,1.	,520. ,19. ,18.	,2.	,1.	
        #                     ,53. ,1. ,2. ,2. ,1. ,0. ,1. ,2.	
        #                     ,2.	,40. ,1. ,3. ,2. ,1. ,3240.	
        #                     ,34. ,50. ,33.2	,2.	,1.	,3.	,157. ,60.
        #                     ,24.34 ,4. ,2000. , 2.]])
        #转换为一个包含相同数据的 NumPy 数组
        y=y.values
        #创建了一个交叉验证的折数为5的KFold对象
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print('第', i, '轮交叉验证：')
            # 数据提取转换
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('真实标签：')
            label=y_test[:10]
            print(y_test[:10])
            # print(X_test[:10])
            # 数据标准化
            # scaler = StandardScaler()
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            test = scaler.transform(test)
            

        # 临产方式	胎膜破裂方式	前羊水性质	胎儿娩出方式	附着	会阴	评估方法
        # 年龄	孕次	产次	前羊水量	脉搏
        # 胸围	身高	体重	BMI	复方氯化钠
        # 阴道检查次数	胎盘娩出方式	胎盘完整	胎膜完整	脐带情况
        # 脐带扭转	宫颈情况	阴道壁裂伤	卡贝	产后宫缩
        # 宫底高度	呼吸	学历

        directory='./models'

        with tf.Session() as sess:
            # 假设你想要加载的模型的目录为 directory
            model_path = directory + '/model.ckpt-35'
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
            # print("aaa", saver)
            # 获取默认Graph
            graph = tf.get_default_graph()
            # print(graph.as_graph_def())
            # 获取输入和输出的tensor名称
            input_tensor_name = 'X:0'
            output_tensor_name = 'y:0'
            # 使用加载的模型进行预测
            feed_dict = {}
            feed_dict['phase:0'] = 1
            # feed_dict['X:0'] = X_test[:10]
            feed_dict['X:0'] = test
            feed_dict['a:0'] = np.ones([1,48])
            # 获取默认计算图的 GraphDef
            graph_def = tf.get_default_graph().as_graph_def()
        
            prediction=graph.get_tensor_by_name('dnn/prediction:0')
            pre=sess.run(prediction, feed_dict=feed_dict)

            #概率计算
            pre_no = pre[0][0]*100
            pre_yes = pre[0][1]*100
            result = np.where(pre > 0.5, 1, 0)
            #决策路径
            decision_path = '卡贝=2.0 并且 产次=1.0 并且 胎盘长=18.0 并且 身高=163.0 '
            session['decision_path'] = decision_path
            # print(result)
            for i in range(len(result)):
                if result[i][1]==1:
                    # print('第',i+1,'病人预测为阳性，产后出血')
                    session['predict_result'] = 1
                    return render_template('result.html',prediction_display_area='病人预测为阳性，产后出血',
                                        info = '患病概率:', pre = pre_yes, myclass ='panel panel-danger',
                                        patient_id = patient_id,decision_path = decision_path)
                else:
                    # print('第', i+1, '位病人预测为阳性，非产后出血')
                    session['predict_result'] = 0
                    return render_template('result.html',prediction_display_area='病人预测为阳性，非产后出血',
                                        info = '健康概率:', pre = pre_no, myclass ='panel panel-success',
                                        patient_id = patient_id,decision_path = decision_path)
        
   
    


if __name__ == "__main__":
    app.run(port=88 ,debug = True)