from flask import Flask,session
import pandas as pd
from io import BytesIO
import os
from flask import render_template, request,  jsonify,make_response
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import Sequential
from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
# flask 类的实例化
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# 设置允许上传的文件格式
ALLOWED_EXTENSIONS = set(['xls', 'xlsx'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s%d(t-%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s%d(t)' % (columns[j], j + 1)) for j in range(n_vars)]
        else:
            names += [('%s%d(t+%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        clean_agg = agg.dropna()
    return clean_agg
    # return agg


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的文件类型，xls、xlsx"})
        s = f.read()
        creader = pd.read_excel(BytesIO(s),header=0, index_col=0)
        creader.info()
        data_set = creader.values
        data_columns = creader.columns

        np.savetxt("record.txt", data_set, fmt="%f", delimiter=",")

        # session['data_set'] = data_set.tolist()
        session['data_columns'] = data_columns.tolist()

        result = dict()

        i = 1
        for group in data_columns:
            data = data_set[:, i-1].tolist()
            title = data_columns[i-1]
            i += 1
            result[title] = [str(i) for i in data]

        date_list = [str(i) for i in creader.index]

        return render_template('index1.html', date_list=date_list, result=result)
    return render_template('upload.html')


@app.route('/train', methods=['POST', 'GET'])  # 添加路由
def train():
    _columns = session.get('data_columns')

    _values = np.loadtxt("record.txt", dtype=float, delimiter=",")

    # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(_values)

    # 将序列数据转化为监督学习数据
    reframed = series_to_supervised(scaled, _columns, 1, 1)
    print('---before----')
    print(reframed.head())

    # 前两个时刻的输入，即时间步长为2
    # reframed = series_to_supervisesd(caled, dataset_columns, 2, 1)

    # print(reframed.head())
    # 只考虑当前时刻(t)的前一时刻（t-1）的值
    _list = []
    _length = len(_columns)
    for i in range(_length+1, 2 * _length):
        _list.append(i)
    reframed.drop(reframed.columns[_list], axis=1, inplace=True)

    # 考虑当前时刻(t)的前一时刻（t-1）以及（t-2）的值
    # reframed.drop(reframed.columns[[7,8]], axis=1, inplace=True)
    print('---after----')
    print(reframed.head())

    values = reframed.values
    rowcount = len(reframed)

    split = 0.8
    n_train_hours = int(rowcount * split)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # 监督学习结果划分,test_x.shape = (, 8)
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    print('-------------')
    print(train_x.shape[0])  # 8760
    print(train_x.shape[1])  # 8
    print(test_x.shape[0])  # 35039
    print(test_x.shape[1])  # 8

    # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
    train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    ##时间步长为2
    # train_X = train_x.reshape((train_x.shape[0], 2, 3))
    # test_X = test_x.reshape((test_x.shape[0], 2, 3))
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))  # 50个隐藏层神经元
    model.add(Dense(1))  # 一个输出层神经元
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=50, batch_size=100, validation_data=(test_X, test_y))

    '''
        对数据绘图
    '''
    '''
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    '''

    # make the prediction,为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
    yHat = model.predict(test_X)

    model.save('my_model.h5')  # 保存模型
    '''
    print('----')
    print(yHat)
    print(test_x)
    print('----')
    '''
    '''
        这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
    '''
    inv_yHat = concatenate((yHat, test_x[:, 1:]), axis=1)  # 数组拼接
    # inv_yHat = concatenate((yHat, test_x[:, 1:3]), axis=1)   # 数组拼接
    inv_yHat = scaler.inverse_transform(inv_yHat)
    inv_yHat = inv_yHat[:, 0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
    # inv_y = concatenate((test_y, test_x[:, 1:3]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)  # 将标准化的数据转化为原来的范围
    inv_y = inv_y[:, 0]

    plt.plot(inv_yHat, 'g-')
    plt.plot(inv_y, 'r-')
    plt.legend(['predict', 'true'])
    # plt.show()

    rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
    print('Test RMSE: %.3f' % rmse)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    response = make_response(img.getvalue())
    response.headers['Content-Type'] = 'image/png'
    img.close()
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True) # 运行flask
