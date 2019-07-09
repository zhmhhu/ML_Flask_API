#  coding: utf-8
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, json
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pyecharts import options as opts
from pyecharts.charts import Scatter,Line

REMOTE_HOST = ""

app = Flask(__name__)


@app.route("/post", methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        data = request.get_data()
        json_data = json.loads(data.decode("utf-8"))
        xname = str(json_data.get("xname"))
        yname = str(json_data.get("yname"))
        xdata = json_data.get("xdata")
        ydata = json_data.get("ydata")

        xarray = np.asarray(xdata, dtype=float)
        yarray = np.asarray(ydata, dtype=float)


        ployreg = str(json_data.get("ployreg"))  # 是否多项式回归
        ploynum = str(json_data.get("ploynum"))  # 多项式次数

        examDict = {
            xname: xarray,
            yname: yarray
        }

        examOrderDict = OrderedDict(examDict)
        exam = pd.DataFrame(examOrderDict)

        # 从DataFrame中把标签和特征导出来
        exam_X = exam[xname]
        exam_Y = exam[yname]

        # 比例为
        X_train, X_test, Y_train, Y_test = train_test_split(exam_X, exam_Y, train_size=0.9, test_size=0.1)
        # 改变一下数组的形状
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

        # 创建一个模型
        model = LinearRegression()
        # 训练一下
        model.fit(X_train, Y_train)

        a = model.intercept_
        b = model.coef_
        a = float(a)
        b = float(b)

        print('该模型的简单线性回归方程为y = {} + {} * x'.format(a, b))

        jingdu = model.score(X_test, Y_test)
        print('模型精度为:' + str(jingdu))

        restxt = '该模型的简单线性回归方程为y = {} + {} * x'.format(a, b)+ ',模型精度为:' + str(jingdu)

        # 绘制最佳拟合曲线
        Y_train_pred = model.predict(X_train)
        Y_train_pred = np.around(Y_train_pred, decimals=2)

        # return jsonify({'status': '0', 'msg': '操作成功！', "斜率": a, "截距": b, "精度": jingdu})
        s = bar_base(xdata, ydata,X_train,Y_train_pred)
        return render_template('pyecharts.html', myechart=s.render_embed(),restxt=restxt)
        # return render_template('index1.html', date_list=date_list, result=result)
    return "failed"


def bar_base(x, y, xline, yline):

    scatter = Scatter().add_xaxis(x).add_yaxis('真实值',y).set_global_opts(
        title_opts=opts.TitleOpts(title="相关线"))

    xline = xline.reshape(-1)
    line = Line().add_xaxis(xaxis_data=xline.tolist()).add_yaxis(series_name="拟合线", y_axis=yline.tolist())

    scatter.overlap(line)
    return scatter


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5008', debug=True)  # 运行flask
