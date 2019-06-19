# -*- coding:utf-8 -*-
# 上传电子表格，解析其数据并在网页绘图显示
from flask import Flask
import pandas as pd
from io import BytesIO
from flask import render_template, request,  jsonify
# flask 类的实例化
app = Flask(__name__)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['xls', 'xlsx'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的文件类型，xls、xlsx"})

        s = f.read()
        dfile = BytesIO(s)
        creader = pd.read_excel(dfile,header=0, index_col=0)  # dfile
        creader.info()
        # loads the 'pollution.csv' and treat each column as a separate subplot
        values = creader.values
        groups = creader.columns

        result = dict()

        i = 1
        for group in groups:
            data = values[:, i-1].tolist()
            title = creader.columns[i-1]
            i += 1
            result[title] = [str(i) for i in data]

        date_list = [str(i) for i in creader.index]
        # jsonData = json.dumps(result)
        # print(jsonData)

        return render_template('index1.html', date_list=date_list, result=result)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run() # 运行flask
