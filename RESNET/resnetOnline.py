
# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from tensorflow import keras
import tensorflow as tf

from flask import render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import time
from datetime import timedelta

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

model = None

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")
    global graph
    graph = tf.get_default_graph()


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        data = {"success": False}

        # user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        image = Image.open(upload_path)
        image.save('static/uploads/test.jpg')  # 文件重命名
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        # classify the input image and then initialize the list
        # of predictions to return to the client
        with graph.as_default():
            preds = model.predict(image)
            results = keras.applications.resnet50.decode_predictions(preds)
        data["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        # indicate that the request was a success
        data["success"] = True

        return render_template('upload_ok.html', userinput=data, val1=time.time())

    return render_template('upload.html')


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()