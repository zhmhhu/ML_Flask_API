# 基于 flask 发布的深度学习 API

本项目使用 python（python3.6） 语言, 使用的类库包括但不限于
   - kreas
   - panada
   - numpy
   - tensorflow
   - flask
   
   请按实际需要安装，并保持最新版本。
   
## RESNET
   
基于 RESNET 算法的目标检测，运行`RESNET/resnetOnline.py`,在浏览器中输入 localhost:5000，然后上传需要检测的图片，即可得到目标检测的结果。
   
## LSTM

基于 LSTM 算法的数据预测，在浏览器中输入 localhost:5000/upload,上传 xls、xlsx 格式的电子表格。表格中，第一列为索引，可以是日期或者是自增长的 id 编号，第二列为因变量（也就是将来需要预测的值），第三到最后一列为自变量。上传完成之后，可以看到数据曲线图。

之后，输入 localhost:5000/upload，后台将进行模型训练，数据中的前 80 作为训练值，后 20 作为测试值。稍等片刻，系统将返回关于测试值和模型预测值的图形，可以查看模型预测是否准确。