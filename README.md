# video_prediction
深度学习预测视频序列
#开发环境：
win10+anaconda+python3.7 + tensorflow 1.14-gpu+pycharm
#硬件：
GTX 1060显卡
#程序功能说明：
- 
Create_tfrecord.py和image_to_tfrecord.py用于将视频帧序列写入为TFrecord格式。


- Lstm_ops.py是LSTM网络结构。


- Output_image.py用于将输出的tensor变量输出为gif。


- Prediction_input.py用于将tfrecord读入。


- Prediction_model.py是整个CDNA模型结构


- Prediction_train.py是用于训练的代码。


- Prediction_test.py 是用于测试的代码。
#文件夹说明：


- Grap 是存放用于预测输入的视频序列


- Grap_tfrecord是存放输入视频序列转换之后的tfrecord


- Output_test是用于存放输出的gif
#特别说明：
程序中的数据集和模型因为太大，无法上传，详细请看论文。
本程序完全用于学习，是针对原论文的复现。
