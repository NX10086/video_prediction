import os
import tensorflow as tf
from PIL import Image
from create_tfrecord import save_tf_record
# 图片路径
cwd = 'grap/test'
# 文件路径
filepath = 'grap_tfrecord/'

# 第几个图片
num=0
# tfrecords格式文件名
ORIGINAL_WIDTH = 160
ORIGINAL_HEIGHT = 120
COLOR_CHAN = 3
# 类别和路径
class_path = cwd + '/'
images_list = []
images = []
for img_name in os.listdir(class_path):

    print('路径', class_path)

    print('第几个图片：', num)

    print('图片名：', img_name)

    img_path = class_path + img_name  # 每一个图片的地址

    img = tf.read_file(filename=img_path)  # 默认读取格式为uint8

    print("img 的类型是", type(img))

    img = tf.image.decode_jpeg(img, channels=0)  # channels 为1得到的是灰度图，为0则按照图片格式来读
    img.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    image = tf.reshape(img, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
    images.append(image)
    print(img)
    if num % 10 == 9:
        images = tf.concat(axis=0, values=images)
        images = tf.reshape(images, [10, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        session = tf.Session()
        images=session.run(images)
        images_list.append(images)
        images=[]
    num = num + 1

print(images_list)

save_tf_record('grap_tfrecord','file',images_list)

print("已将视频序列写入为TFrecord")