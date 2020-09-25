
import numpy as np
import tensorflow as tf
import moviepy.editor as mpy


def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip(list(npy), fps=10)
    clip.write_gif(filename)

def output_gif(images,name):
    images = np.array(images)
    images = images * 255.0
    images = tf.convert_to_tensor(images, dtype=tf.uint8)
    images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
    images = [tf.squeeze(act) for act in images]
    images = tf.convert_to_tensor(images)
    images = images.eval()

    for i in range(32):
        video = images[i]
        npy_to_gif(video, 'output_test/'+name + str(i) + '.gif')