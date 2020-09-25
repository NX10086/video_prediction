# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for training the prediction model."""
    #训练和预测模型代码
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input import build_tfrecord_input
from prediction_model import construct_model

# How often to record tensorboard summaries.
#记录张量板摘要的频率
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
#通过验证模型运行批处理的频率
VAL_INTERVAL = 200


# How often to save a model checkpoint
#保存模型检查点的频率
SAVE_INTERVAL = 2000

# tf record data location:
#数据集位置
DATA_DIR = 'push/push_train'

# local output directory
#输出目录
OUT_DIR = 'tmp/data'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.') #包含数据的目录
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.') #模型检查点目录
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.') #总结目录
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')#迭代次数
flags.DEFINE_string('pretrained_model', 'tmp/data/model6002',
                    'filepath of a pretrained model to initialize from.')#初始化模型的路径，如果为空则为随机

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')#序列长度
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')#预测前的帧数
flags.DEFINE_integer('use_state', 0,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')#转换数量和相应的掩码
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'#用于计划采样的超参数
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'#用于训练集的百分比
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')#训练批量
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')#学习率


## Helper functions
#辅助函数 PSNR度量
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.
    #基于最大信号功率与噪声功率的图像质量度量。
  Args:
    true: the ground truth image.地面真实图像
    pred: the predicted image. 预测图象
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)

#MSE为均方误差
def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.
  张量（预测和真实）之间的L2距离

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):

  def __init__(self,
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None,
               prefix=None):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    if prefix is None:
        prefix = tf.placeholder(tf.string, [])
    self.prefix = prefix
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

    # Split into timesteps.
    #分为多个时间步
    actions = tf.split(axis=1, num_or_size_splits=int(actions.get_shape()[1]), value=actions)
    actions = [tf.squeeze(act) for act in actions]
    states = tf.split(axis=1, num_or_size_splits=int(states.get_shape()[1]), value=states)
    states = [tf.squeeze(st) for st in states]
    images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
    images = [tf.squeeze(img) for img in images]

    if reuse_scope is None:
      gen_images, gen_states = construct_model(
          images,
          actions,
          states,
          iter_num=self.iter_num,
          k=FLAGS.schedsamp_k,
          use_state=FLAGS.use_state,
          num_masks=FLAGS.num_masks,
          cdna=FLAGS.model == 'CDNA',
          dna=FLAGS.model == 'DNA',
          stp=FLAGS.model == 'STP',
          context_frames=FLAGS.context_frames)
    else:  # If it's a validation or test model. 如果是一个验证或者测试模型

      with tf.variable_scope(reuse_scope, reuse=True):
        gen_images, gen_states = construct_model(
            images,
            actions,
            states,
            iter_num=self.iter_num,
            k=FLAGS.schedsamp_k,
            use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks,
            cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA',
            stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)

    # L2 loss, PSNR for eval. L2损失PSNR为评估值
    loss, psnr_all = 0.0, 0.0
    for i, x, gx in zip(
        range(len(gen_images)), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      recon_cost = mean_squared_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      summaries.append(
          tf.summary.scalar(prefix + '_recon_cost' + str(i), recon_cost))
      summaries.append(tf.summary.scalar(prefix + '_psnr' + str(i), psnr_i))
      loss += recon_cost

    for i, state, gen_state in zip(
        range(len(gen_states)), states[FLAGS.context_frames:],
        gen_states[FLAGS.context_frames - 1:]):
      state_cost = mean_squared_error(state, gen_state) * 1e-4
      summaries.append(
          tf.summary.scalar(prefix + '_state_cost' + str(i), state_cost))
      loss += state_cost
    summaries.append(tf.summary.scalar(prefix + '_psnr_all', psnr_all))
    self.psnr_all = psnr_all

    self.loss = loss = loss / np.float32(len(images) - FLAGS.context_frames)

    summaries.append(tf.summary.scalar(prefix + '_loss', loss))

    self.lr = tf.placeholder_with_default(FLAGS.learning_rate, ())

    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    self.summ_op = tf.summary.merge(summaries)


def main(unused_argv):

  print('Constructing models and inputs.')
  #训练集
  with tf.variable_scope('model', reuse=None) as training_scope:
    images, actions, states = build_tfrecord_input(training=True,vil=False)
    model = Model(images, actions, states, FLAGS.sequence_length,
                  prefix='train')
  #验证集
  with tf.variable_scope('val_model', reuse=None):
    val_images, val_actions, val_states = build_tfrecord_input(training=False,vil=True)#创建input
    val_model = Model(val_images, val_actions, val_states,
                      FLAGS.sequence_length, training_scope, prefix='val') #重用变量空间

  print('Constructing saver.')
  # Make saver.保存
  saver = tf.train.Saver(
      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)

  # Make training session. 培训
  sess = tf.InteractiveSession() #可加入计算图
  sess.run(tf.global_variables_initializer())

  summary_writer = tf.summary.FileWriter(
      FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

  if FLAGS.pretrained_model:
    saver.restore(sess, FLAGS.pretrained_model)

  tf.train.start_queue_runners(sess)# 启动填充队列线程

  tf.logging.info('iteration number, cost')

  # Run training. 训练
  for itr in range(FLAGS.num_iterations):
    # Generate new batch of data. 生成新的批量
    feed_dict = {model.iter_num: np.float32(itr),
                 model.lr: FLAGS.learning_rate}
    cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                    feed_dict) #计算cost 和 summary_str

    # Print info: iteration #, cost.
    tf.logging.info(str(itr) + ' ' + str(cost))

    if (itr) % VAL_INTERVAL == 2:
      # Run through validation set. 运行验证集
      feed_dict = {val_model.lr: 0.0,
                   val_model.iter_num: np.float32(itr)}
      _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                     feed_dict)
      summary_writer.add_summary(val_summary_str, itr)

    if (itr) % SAVE_INTERVAL == 2: #保存模型
      tf.logging.info('Saving model.')
      saver.save(sess, FLAGS.output_dir + '/model' + str(itr))

    if (itr) % SUMMARY_INTERVAL:
      summary_writer.add_summary(summary_str, itr)

  tf.logging.info('Saving model.')
  saver.save(sess, FLAGS.output_dir + '/model')
  tf.logging.info('Training complete')
  tf.logging.flush()


if __name__ == '__main__':
  app.run()
