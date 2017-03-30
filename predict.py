from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import ops
import scopes

from keras import backend as K

from keras.models import Model



def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.
    "Rethinking the Inception Architecture for Computer Vision"
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for name_scope.
  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  
	
  with tf.name_scope(scope, 'inception_v3', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # 147 x 147 x 64
        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      # Inception blocks
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x256a'] = net
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x288a'] = net
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x288b'] = net
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat([branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_17x17x768a'] = net
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 128, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768b'] = net
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768c'] = net
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768d'] = net
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 192, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768e'] = net
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                  padding='VALID')
          aux_logits = ops.flatten(aux_logits)
          aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                              stddev=0.001, restore=restore_logits)
          end_points['aux_logits'] = aux_logits
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 192, [1, 1])
            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat([branch3x3, branch7x7x3, branch_pool], 3)
          end_points['mixed_17x17x1280a'] = net
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat([ops.conv2d(branch3x3, 384, [1, 3]),
                                   ops.conv2d(branch3x3, 384, [3, 1])], 3)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat([ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                      ops.conv2d(branch3x3dbl, 384, [3, 1])], 3)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_8x8x2048a'] = net
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat([ops.conv2d(branch3x3, 384, [1, 3]),
                                   ops.conv2d(branch3x3, 384, [3, 1])], 3)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat([ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                      ops.conv2d(branch3x3dbl, 384, [3, 1])], 3)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_8x8x2048b'] = net
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = ops.dropout(net, dropout_keep_prob, scope='dropout')
          net = ops.flatten(net, scope='flatten')
          # 2048
          logits = ops.fc(net, num_classes, activation=None, scope='logits',
                          restore=restore_logits)
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.misc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")

WORKDIR='/media/analytics/data/DiabeticRetinopathy/'
def resize_img(image,name):
    img = scipy.misc.imread(image)
    img_resize = scipy.misc.imresize(img,(299,299))
    scipy.misc.imsave(('%s%s')%(WORKDIR,name),img_resize)
    

def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def eval_image(image, height, width, scope=None):
  """Prepare one image for evaluation.
  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.op_scope([image, height, width], scope, 'eval_image'):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image


def image_preprocessing(image_buffer, thread_id=0):
  """Decode and preprocess one image for evaluation or training.
  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread
  Returns:
    3-D float Tensor containing an appropriately scaled image
  Raises:
    ValueError: if user does not provide bounding box
  """
  

  image = decode_jpeg(image_buffer)
  height = FLAGS.image_size
  width = FLAGS.image_size
  
  image = eval_image(image, height, width)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image

import tensorflow.contrib.slim as slim
import ops
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
def inference(images, num_classes, for_training=False, restore_logits=True,scope=None):
  """Build Inception v3 model architecture.
  See here for reference: http://arxiv.org/abs/1512.00567
  Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.
  Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
  """
  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  ''' with slim.arg_scope([ops.conv2d,ops.fc], weight_decay=0.00004):
    with slim.arg_scope([ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):'''
  logits, endpoints = inception_v3(
          images,num_classes=5)
  return logits
          

"""
cntains common code shared by all inception models.
Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)
"""

import tensorflow as tf

slim = tf.contrib.slim


def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001):
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

'''
config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=config) as sess:
	
	
	saver = tf.train.import_meta_graph('/media/analytics/data/DiabeticRetinopathy/Models/PhaseII/model.ckpt-999.meta')
	saver.restore(sess,'/media/analytics/data/DiabeticRetinopathy/Models/PhaseII/model.ckpt-999')
	print ('Started session....')
	tens=image_preprocessing(tf.read_file('/media/analytics/data/DiabeticRetinopathy/16_left.jpeg'))
	tf.initialize_all_variables().run()
	graph = tf.get_default_graph().as_graph_def()
	writer = tf.summary.FileWriter(logdir='logdir', graph=graph)
	writer.flush()
	##op = sess.graph.get_operations()
	#y_conv = graph.get_operation_by_name('import/pool_3:0')
	#print saver
	#logits,endpoints = sess.run(saver(tf.expand_dims(tens,0),5))
	#tf.Print(logits,[logits])
	#s = tf.nn.softmax(logits,name=None)
	#print sess.run(logits)
	#print logits
'''
config = tf.ConfigProto(allow_soft_placement=True)
saver = tf.train.import_meta_graph('/media/analytics/data/DiabeticRetinopathy/Models/PhaseII/model.ckpt-999.meta')
graph = tf.get_default_graph()
tf_train_dataset = tf.placeholder(tf.float32, shape=(1, 299, 299, 3))
with tf.Session(config=config,graph=graph) as session:
      
      ckpt = tf.train.get_checkpoint_state('/media/analytics/data/DiabeticRetinopathy/Models/PhaseII/model.ckpt-999')
      saver.restore(session, '/media/analytics/data/DiabeticRetinopathy/Models/PhaseII/model.ckpt-999')
      tens=image_preprocessing(tf.read_file('/media/analytics/data/DiabeticRetinopathy/16_left.jpeg'))
      test_batch = tf.expand_dims(tens,0)
      init_op = tf.group(t
f.initialize_all_variables(), tf.initialize_local_variables())
      session.run(init_op)
      with slim.arg_scope(inception_arg_scope()):  
	logits = inception_v3(test_batch,num_classes=6,is_training=False)
      test_prediction = tf.nn.softmax(logits)
      predictions = session.run(test_prediction)	
	
	

	

