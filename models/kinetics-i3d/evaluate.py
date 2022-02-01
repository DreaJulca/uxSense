from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from sys import getsizeof
from sys import exc_info

import os
import cv2

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import i3d
import gc

import getopt


_FRAME_RATE = 25
#unfortunately, we get some problems with this frame rate; fine to round
round_frame = 25

_IMAGE_SIZE = 224
_NUM_CLASSES = 400
_CHECKPOINT_PATHS = {
  'rgb': 'models/kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
  'flow': 'models/kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
  'rgb_imagenet': 'models/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
  'flow_imagenet': 'models/kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'models/kinetics-i3d/data/label_map.txt'

def evaluate(data, outfile, _SAMPLE_VIDEO_FRAMES):
  f = open(outfile, "w+")
  # define some options for the session/run
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True

  #sess_config.gpu_options.per_process_gpu_memory_fraction = 0.98
  
  run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  eval_type = 'rgb'
  imagenet_pretrained = True


  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.compat.v1.placeholder(
      tf.float32,
      shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.compat.v1.variable_scope('RGB', reuse=tf.compat.v1.AUTO_REUSE):
      rgb_model = i3d.InceptionI3d(
        _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
        rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.compat.v1.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.compat.v1.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.compat.v1.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.compat.v1.variable_scope('Flow', reuse=tf.compat.v1.AUTO_REUSE):
      flow_model = i3d.InceptionI3d(
        _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
        flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.compat.v1.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.compat.v1.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.compat.v1.Session(config=sess_config) as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
      tf.compat.v1.logging.info('RGB checkpoint restored')
      rgb_sample = data
      tf.compat.v1.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.compat.v1.logging.info('Flow checkpoint restored')
      flow_sample = np.load(_SAMPLE_PATHS['flow'])
      tf.compat.v1.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict,
        options = run_opts)

    out_logits = out_logits[0]
    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    f.write('Norm of logits: %f' % np.linalg.norm(out_logits))
    f.write('\nTop classes and probabilities')
    for index in sorted_indices[:_NUM_CLASSES]:
      f.write('\n'+str(out_predictions[index])+" "+str(out_logits[index])+"  "+'"'+kinetics_classes[index]+'"')
    f.write('\n')
    f.close()

    try:
      for var, obj in locals().items():
        if getsizeof(obj) > 1000000:
          print (var)
          print(getsizeof(obj))
    except:
      print("Unexpected error:", exc_info()[0])

    try:
      for var, obj in globals().items():
        if getsizeof(obj) > 1000000:
          print (var)
          print(getsizeof(obj))
    except:
      print("Unexpected error:", exc_info()[0])

    sess.close()
    gc.collect()

def main(argv):
  argument = ''
  usage = 'usage: echo_args.py -f <sometext>'
  # parse incoming arguments
  try:
    opts, args = getopt.getopt(argv,"hf:",["vid=", "startframe=", "endframe="])
  except getopt.GetoptError:
    print(usage)
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print(usage)
      sys.exit()
    if opt in ("-f", "--vid"):
      argument = arg
    if opt in ("--startframe"):
      startframe = int(arg)
    if opt in ("--endframe"):
      endframe = int(arg)

#Limit frames in single subset -- MUST BE 10x frame rate OR LESS! Add one bc R index starts at 1
  _SAMPLE_VIDEO_FRAMES = 1 + endframe - startframe

  cd = os.getcwd()
  vidpath = cd + "/" + argument
  actpath = vidpath.replace(".mp4", "/actions/")
  
  if not os.path.exists(actpath):
    try:
      os.mkdir(actpath)
    except OSError:
      print ("Creation of the directory %s failed" %  actpath)
    else:
      print ("Successfully created the directory %s " %  actpath)

  outpath = actpath + "frames_" + str(startframe) + "_through_" + str(endframe) + '.txt'
  data = []
  data.append([])
  vidcap = cv2.VideoCapture(vidpath)
  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    #Add to count first because we did previous step in R, which indexes at 1
    count += 1
    success,image = vidcap.read()

    if(count)>=startframe:
      
      res = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
      data[0].append(res)
      #print(imgcount)

      if(count)>=endframe:
        print("Applying model and saving array for "+outpath)
        dat = np.asarray(data, dtype=np.float32)/255
        if(dat.shape == (1, _SAMPLE_VIDEO_FRAMES, 224, 224, 3)):
          evaluate(dat, outpath, _SAMPLE_VIDEO_FRAMES)
          quit()
        else:
          print("Chunk of wrong dimensions: ")
          print(dat.shape)
          quit()
if __name__ == "__main__":
  main(sys.argv[1:])