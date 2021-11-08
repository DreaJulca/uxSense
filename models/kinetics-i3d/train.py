"""
To train an acition classification model (CNN + LSTM)

  CNN - OpenPose model
    input : ego-centric videos
    output: PAFMap + HeatMap

  LSTM - single LSTM model
    input : output of the CNN model
    output: action classification

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import json
import os
import re
# visualization purpose
import time

# OpenPose
from tf_openpose.src.estimator import TfPoseEstimator
from tf_openpose.src.networks import get_graph_path, model_wh
from tf_openpose.src import common
import tf_openpose

# performance metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

"""
Download datasets

"""
def download_dataset(dataset_list, num_VGA, num_HD, base_dir):
  # Split the available datasets into batches for training and testing that don't exceed maybe 10GB
  # Iterate over each batch
  # Iterate over each dataset in the batch
  get_data_sh = os.path.join(base_dir, "scripts/getData.sh")
  extract_all_sh = os.path.join(base_dir, "scripts/extractAll.sh")
  fmt = "png"

  for name in dataset_list:
    dataset_path = os.path.join(base_dir, name)
    if not os.path.exists(dataset_path):
      #Download the video and 3d pose tar files
      print("Downloading data for", name)
      runStatus = subprocess.run([get_data_sh, name, str(num_HD), str(num_HD)],\
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      print(runStatus.stdout.decode("utf-8"))
      print(runStatus.stderr.decode("utf-8"))

      #Split the videos into frames and extract the 3d pose tar files
      print("Extracting data for", name)
      runStatus = subprocess.run([extract_all_sh, dataset_path, fmt], \
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      print(runStatus.stdout.decode("utf-8"))
      print(runStatus.stderr.decode("utf-8"))
    else:
      print(name, "is alreay existed")


"""
create LSTM-based model
"""
def lstm_layer(lstm_size, num_layers, batch_size, dropout_rate=None):
  """
  def cell(lstm_size, dropout_rate=None):
    layer = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)

  cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size, dropout_rate) for _ in range(num_layers)])
  """
  cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
  init_state = cell.zero_state(batch_size, tf.float32)

  return cell, init_state


"""
transform 4d conv to 2d matrix
"""
def flatten(layer, batch_size, time_step):
  dims = layer.get_shape()
  num_elements = dims[2:].num_elements()

  reshaped_layer = tf.reshape(layer, [batch_size, int(time_step / 2), num_elements])

  return reshaped_layer, num_elements


"""
create DENSE layer
"""
def dense_layer(inputs, in_size, out_size, dropout=False, activation=tf.nn.relu):
  weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=0.05))
  bias = tf.Variable(tf.zeros(out_size))

  inputs = tf.reshape(inputs, [-1, in_size])
  layer = tf.matmul(inputs, weights) + bias

  if activation != None:
    layer = activation(layer)

  if dropout:
    layer = tf.nn.dropout(layer, 0.5)

  return layer


"""
Softmax cross entropy loss function 
  @input
    loss:     loss function name
    predict:  softmax output of the LSTM model
    labels:   ground-truth labels (activity classifiation)   
  @output
    loss ouput
"""
def compute_loss(loss, predict, labels):
  if loss == "xentropy":
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(predict), 1),
                  reduction_indices=[1])
    loss_output = tf.reduce_mean(cross_entropy, name='xentropy_mean')

  return loss_output

"""
ADAM Optimizer
  @input
    predict:  LSTM output
    labels:   ground-truth labels (activity classification)
    opt:      dictionary for optimizer
  @output
"""
def loss_optimizer(predict, labels, opt):
  # get loss function
  loss = compute_loss(opt["loss"], predict, labels)
  # get optimizer
  if opt["name"] == "adam":
    #optimizer = tf.train.AdamOptimizer(learning_rate=opt["learning_rate"]).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

  return loss, optimizer


"""
config settings
"""
# define some options for the session/run
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
#sess_config.gpu_options.per_process_gpu_memory_fraction = 0.98

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

"""
dataset specification
"""
# list of training dataset
training_dataset = [#"141126_pose1", "141126_pose2", "141126_pose3", "141126_pose4", \
                    "141215_pose1", "141215_pose2", "141215_pose3", \
                    "141215_pose4", "141215_pose5", "141215_pose6", \
                    "141216_pose1", "141216_pose2", "141216_pose3", \
                    "141216_pose4", "141216_pose5", \
                    "141217_pose1", "141217_pose2", "141217_pose3", \
                    "141217_pose4", "141217_pose5"]

# training_dataset = ["141217_pose1_test"]
testing_dataset = [""]
# Specify the number of vga views you want to download. Up to 480
num_VGA = 1
# Specify the number of hd views you want to donwload. Up to 31
num_HD = 0
# dataset directory
base_dir = os.getcwd()
#download_dataset(training_datasets, num_VGA, num_HD, base_dir)

# TfPoseEstimator
# pretrained CNN model
scales = [None]
model = 'mobilenet_thin'

graph_path = get_graph_path(model)
graph_def = tf.GraphDef()
with tf.gfile.GFile(graph_path, 'rb') as f:
  graph_def.ParseFromString(f.read())

cnn_graph = tf.get_default_graph()
tf.import_graph_def(graph_def, name="TfPoseEstimator")
# pose_cnn = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
cnn_sess = tf.Session(graph=cnn_graph, config=sess_config)

cnn_input = cnn_graph.get_tensor_by_name('TfPoseEstimator/image:0')
cnn_output = cnn_graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')

w, h = 480, 480
"""
hyperparametes configuration
  lstm_size:  number of units in LSTM cell
  batch_size: batch size
  num_lstm_layers: the number of layers in LSTM
  epochs: epoch numbers
  opt: optimizer parameters
"""
lstm_size = 256 #the number of units in LSTM cell -- changed from 512
batch_size = 1
num_lstm_layers = 1
epochs = 700
max_timestep = 1

learn_rate = 0.001


opt = {}
opt["loss"] = "xentropy"
opt["name"] = "adam"
#opt["learning_rate"] = learn_rate
opt["eps"] = 0.000000001

min_learn_rate = 0.00001
#linear decrease until fiftieth epoch
shrink_learn_rate_by = 0.000018

# target meta
#gesture_size = 59
#We have decided to just look at a subset of 10 gestures
#gesture_size = 167
gesture_size = 10
lstm_input_size = 60 * 60 * 57
# placeholder for input
# TODO: change the input size
# [batch_size, max_timestep, feature dims]
lstm_input = tf.placeholder(tf.float32, [batch_size, 60, 60, 57], name="lstm_input")
lstm_label = tf.placeholder(tf.float32, [batch_size, gesture_size], name="target_action")

# human = tf.placeholder(tf_openpose.src.estimator.Human, [batch_size, 1])
# placeholder for ground-truth
initializer = tf.random_uniform_initializer(-1, 1)
lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, lstm_input_size, initializer=initializer)
# init_state = cell.zero_state(batch_size, tf.float32)

# cell_out : [batch_size, max_time, cell.output_size]
# cell_out = tf.nn.rnn_cell.OutputProjectionWrapper(cell, gesture_size)
cell_input = tf.reshape(lstm_input, [batch_size, max_timestep, -1])

outputs, states = tf.nn.dynamic_rnn(lstm_cell, cell_input, dtype=tf.float32)

logits = dense_layer(outputs, lstm_size, gesture_size, activation=None)

"""
cell, init_state = lstm_layer(lstm_size, num_lstm_layers, batch_size)
reshaped_outputs = flatten(outputs, batch_size, time_step)
outputs, states = tf.nn.dynamic_rnn(cell, lstm_input, initial_state=init_state)

# TODO: change the parameters for dense layer
logits = dense_layer(outputs, lstm_size, gesture_size, activation=None)
"""

prediction = tf.nn.softmax(tf.reshape(logits, [-1, gesture_size]))

loss, opt = loss_optimizer(prediction, lstm_label, opt)

# calculate accuracy
# accuracy = tf.reduce_all(tf.equal(prediction, lstm_label))
print("Training with data from " + os.path.join(os.path.dirname(os.getcwd()),"datasets"))

# get Saver to save the model
saver = tf.train.Saver()

#Prefer a managed context...
lstm_sess = tf.Session(config=sess_config)
lstm_sess.run(tf.global_variables_initializer(), options = run_opts)


# initial state of the LSTM memory
"""
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
"""

# probabilities = []
# loss = 0.0

for i in range(epochs):
    """
    Each video in a dataset has a json file
    key: frame number
    value: gesture

    value should be used as the ground-truth for our LSTM model
    """
    epoch_loss = []
    # train_accuracy = []
    y_pred = []
    y_test = []
    # num_skip_frames = 0
    # NOTE: datasets are located under the parent folder on Andrea's desktop
    parent_dir = os.path.dirname(os.getcwd())
    # TODO: implement batch-input
    for each_video in training_dataset:
        # directory for images extracted from the video
        img_dir = os.path.join(parent_dir, "datasets", each_video, "vgaImgs/01_01")
        # a json file containts the ground-truth (label) data of the video
        json_filename = each_video + ".json"
    
        # read the ground-truth for the video
        with open(os.path.join("jsons", json_filename), 'r') as json_f:
            json_data = json.load(json_f)
            # print(json_data)

            for each_frame in sorted(os.listdir(img_dir)):
                label = ""
                frame_num = 0
                gesture_out_onehot = np.zeros([gesture_size], dtype=float)

                if each_frame.endswith(".jpg"):
                    frame_num = int(re.split("_|\.", each_frame)[2])
                    try:
                        label = json_data[str(frame_num)]
                    # print(frame_num, label)
                    except KeyError:
                        label = "NONEXISTED"
                    # print("%d not existed in json" % (frame_num))

                    # train our LSTM model only if label is not "NONEXISTED"
                    # NOTE: do not skip "NA" labels anymore
                    #if label != "NONEXISTED" and label != "NA":
                    if label != "NONEXISTED":
                    # print("Frame %d is labelled as \'%s\'" % (frame_num, label))
                        image = common.read_imgfile(os.path.join(img_dir, each_frame), w, h)
            
                        # get CNN output
                        # t = time.time()
                        # pose_output = pose_cnn.inference(image)
                        image = np.expand_dims(image, axis=0)
                        pose_output = cnn_sess.run(cnn_output, feed_dict={cnn_input: image}, options = run_opts)
                        # elapsed = time.time() - t
                        # print("took %.6f seconds to get output for [%d] frame with \'%s\' label"\
                                # % (elapsed, frame_num, elapsed))
            
                        # print(pose_output.shape)

                        # gesture label starts with G followed by action number
                        #if label == "NA":
                        #    action_num = 0
                        #else:
                        if label != "NA" and label != "NONEXISTED":
                            #Subtract 1 because, in R, indexing starts at 1 & I am lazy
                            action_num = int(label[1:]) - 1
                            gesture_out_onehot[action_num] = 1
                  
                        # print("Action: %s and its action number: %d" % (label, action_num))
            
                        # print(gesture_out_onehot)

                        # TODO: handle "batch_size > 1" and "max_timestep > 1"
                        gt_label = np.expand_dims(gesture_out_onehot, axis=0)

                        # input the CNN output to the LSTM model
                        cost, _, pred = lstm_sess.run([loss, opt, prediction],\
                                    feed_dict={lstm_input: pose_output, lstm_label: gt_label})

                        # return an index of maximum (return a predicated action number)
                        action_output = np.argmax(pred)

                        if label != "NA":
                            y_test.append(action_num)
                            y_pred.append(action_output)

                        epoch_loss.append(cost)
                        # train_accuracy.append(accu)

    """
    performan metrics
    - confusion matrix
    - accuracy
    """
    train_accuracy_score = accuracy_score(y_test, y_pred)
    # train_f1 = f1_score(y_test, y_pred)

    print("Epoch: %d/%d | Epoch loss: %f | Train accuracy: %f" % 
        (i, epochs, np.mean(epoch_loss), train_accuracy_score))

    # print("Epoch: %d/%d | Epoch loss: %f | Mean accuracy: %f | Train accuracy: %f | Train F1: %f" % 
    #         (i, epochs, np.mean(epoch_loss), np.mean(train_accuracy), train_accuracy_score, train_f1))

    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)

    # now save the model for later use
    if i%5 == 0:
        model_path = "model/lstm_model+" + str(epochs) + "-" + str(i) + ".ckpt"
        save_path = saver.save(lstm_sess, model_path)
        print("Model saved in path: %s" % save_path)

    if learn_rate > min_learn_rate:
        learn_rate += -shrink_learn_rate_by
        #opt["learning_rate"] = learn_rate


print("All epochs completed")
model_path = "model/lstm_model+" + str(epochs) + "-" + str(epochs) + ".ckpt"
save_path = saver.save(lstm_sess, model_path)
print("Model saved in path: %s" % save_path)


"""
[Pseudocode]
for each_video in dataset
for each_frame in each_video:
pose_output = pose_cnn.inference(each_frame)
# the value of the state is upated after processing each frame
output, state = lstm(each_frame, state)
    
logits = tf.matmul(output, softmax_w) + softmax_b
probabilities.append(tf.nn.softmax(logits))
loss += compute_cross_entropy_loss(softmax, labels, head)
"""

  