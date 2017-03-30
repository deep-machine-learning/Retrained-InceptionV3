#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import os, argparse
from tensorflow.python.framework import graph_util
import scipy.misc
import numpy as np
from scipy import ndimage
import cv2
import glob
import itertools
import os.path
import re
import sys
import tarfile
import Queue
import time
import pandas as pd

dir = os.path.dirname(os.path.realpath(__file__))
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

#graph = load_graph('/serving/frozen_model.pb')

def create_image_batch(folder):
	image_list = glob.glob(folder)
	img_np_list=[]
	for img in image_list:
		img_np =cv2.imread(img,1)
		img_np_list.append(img_np)
	img_pad = np.zeros(np.shape(img_np))
	input_list = list(itertools.izip_longest(*[iter(img_np_list)]*64,fillvalue=img_pad))
	return input_list

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str,default='/serving/')
    parser.add_argument("--num_top_predictions",type=int,default=1)
    parser.add_argument("--frozen_model_filename", default="/serving/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--image_name",type=str,help="Image to test")
    parser.add_argument("--layer",type=str,help="Output layer")

    FLAGS,unparsed = parser.parse_known_args()

    # Create input queue
    img_q = Queue.Queue()
    map(img_q.put, glob.glob('/serving/frames/*.jpg'))

    # We use our "load_graph" function
    graph = load_graph(FLAGS.frozen_model_filename)
	
    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/DecodeJpeg/contents:0') # Input tensor
    if FLAGS.layer=='pool':
    	y = graph.get_tensor_by_name('prefix/pool_3/_reshape:0') # Output tensor
    elif FLAGS.layer=='softmax':
	y = graph.get_tensor_by_name('prefix/softmax:0')
    
    # We launch a Session
    start = time.time()
    with tf.Session(graph=graph) as sess:
       # Note: we didn't initialize/restore anything, everything is stored in the graph_def
       embedding_list = []
       prediction_list = []
       while not img_q.empty():
       	image_data = tf.gfile.FastGFile(img_q.get(),'rb').read()
       	embedding = sess.run(y, feed_dict={x:image_data})
	prediction = sess.run(graph.get_tensor_by_name('prefix/softmax:0'),feed_dict={x:image_data})
	predictions = np.squeeze(prediction)

    	# Creates node ID --> English string lookup.
    	#node_lookup = NodeLookup()
    	#top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    	#for node_id in top_k:
      	#	human_string = node_lookup.id_to_string(node_id)
      	#	score = predictions[node_id]

	embedding_list.append(embedding)
	df.DataFrame.from_items([('embedding',embedding_list)])
	#prediction_list.append(human_string)
       #df = pd.DataFrame.from_items([('filenames',glob.glob('/serving/frames/*.jpg')),('embedding',embedding_list),('prediction',prediction_list)])
       #print (df)
       print ('total time: %s seconds'%(time.time()-start))
       df.to_csv('embedding_csv.csv', sep='\t')
