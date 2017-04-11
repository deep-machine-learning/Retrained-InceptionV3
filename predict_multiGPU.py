#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:11:24 2017

Prediction script for generating 2048 dimensional embedding from pooling layer using muliple (tested for 8) GPUS. 

"""

import os, argparse
import numpy as np
from multiprocessing import Pool
import re
import pandas as pd
from tqdm import *
import tensorflow as tf
import sys
from memory_profiler import profile
import unicodecsv
import csv
import gc


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


def get_list_images_paths(folder):
    '''
    Get list of image filenames
    INPUT: folder of files
    RETURN: List of filenames
    '''
    return (os.path.join(folder, f) for f in tqdm((sorted(os.listdir(folder)))) if 'jpg' in f)


def get_tensor_from_image(image):
    '''
    Tuple of image name and image name
    INPUT: image name
    RETURN: Tuple of image name and image name
    '''
    return image, image


def get_list_tensors_tuples(list_images):
    '''
    Multithreading of list of images to tuples of image name and image name
    INPUT: list of image names
    RETURN: Tuple of image names and image names
    '''
    p = Pool(10)
    list_tensors = p.map(get_tensor_from_image, list_images)
    p.close()
    p.join()
    return list_tensors

def worker(tuple_gpu_batch):
    
    gc.set_debug(gc.DEBUG_STATS)
    '''
    Worker function for processing each image and converting to embedding and softmax prediction
    '''
    # Making one GPU visible per worker
    os.environ["CUDA_VISIBLE_DEVICES"] = tuple_gpu_batch[0] 
    
    # Open worker specific .csv file
    file_embedding = open('generated/embedding_worker_%s.csv'%(tuple_gpu_batch[0]),'wb')
    #file_softmax = open('generated/softmax_worker_%s.csv'%(tuple_gpu_batch[0]),'wb')
    file_name = open('generated/name_worker_%s.csv'%(tuple_gpu_batch[0]),'wb')
    
    # We use our "load_graph" function
    graph = load_graph(FLAGS.frozen_model_filename)
    
    # Input tensor
    x = graph.get_tensor_by_name('prefix/DecodeJpeg/contents:0')
    
    # Tensorflow session
    with tf.Session(graph=graph) as sess:           
        for id,tuple_path_tensor in enumerate(tuple_gpu_batch[1]):
            
            human_string = None
            # Input image file converted to tensor
            
            imageFileName = tuple_path_tensor[1]
            input_image = tf.gfile.FastGFile(imageFileName,'rb').read()
            
            # Embedding from tensor
            embedding = sess.run(graph.get_tensor_by_name('prefix/pool_3/_reshape:0'), feed_dict={x: input_image})
            
            '''
            # Softmax prediction from tensor
            prediction = sess.run(graph.get_tensor_by_name('prefix/softmax:0'), feed_dict={x: input_image})
            prediction_squeeze = np.squeeze(prediction)
            
            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()
            top_k = prediction_squeeze.argsort()[-FLAGS.num_top_predictions:][::-1]
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
            '''
                
            # Write embedding to csv            
            np.savetxt(file_embedding,embedding,fmt='%f')
            
            # Write softmax prediction to csv
            #np.savetxt(file_softmax,prediction,fmt='%f')
            
            # write to csv
            np.savetxt(file_name,[imageFileName],fmt='%s')
            
            # Log progress
            print ('Processed embedding number %s'%(id))
            
            if (id % 200 == 0):
                collected = gc.collect()
                print "Garbage collector: collected %d objects." % (collected)
           
        file_name.close()
        file_embedding.close()
        #file_softmax.close()
        
        
   

def map_gpus_to_tensors(list_gpus, tensors_batches):
    '''
    Map GPU batches to list of images
    INPUT: list of gpus, list of images
    RETURN: Tuple of (gpu_name,list_images)
    '''
    list_tuples = zip(list_gpus, tensors_batches)
    return list_tuples


def gen_batches(xs, size):
    '''
    Generate image batches of specific size
    INPUT: list of images , size of batch
    RETURN: Generated batches
    '''
    acc = []
    for i, x in enumerate(xs):
        if i and i % size == 0:
            yield acc
            acc = []
        acc.append(x)
    if acc:
        yield acc


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
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

if __name__ == '__main__':
    
    gc.set_debug(gc.DEBUG_STATS)
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./')
    parser.add_argument("--images_dir",type=str,help='Directory to read images from')
    parser.add_argument("--num_top_predictions", type=int, default=1)
    parser.add_argument("--frozen_model_filename", default="./classify_image_graph_def.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--image_name", type=str, help="Image to test") 
    #parser.add_argument("--class_name",type=str,help="Class Name")
    FLAGS, unparsed = parser.parse_known_args()
    
    # List of GPUS
    list_gpus = ["0","1","2","3","4","5","6","7"]
    # List of Images
    list_images = get_list_images_paths(FLAGS.images_dir)
    # List of image tuples
    list_images_tuples = get_list_tensors_tuples(list_images)
    print ('Number of images processed per GPU:%s')%(len(list(list_images_tuples))/8)
    # Image batches
    images_batches = gen_batches(list_images_tuples,(len(list(list_images_tuples))/len(list_gpus))) 
    # Map gpus to image batches
    list_tensors_gpus = map_gpus_to_tensors(list_gpus, images_batches)
    
    # Worker multithreading
    pool = Pool(8)
    list_tuples = pool.map(worker,list_tensors_gpus)
    pool.close()
    pool.join()
    
#     # merge lists from thread
#     name_list = []
#     #embedding_list = []
#     #softmax_list = []
#     class_prediction_list = []
#     
#     for tuple in list_tuples:
#         name_list.append(tuple[0])
#         #embedding_list.append(tuple[1])
#         #softmax_list.append(tuple[2])
#         class_prediction_list.append(tuple[1])
#     
#     # Merge all lists into big lists
#     final_name_list = [item for sublist in name_list for item in sublist]
#     #final_embedding_list = [item for sublist in embedding_list for item in sublist]
#     #final_softmax_list = [item for sublist in softmax_list for item in sublist]
#     final_class_prediction_list = [item for sublist in class_prediction_list for item in sublist]
#     
# #     # to csv
# #     f_emb = open('embedding_%s.csv'%(FLAGS.class_name),'wb')
# #     for emb in final_embedding_list:
# #         np.savetxt(f_emb,emb,delimiter=',',fmt='%f')
# #      
# #     f_sm = open('softmax_%s.csv'%(FLAGS.class_name),'wb')
# #     for sm in final_softmax_list:
# #         np.savetxt(f_sm,sm,delimiter=',',fmt='%f')
#     
#     df = pd.DataFrame({'Name':final_name_list,'Class Prediction':final_class_prediction_list})
#     df.to_csv('Name_Class_%s.csv'%(FLAGS.class_name))

    
    
    

