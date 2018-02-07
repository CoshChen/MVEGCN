# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:55:14 2017

@author: Ko-Shin Chen
"""

import tensorflow as tf
# import numpy as np

def init_weight(shape, layer_name = None):
        random_dist = tf.truncated_normal(shape, mean = 0.1, stddev = 0.5) # float32
        return tf.Variable(random_dist, name = layer_name)


def full_layer(input_layer, output_size, layer_name = None):
    input_size = int(input_layer.get_shape()[-1])
    W = init_weight([input_size, output_size])
    b = tf.Variable(tf.ones([output_size]))
    return tf.add(tf.matmul(input_layer, W), b, name = layer_name)

    
def edgeFeature_conv_layer(input_layer, output_size, act = tf.nn.relu, layer_name = None):
    """
    @ param output_size: output edge feature dimension
    @ return: tf.tensor [batch, nodeNum, nodeNum, output_size]
    """
    input_size = int(input_layer.get_shape()[-1])
    edgeFilter = init_weight([1, 1, input_size, output_size])
    return act(tf.nn.conv2d(input_layer, edgeFilter, strides = [1,1,1,1], padding = 'SAME'), name = layer_name)


def node_conv_layer(input_layer, adjMatrix, output_size, act = None, layer_name = None):
    """
    @ param adjMatrix: [batch, nodeNum, nodeNum]
    @ return: New nodes features of dimension output_size.
    """
    dims = input_layer.get_shape().as_list()
    batch_size = tf.shape(input_layer)[0]
    nodeNum = int(dims[1])
    input_size = int(dims[2])
    
    AX = tf.matmul(adjMatrix, input_layer) # AX = [batch, nodeNum, input_size]
    AX_flat = tf.reshape(AX, [batch_size * nodeNum, input_size])
    W = init_weight([input_size, output_size])
    AXW = tf.matmul(AX_flat, W)

    if not act:
        return tf.reshape(AXW, [batch_size, nodeNum, output_size], name = layer_name)

    return act(tf.reshape(AXW, [batch_size, nodeNum, output_size]), name = layer_name)


def reduction_along_nodes(nodesFeatures, option = None, layer_name = None):
    """
    @ param nodesFeatures: [batch, nodeNum, nodeFeatures]
    """
    if option == 'sum':
        return tf.reduce_sum(nodesFeatures, [1], name = layer_name)
    
    dims = nodesFeatures.get_shape().as_list()
    batch_size = tf.shape(nodesFeatures)[0]
    nodeNum = int(dims[1])
    nodeFeatureDim = int(dims[2])
    
    nodesFeatures_t = tf.transpose(nodesFeatures, [0, 2, 1])
    nodesFeatures_t_flat = tf.reshape(nodesFeatures_t, [batch_size * nodeFeatureDim, nodeNum])
    W = init_weight([nodeNum, 1])
    b = tf.Variable(tf.ones([nodeFeatureDim]))
    reduced = tf.matmul(nodesFeatures_t_flat, W)
    return tf.add(tf.reshape(reduced, [batch_size, nodeFeatureDim]), b, name = layer_name)
