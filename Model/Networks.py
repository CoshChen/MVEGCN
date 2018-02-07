# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:22:31 2017

@author: Ko-Shin Chen
"""

import tensorflow as tf
from Model import Layers

def EGCN(adjTensor, placeholder_list, A_hat_hidden_layer_list, GCN_layer_list, EGCN_name = None):
    """
    @param adjTensor: placeholder [batch, maxNodes, maxNodes, edgeFeatureDim + 1]
    @param placeholder_list: [idPlusAdj, nodeFeatureMatrix]
    @param A_hat_hidden_layer_list: output channels in hidden layers, e.g. [4]
    @param GCN_layer_list: output node feature dimensions, e.g. [16, 8]
    @param EGCN_name: optional layer name
    """
    if not EGCN_name: EGCN_name = 'EGCN/'
    
    A_hat_hidden_layer_list.append(1)  # make the last output channel number 1
    
    idPlusAdj = placeholder_list[0]  # [batch, maxNodes, maxNodes]
    nodeFeatureMatrix = placeholder_list[1]  # [batch, maxNodes, nodeFeatureDim]
    
    eL = []  # eL[i] dim = [batch, maxNodes, maxNodes, A_hat_layer_list[i]]
    nL = []  # nL[i] dim = [batch, maxNodes, GCN_layer_list[i]]
    
    '''
    Network -- A_hat
    '''
    m = len(A_hat_hidden_layer_list)
    
    if m == 1:
        eL.append(Layers.edgeFeature_conv_layer(adjTensor, A_hat_hidden_layer_list[0], tf.nn.sigmoid, layer_name = 'AdjWeight'))
    else:
        eL.append(Layers.edgeFeature_conv_layer(adjTensor, A_hat_hidden_layer_list[0]))
        
        for i in range(1, m-1):
            eL.append(Layers.edgeFeature_conv_layer(eL[i-1], A_hat_hidden_layer_list[i]))
            
        eL.append(Layers.edgeFeature_conv_layer(eL[m-2], A_hat_hidden_layer_list[m-1], tf.nn.sigmoid, layer_name = 'AdjWeight'))
    
    eL_sq = tf.squeeze(eL[-1], squeeze_dims = [3])  # [batch, maxNodes, maxNodes]
    A_tilde = tf.add(idPlusAdj, eL_sq, name = EGCN_name + 'A_tilde')  # [batch, maxNodes, maxNodes]
    
    degrees = tf.reduce_sum(A_tilde, 2)
    degrees_inv = tf.pow(degrees, -0.5)
    diagTensor = tf.diag(degrees_inv)
    D_tilde = tf.reduce_sum(diagTensor, 2)
    
    DA = tf.matmul(D_tilde, A_tilde)
    A_hat = tf.matmul(DA, D_tilde, name = EGCN_name + 'A_hat')
    
    '''
    Network -- Graph Convolution
    '''
    n = len(GCN_layer_list)

    if n == 1:
        return Layers.node_conv_layer(nodeFeatureMatrix, A_hat, GCN_layer_list[0], layer_name = EGCN_name + 'layer_0')

    nL.append(Layers.node_conv_layer(nodeFeatureMatrix, A_hat, GCN_layer_list[0], act = tf.nn.relu, layer_name = EGCN_name + 'layer_0'))
    
    for i in range(1, n-1):
        nL.append(Layers.node_conv_layer(nL[i-1], A_hat, GCN_layer_list[i], act = tf.nn.relu, layer_name = EGCN_name + 'layer_' + str(i)))
        
    return Layers.node_conv_layer(nL[-1], A_hat, GCN_layer_list[n-1], layer_name = EGCN_name + str(n-1))


def get_graph_representation(EGCN_output_list, option = None):
    if len(EGCN_output_list) == 1:
        return Layers.reduction_along_nodes(EGCN_output_list[0], option, "graph_representation")
    
    fingerPrints = []
    
    for i in range(len(EGCN_output_list)):
        fingerPrints.append(Layers.reduction_along_nodes(EGCN_output_list[i], option))
    
    return tf.concat(fingerPrints, 1, name = "graph_representation")  # [batch, totalFeatures]


def get_prediction(graph_representation, hidden_full_layer_list):
    """
    @ param graph_representation: [batch, features]
    @ param full_layer_list: output feature dimensions in hidden layers, e.g. [4, 2] 
    """

    m = len(hidden_full_layer_list)

    if(m == 0):
        output = Layers.full_layer(graph_representation, 1)
        return output

    DL = []
    DL_act = []
    DL.append(Layers.full_layer(graph_representation, hidden_full_layer_list[0]))

    for i in range(1, m):
        DL_act.append(tf.nn.relu(DL[i-1]))
        DL.append(Layers.full_layer(DL_act[i-1], hidden_full_layer_list[i]))

    DL_act.append(tf.nn.relu(DL[-1]))
    output = Layers.full_layer(DL_act[-1], 1)
    return output
            
       