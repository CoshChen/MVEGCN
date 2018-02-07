# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:56:04 2017

@author: Ko-Shin Chen
"""

import tensorflow as tf
from Model import Networks


def model_prediction(adjTensor_list, placeholder_list, \
                     list_A_hat_hidden_layer_list, list_GCN_layer_list, \
                     hidden_full_layer_list, reduction_option = None):
    
    m = len(adjTensor_list)
    
    if (m != len(list_A_hat_hidden_layer_list)) \
        or (m != len(list_GCN_layer_list)):
            raise ValueError("The size of adjTensor_list, \
                             list_A_hat_hidden_layer_list, and \
                             list_GCN_layer_list should be the same.")
        
    EGCN_output = []
    for i in range(m):
        EGCN_output.append(Networks.EGCN(adjTensor_list[i], placeholder_list, \
                                         list_A_hat_hidden_layer_list[i], \
                                         list_GCN_layer_list[i], 'EGCN_view_' + str(i) + '/'))
    
    fingerPrint = Networks.get_graph_representation(EGCN_output, option = reduction_option)
    prediction = Networks.get_prediction(fingerPrint, hidden_full_layer_list)
    prediction_sq = tf.squeeze(prediction, squeeze_dims = [1], name = "pred_value")
    return prediction_sq


def loss_MSE(prediction, true_val):
    return tf.reduce_mean(tf.square(true_val - prediction), name = 'loss_MSE')


def loss_RMSE(prediction, true_val):
    mse = tf.reduce_mean(tf.square(true_val - prediction), name = 'MSE')
    return tf.sqrt(mse, name = 'loss_RMSE')


def loss_CE_binary(prediction, true_val):
    loss_vector = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_val, logits=prediction)
    return tf.reduce_mean(loss_vector, name = 'loss_CE_binary')
