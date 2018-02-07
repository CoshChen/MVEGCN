# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:43:38 2017

@author: Ko-Shin Chen

Dataset: BradleyDoublePlusGoodMeltingPointDataset.csv
Data Source: https://github.com/connorcoley/conv_qsar_fast/tree/master/data
No edge features (AdjMatrix only)
"""

import tensorflow as tf
import math
import random
import os
from Model import LossFunction
from Graph import GraphBuilder, DataLoader


tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

filePath = '../data/BradleyDoublePlusGoodMeltingPointDataset.csv'
out_file_dir = '../Output/Baseline/Round_1'
check_point_file = out_file_dir + '/BD_trained.ckpt'
min_check_point_file = out_file_dir + '/BD_min.ckpt'
check_point_file_to_load = check_point_file
processed_data = out_file_dir + '/BD_split.csv'

epoch = 2000
start = 0
report = 10
batch_size = 200

if not os.path.exists(out_file_dir):
    os.makedirs(out_file_dir)


'''
Load Data
'''
data = DataLoader.RegressionDataLoader()

if not os.path.exists(processed_data):
    maxNodes, nodeFeatureDim, edgeFeatureDim = data.load_csv(filePath, 'smiles', 'mpC')
    trainGraph, testGraph = data.get_trainTestGraph()
    data.export_data(processed_data)
    print(" ")

else:
    [trainGraph, testGraph], maxNodes, nodeFeatureDim, edgeFeatureDim = data.import_csv_data(processed_data)
    print(" ")
    

train_size = len(trainGraph)
test_size = len(testGraph)


'''
Model Graph
'''
if not os.path.exists(check_point_file_to_load + '.meta'):
    print("Building new model graph.")
    
    new_model = True
    
    with tf.device('gpu:1'):
        '''
        Placeholders
        '''
        adjTensor_view_1 = tf.placeholder(tf.float32, shape = [None, maxNodes, maxNodes, 0 + 1], name = "adjTensor_view_1")
        idPlusAdj = tf.placeholder(tf.float32, shape = [None, maxNodes, maxNodes], name = "idPlusAdj")
        nodeFeatureMatrix = tf.placeholder(tf.float32, shape = [None, maxNodes, nodeFeatureDim], name = "nodeFeatureMatrix")
        true_val = tf.placeholder(tf.float32, shape = [None], name = "true_val")

        '''
        Prediction and Loss Function
        '''
        prediction = LossFunction.model_prediction([adjTensor_view_1], \
                                               [idPlusAdj, nodeFeatureMatrix], \
                                               [[]], [[16, 8]], \
                                               [8, 4, 2])

        loss = LossFunction.loss_RMSE(prediction, true_val)
        saver = tf.train.Saver()
    
else:
    print("Reloading existing model")
    
    new_model = False

    with tf.device('gpu:1'):
        saver = tf.train.import_meta_graph(check_point_file_to_load + '.meta')

        '''
        Placeholders
        '''
        adjTensor_view_1 = tf.get_default_graph().get_tensor_by_name('adjTensor_view_1:0')
        idPlusAdj = tf.get_default_graph().get_tensor_by_name('idPlusAdj:0')
        nodeFeatureMatrix = tf.get_default_graph().get_tensor_by_name('nodeFeatureMatrix:0')
        true_val = tf.get_default_graph().get_tensor_by_name('true_val:0')

        '''
        Prediction and Loss Function
        '''
        prediction = tf.get_default_graph().get_tensor_by_name('pred_value:0')
        loss = tf.get_default_graph().get_tensor_by_name('loss_RMSE:0')
    

'''
Optimizer
'''
if new_model:
    with tf.device('gpu:1'):
        train = tf.train.AdamOptimizer().minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.add_to_collection('train', train)
    
else:
    with tf.device('gpu:1'):
        train = tf.get_collection('train')[0]
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, check_point_file_to_load)
    
    
'''
Feed Dict
'''
view_1 = [0]

def get_batch(graph_list, batch_size):
    n = len(graph_list)
    batch_num = max(1, int(n/batch_size))
    indices = [i for i in range(n)]
    random.shuffle(indices)
    
    feed_dict_list = []
    
    for b in range(batch_num):
        graph_sub_list = [graph_list[i] for i in indices[b*batch_size:(b+1)*batch_size]]
        adjTensor_view_1_batch = [g.getAdjTensor(maxNodes)[:,:, view_1] for g in graph_sub_list]
        idPlusAdj_batch = [g.getIdPlusAdjMatrix(maxNodes) for g in graph_sub_list]
        X_batch = [g.getNodeFeatureMatrix(maxNodes) for g in graph_sub_list]
        label_batch = [g.scaled_label for g in graph_sub_list]
        
        feed_dict_list.append({adjTensor_view_1: adjTensor_view_1_batch,
            idPlusAdj: idPlusAdj_batch,
            nodeFeatureMatrix: X_batch,
            true_val: label_batch})
    
    return feed_dict_list


def single_feed_dict(graph = GraphBuilder.Graph()):
    return {adjTensor_view_1: [graph.getAdjTensor(maxNodes)[:,:,view_1]],
            idPlusAdj: [graph.getIdPlusAdjMatrix(maxNodes)],
            nodeFeatureMatrix: [graph.getNodeFeatureMatrix(maxNodes)],
            true_val: [graph.scaled_label]}


testFeed = {adjTensor_view_1: [g.getAdjTensor(maxNodes)[:,:,view_1] for g in testGraph],
            idPlusAdj: [g.getIdPlusAdjMatrix(maxNodes) for g in testGraph],
            nodeFeatureMatrix: [g.getNodeFeatureMatrix(maxNodes) for g in testGraph],
            true_val: [g.scaled_label for g in testGraph]}

    
'''
Training
'''

def get_total_loss(graph_list):
    total = 0.0
    for g in graph_list:
        total += math.pow(loss.eval(feed_dict = single_feed_dict(g)), 2.0)
    
    return math.sqrt(total/len(graph_list))


min_RMSE_train = get_total_loss(trainGraph)
min_RMSE_test = loss.eval(feed_dict = testFeed)
R2_min_train = 1 - (data.var * math.pow(min_RMSE_train, 2.0) / data.trainVar)
R2_min_test = 1 - (data.var * math.pow(min_RMSE_test, 2.0) / data.testVar)

print("Initial RMSE: " + str(min_RMSE_train))
print(" ")

for i in range(start, epoch):
    batch_list = get_batch(trainGraph, batch_size)
    for batch in batch_list:
        sess.run(train, feed_dict = batch)
        print("one batch done")

    print(str(i) + ": ----- one epoch done -----")
        
    if (i+1)%report == 0:
        saver.save(sess, check_point_file)
        train_RMSE = get_total_loss(trainGraph)
        test_RMSE = loss.eval(feed_dict = testFeed)

        print(i)
        print("Training RMSE: " + str(train_RMSE))
        print("Test RMSE: " + str(test_RMSE))
        print("Training R2: " + str(1 - (data.var * math.pow(train_RMSE, 2.0) / data.trainVar)))
        print("Test R2: " + str(1 - (data.var * math.pow(test_RMSE, 2.0) / data.testVar)))

        total = (test_size * math.pow(test_RMSE, 2.0) + train_size * math.pow(train_RMSE, 2.0))/(test_size + train_size)

        print("Overall R2: " + str(1 - total))
        print(" ")

        if train_RMSE < min_RMSE_train:
            saver.save(sess, min_check_point_file)
            min_RMSE_train = train_RMSE
            R2_min_train = 1 - (data.var * math.pow(train_RMSE, 2.0) / data.trainVar)
            R2_min_test = 1 - (data.var * math.pow(test_RMSE, 2.0) / data.testVar)


print("Done training epoch = " + str(epoch))
print("Training RMSE: " + str(min_RMSE_train))
print("Test RMSE: " + str(min_RMSE_test))
print("Training R2: " + str(R2_min_train))
print("Test R2: " + str(R2_min_test))

