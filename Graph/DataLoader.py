# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:32:57 2017

@author: Ko-Shin Chen
"""

import random
import math
import csv
import rdkit.Chem.rdmolfiles as rf
from Graph import GraphBuilder


class DataLoader:
    def __init__(self):
        self.maxNodes = 0
        self.graphList = []
        self.trainInd = []
        self.testInd = []
        self.split = False
        return
    
    def load_csv(self, file = '', col_1 = 'smiles', col_2 = 'label'):
        print("Load Data")
        
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for row in reader:
                counter += 1
                smiles = row[col_1]
                mol = rf.MolFromSmiles(smiles)
                
                if(len(smiles) > 0 and mol):
                    graph = GraphBuilder.molToGraph(mol)
                    if len(graph.edges) > 0:
                        graph.smiles = smiles
                        graph.label = float(row[col_2])
                        graph.scaled_label = float(row[col_2])
                        
                        self.graphList.append(graph)

                        if graph.nodeNum > self.maxNodes:
                            self.maxNodes = graph.nodeNum

                    else: print("Row " + str(counter) + " has no edge. \n")
                else: print("Row " + str(counter) + " has no smiles/mol. \n")
                
        print("Number of Samples in the DataSet: " + str(len(self.graphList)))
        print("Node Number: " + str(self.maxNodes))
        print("Node Feature Dimension: " + str(self.graphList[0].nodeFeatureDim))
        print("Edge Feature Dimension: " + str(self.graphList[0].edgeFeatureDim))
        return
    
    
    def reset_splitting(self):
        self.trainInd = []
        self.testInd = []
        self.split = False
        return
    
    
    def get_trainTestGraph(self, portion = 0.9):
        self.trainInd, self.testInd = self.__splitIndices(portion)
        trainGraph = [self.graphList[i] for i in self.trainInd]
        testGraph = [self.graphList[i] for i in self.testInd]
        print("Training Size: " + str(len(trainGraph)))
        print("Test Size: " + str(len(testGraph)))
        return trainGraph, testGraph
    
    
    def __splitIndices(self, portion = 0.9):
        if self.split: return self.trainInd, self.testInd

        N = len(self.graphList)
        trainSize = int(N*portion)
        indices = [i for i in range(N)]
        random.shuffle(indices)
        self.split = True
        return indices[:trainSize], indices[trainSize:]   



class RegressionDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.mean = 0.0
        self.var = 0.0
        self.trainVar = 0.0
        self.testVar = 0.0
        return
    
    
    def load_csv(self, file = '', col_1 = 'smiles', col_2 = 'label'):
        self.__init__()
        super().load_csv(file, col_1, col_2)
        
        self.mean, self.var = self.__getMeanVar(self.graphList)
        sd = math.sqrt(self.var)
        
        self.rescale_labels(self.mean, sd)
        
        
        print("Data Mean: " + str(self.mean))
        print("Data Sd: " + str(sd))
        
        return self.maxNodes, self.graphList[0].nodeFeatureDim, self.graphList[0].edgeFeatureDim 
    
    
    def rescale_labels(self, center, scale):
        """
        Update the scaled_lable for each graph in graphList.
        Then return mean and sd of the original labels for reference
        """
        for g in self.graphList:
            g.scaled_label = (g.label - center)/scale    
        return
        
    
    def get_trainTestGraph(self, portion = 0.9):
        trainGraph, testGraph = super().get_trainTestGraph(portion)
        _, self.trainVar = self.__getMeanVar(trainGraph)
        _, self.testVar = self.__getMeanVar(testGraph)
        print("TestVar: " + str(self.testVar))
        return trainGraph, testGraph
 
    
    def export_data(self, file):
        with open(file, 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Smiles', 'Label', 'Scaled Label', 'In Train']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(len(self.graphList)):
                graph = self.graphList[i]
                writer.writerow({'Index': i, 'Smiles': graph.smiles,
                                 'Label': graph.label,
                                 'Scaled Label': graph.scaled_label,
                                 'In Train': i in self.trainInd})
    
        print("Processed data is exported.")
        return

    
    def import_csv_data(self, file):
        print("Load Processed Data")
        self.__init__()
        self.split = True
        
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mol = rf.MolFromSmiles(row['Smiles'])
                graph = GraphBuilder.molToGraph(mol)
                graph.smiles = row['Smiles']
                graph.label = float(row['Label'])
                graph.scaled_label = float(row['Scaled Label'])
                self.graphList.append(graph)
                
                if graph.nodeNum > self.maxNodes:
                    self.maxNodes = graph.nodeNum
                    
                if row['In Train'] == 'True':
                    self.trainInd.append(int(row['Index']))
                else: self.testInd.append(int(row['Index']))
                
        self.mean, self.var = self.__getMeanVar(self.graphList)
        sd = math.sqrt(self.var)
        
        print("Number of Samples in the DataSet: " + str(len(self.graphList)))
        print("Data Mean: " + str(self.mean))
        print("Data Sd: " + str(sd))
        print("Node Number: " + str(self.maxNodes))
        print("Node Feature Dimension: " + str(self.graphList[0].nodeFeatureDim))
        print("Edge Feature Dimension: " + str(self.graphList[0].edgeFeatureDim))
                
        return self.get_trainTestGraph(), self.maxNodes, self.graphList[0].nodeFeatureDim, self.graphList[0].edgeFeatureDim 
        
        
    def reset_splitting(self):
        super().reset_splitting()
        self.trainVar = 0.0
        self.testVar = 0.0
        return
    

    def __getMeanVar(self, graph_list):
        if len(graph_list) == 0: return 0.0, 0.0;
        
        mean = 0.0
        variance = 0.0
        for g in graph_list:
            mean += g.label
        mean /= len(graph_list)
        
        for g in graph_list:
            variance += math.pow(g.label - mean, 2.0)
        variance /= len(graph_list)
        return mean, variance



class BinaryDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.positive = 0
        self.negative = 0
        return
    
    
    def load_csv(self, file = '', col_1 = 'smiles', col_2 = 'label'):
        self.__init__()
        super().load_csv(file, col_1, col_2)
        
        for g in self.graphList:
            if g.label == 1.0: self.positive += 1
            else: self.negative += 1
            
        print("Number of Positive Samples: " + str(self.positive))
        print("Number of Negative Samples: " + str(self.negative))
        
        return self.maxNodes, self.graphList[0].nodeFeatureDim, self.graphList[0].edgeFeatureDim
    
    
    def export_data(self, file):
        with open(file, 'w', newline='') as csvfile:
            fieldnames = ['Index', 'Smiles', 'Label', 'In Train']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(len(self.graphList)):
                graph = self.graphList[i]
                writer.writerow({'Index': i, 'Smiles': graph.smiles,
                                 'Label': graph.label,
                                 'In Train': i in self.trainInd})
    
        print("Processed data is exported.")
        return
        
    
    def import_csv_data(self, file):
        print("Load Processed Data")
        self.__init__()
        self.split = True
        
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mol = rf.MolFromSmiles(row['Smiles'])
                graph = GraphBuilder.molToGraph(mol)
                graph.smiles = row['Smiles']
                graph.label = float(row['Label'])
                self.graphList.append(graph)
                
                if graph.label == 1.0: self.positive += 1
                else: self.negative += 1
                
                if graph.nodeNum > self.maxNodes:
                    self.maxNodes = graph.nodeNum
                    
                if row['In Train'] == 'True':
                    self.trainInd.append(int(row['Index']))
                else: self.testInd.append(int(row['Index']))
                
        print("Number of Samples in the DataSet: " + str(len(self.graphList)))
        print("Node Number: " + str(self.maxNodes))
        print("Node Feature Dimension: " + str(self.graphList[0].nodeFeatureDim))
        print("Edge Feature Dimension: " + str(self.graphList[0].edgeFeatureDim))
        print("Number of Positive Samples: " + str(self.positive))
        print("Number of Negative Samples: " + str(self.negative))
                
        return super().get_trainTestGraph(), self.maxNodes, self.graphList[0].nodeFeatureDim, self.graphList[0].edgeFeatureDim 
        