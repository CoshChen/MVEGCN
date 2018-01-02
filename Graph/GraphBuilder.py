# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:36:05 2017

@author: Ko-Shin Chen
"""
import numpy as np
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges

class Graph():
    def __init__(self):
        self.nodes = []
        self.nodeNum = 0
        self.edges = []
        self.label = 0.0
        self.scaled_label = 0.0
        self.nodeFeatureDim = 0
        self.edgeFeatureDim = 0
        self.smiles = None
        return
    
    def getNodeFeatureMatrix(self, maxNodes):
        '''
        Returns a 2D matrix.
        The i-th row is the feature vector of the node i.
        '''
        matrix = np.vstack([node.features for node in self.nodes])
        return np.pad(matrix, ((0, maxNodes - self.nodeNum),(0,0)), 'constant')
    
    def getIdPlusAdjMatrix(self, maxNodes):
        matrix = np.zeros([maxNodes, maxNodes])
        for edge in self.edges:
            (i,j) = edge.ends
            matrix[i,j] = 1.0
            matrix[j,i] = 1.0
            
        for i in range(maxNodes):
            matrix[i,i] += 1.0
            
        return matrix
        
    
    def getWeightedAdjMatrix(self, maxNodes):
        adjMatrix = np.zeros([maxNodes, maxNodes])
        for edge in self.edges:
            (i,j) = edge.ends
            adjMatrix[i,j] = edge.weight
            adjMatrix[j,i] = edge.weight
            
        return adjMatrix 
    
    def getLaplacian(self, maxNodes):
        adjMatrix = self.getWeightedAdjMatrix(maxNodes)
        degrees = np.array([np.sum(row) for row in adjMatrix])
        return np.diag(degrees) - adjMatrix
    
    def getAdjTensor(self, maxNodes):
        adjTensor = np.zeros([maxNodes, maxNodes, self.edgeFeatureDim + 1])
        for edge in self.edges:
            (i,j) = edge.ends
            adjTensor[i,j,0] = 1.0
            adjTensor[j,i,0] = 1.0
            adjTensor[i,j,1:] = edge.features
            adjTensor[j,i,1:] = edge.features
        return adjTensor
    

class Node():
    def __init__(self, id = None, features = np.array([])):
        self.id = id
        self.features = features # 1D array (vector)
        self.neighbors = [] # (neighborNode id, edge id)
        return


class Edge():
    def __init__(self, id = None, ends = (), features = np.array([])):
        self.id = id
        self.ends = ends
        self.features = features # 1D array (vector)
        self.weight = 1.0
        return
    
    def setWeight(self, newWeight):
        self.weight = newWeight
        return


def molToGraph(rdmol):
    '''
    Converts an RDKit molecule to an attributed undirected graph
    @param rdmol: RDKit molecule
    @return: Graph
    '''
    graph = Graph()
    
    # Calculate atom-level molecule descriptors
    nodesFeatures = [[] for i in rdmol.GetAtoms()]
    
    #6 (25) Crippen contribution to logp
    [nodesFeatures[i].append(x[0]) \
     for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]  
    
    #7 (26) Crippen contribution to mr
    [nodesFeatures[i].append(x[1]) \
     for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
    
    #8 (27) TPSA contribution
    [nodesFeatures[i].append(x) \
     for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
    
    #9 (28) Labute ASA contribution
    [nodesFeatures[i].append(x) \
     for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
    
    #10 (29) EState Index
    [nodesFeatures[i].append(x) \
     for (i, x) in enumerate(EState.EStateIndices(rdmol))]
    
    # Calculate Gasteiger charges for features 30 and 31
    rdPartialCharges.ComputeGasteigerCharges(rdmol)
    # The computed charges are stored on each atom with computed property
    # under the name _GasteigerCharge and _GasteigerHCharge.
    # Values could be NaN.
    
    #11 (30)
    for (i,a) in enumerate(rdmol.GetAtoms()):
        if np.isnan(float(a.GetProp('_GasteigerCharge'))) or np.isinf(float(a.GetProp('_GasteigerCharge'))):
            nodesFeatures[i].append(0.0)
        else:
            nodesFeatures[i].append(float(a.GetProp('_GasteigerCharge')))
            
    #12 (31)
    for (i,a) in enumerate(rdmol.GetAtoms()):
        if np.isnan(float(a.GetProp('_GasteigerHCharge'))) or np.isinf(float(a.GetProp('_GasteigerHCharge'))):
            nodesFeatures[i].append(0.0)
        else:
            nodesFeatures[i].append(float(a.GetProp('_GasteigerHCharge')))
            
    # Add edges to graph
    for bond in rdmol.GetBonds():
        edge = Edge()
        edge.id = bond.GetIdx()
        edge.features = getBondFeatures(bond).astype('float32')
        edge.ends = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        graph.edges.append(edge)
        
    # Add nodes to graph
    for i, atom in enumerate(rdmol.GetAtoms()):
        node = Node()
        node.id = atom.GetIdx()
        node.features = getAtomFeatures(atom, nodesFeatures[i])
        
        for neighbor in atom.GetNeighbors():
            node.neighbors.append((
                    neighbor.GetIdx(),
                    rdmol.GetBondBetweenAtoms(
                            atom.GetIdx(),
                            neighbor.GetIdx()
                    ).GetIdx()
            ))
            
        graph.nodes.append(node)
	
    graph.nodeNum = len(graph.nodes)
    graph.nodeFeatureDim = len(graph.nodes[0].features)
    if(len(graph.edges) > 0): graph.edgeFeatureDim = len(graph.edges[0].features)
    
    return graph


def getBondFeatures(bond):
    '''
    @param bond: a bond in RDKit molecule
    @return: numpy array (1D)
    '''  
    features = []
    
    # bond type
    features += oneHotVector(
		bond.GetBondTypeAsDouble(),
		[1.0, 1.5, 2.0, 3.0])
    
    # boolean: is aromatic
    features.append(bond.GetIsAromatic())
    # boolean: is conjugated
    features.append(bond.GetIsConjugated())
    # boolean: is part of ring
    features.append(bond.IsInRing())
    return np.array(features)


def getAtomFeatures(atom, existingFeatures):
    '''
    Add additional features infront of existingFeatures.
    @param atom: an atom in RDKit molecule
    @param existingFeatures: list
    @return: numpy array (1D)
    
    ''' 
    featuresToAdd = []
    
    #0 (0-10): Atomic id
    featuresToAdd += oneHotVector(
		atom.GetAtomicNum(), 
		[5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    
    #1 (11-16): number of heavy neighbors
    featuresToAdd += oneHotVector(
		len(atom.GetNeighbors()),
		[0, 1, 2, 3, 4, 5])
    
    #2 (17-21): number of hydrogens
    featuresToAdd += oneHotVector(
		atom.GetTotalNumHs(),
		[0, 1, 2, 3, 4])
    
    #3 (22): formal charge
    featuresToAdd.append(atom.GetFormalCharge())   
    #4 (23): boolean if in ring
    featuresToAdd.append(atom.IsInRing())   
    #5 (24): boolean if aromatic atom
    featuresToAdd.append(atom.GetIsAromatic())   
    featuresToAdd += existingFeatures
    return np.array(featuresToAdd)

def oneHotVector(val, lst):
	'''
    Converts a value to a one-hot vector based on options in lst
    '''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)