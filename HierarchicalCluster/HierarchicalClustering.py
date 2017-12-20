# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:31:43 2017

@author: mjq
"""

import numpy as np
import pandas as pd
from sklearn import datasets

import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy import linalg as LA
from sklearn.neighbors import DistanceMetric
from mpl_toolkits.mplot3d import Axes3D


def get_centroid(indexs,D):
    max_D_per_point =[None for i in range(len(indexs))]
    for i in range(len(indexs)):
        max_distance = None
        for j in range(len(indexs)):
            cur_distance = D[i,j]
            if max_distance is None or (cur_distance>max_distance):
                max_distance=cur_distance
        max_D_per_point[i]=max_distance
    index_center = indexs[np.argmin(max_D_per_point)]
    return index_center

def HierarchicalCluster(X,threshold,linkage='single'):
    '''
    :type X: 2D np array,each row is a record
          threshold: positive int
          linkage:'single' or 'complete' or 'average' or 'centroid'       
    :rtype: dictionary
    '''
    
    def DistanceBetweenClusters(indexs_i,indexs_j,linkage):
        res_distance=None
        if linkage=='single':
            for index_i in indexs_i:
                for index_j in indexs_j:
                    cur_distance = pairwise_dist[index_i,index_j]
                    if res_distance is None or cur_distance<res_distance:
                        res_distance=cur_distance
        elif linkage=='complete':
            for index_i in indexs_i:
                for index_j in indexs_j:
                    cur_distance = pairwise_dist[index_i,index_j]
                    if res_distance is None or cur_distance>res_distance:
                        res_distance=cur_distance
        elif linkage=='average':
            sum_distance = 0
            for index_i in indexs_i:
                for index_j in indexs_j:
                    sum_distance += pairwise_dist[index_i,index_j]
                    res_distance=sum_distance/(len(indexs_i)*len(indexs_j))
        else:#centroid
            centroid_i = get_centroid(indexs_i,pairwise_dist)
            centroid_j = get_centroid(indexs_j,pairwise_dist)
            res_distance = pairwise_dist[centroid_i,centroid_j]   
        return res_distance
            
            
    dist = DistanceMetric.get_metric('euclidean')
    pairwise_dist = dist.pairwise(X)
    indexs = np.array([i for i in range(len(X))])
    labels=np.array([i for i in range(len(X))])
    step=0
    while len(np.unique(labels))>1:
        step+=1
        num_cluster=len(np.unique(labels))
        print "The %d's iterative with %d clusters"%(step,num_cluster)
        min_distance = None
        clusters2merge = None
        for i in range(num_cluster):
            for j in range(i+1,num_cluster):
                II_i = (labels==i)
                indexs_i = indexs[II_i]
                II_j = (labels==j)               
                indexs_j = indexs[II_j]
                cur_distance = DistanceBetweenClusters(indexs_i,indexs_j,linkage)
                if min_distance is None or cur_distance<min_distance:
                    min_distance=cur_distance
                    clusters2merge = [i,j]
        if min_distance <= threshold:
            #merge two cluster
            for i in range(len(labels)):
                if labels[i]==clusters2merge[1]:
                    labels[i]=clusters2merge[0]
                elif labels[i]>clusters2merge[1]:
                    labels[i]-=1
            print "In iteration %d, min_distance is %f"%(step, min_distance)  
        else:
            print "Exceed threshold, min_distance=%f"%(min_distance)
            break
    
    result = {'num_cluster_':num_cluster,
              "labels_":labels}
    return result
        
if __name__ == '__main__':
    np.random.seed(5)
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    y = iris.target
    target_names = iris.target_names
   
    myResults = HierarchicalCluster(X,threshold=2,linkage='centroid')
    labels = myResults['labels_']
    num_cluster = myResults['num_cluster_']
    print "The number of clusters in the result is %d"%num_cluster
    
    #plot 2D
    plt.figure(1, figsize=(8, 6))
    for cluster in range(np.unique(labels).shape[0]):
        II = labels==cluster
        plt.scatter(X[II,0],X[II,1],label=cluster)
    plt.title('2D visualization')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend() 
    
    #plot 3D
    fig = plt.figure(2, figsize=(10, 8))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12
    for cluster in range(np.unique(labels).shape[0]):
        II = labels==cluster
        ax.scatter(X[II, 3], X[II, 0], X[II, 2], label=cluster)
    plt.title('3D visualization')
    plt.legend() 
    plt.show()
               
            
        