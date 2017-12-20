# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 21:30:38 2017

@author: mjq
"""

import numpy as np
import pandas as pd
from sklearn import datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from numpy import linalg as LA

def kMeans(X,n_clusters=8,max_iter=300,minWCSS=10000):
    '''
    :type X: 2D np array,each row is a record
          n_clusters: positive int
          max_iter: positive int
          minWCSS:positive int 
    :rtype: dictionary
    
    '''
    
    if n_clusters>len(X):
        print "Number of cluster exceeds number of input records"
        print "Please select K again."
        return 
    
    def WCSS(X,centroids,cluster_result):
        sum_WCSS=0
        for k in range(len(centroids)):
            WCSS_cluster = 0
            II = (cluster_result==k)        
            for j in range(np.sum(II)):
                WCSS_cluster+=distance.euclidean(X[II][j],centroids[k])
            sum_WCSS+=WCSS_cluster
        return sum_WCSS
            
    #randomly select initial centroids 
    idx = np.random.choice([i for i in range(len(X))],size=n_clusters,replace=False)
    centroids = X[idx,:]
    cluster_result = np.zeros(len(X))
    pre_cluster_result=None
    
    i=0
    while i<=max_iter:    
        #calculate distance
        for j in range(len(X)):
            min_distance = distance.euclidean(X[j],centroids[0])
            num_cluster = 0
            for k in range(1,n_clusters):
                cur_distance=distance.euclidean(X[j],centroids[k])
                if cur_distance<min_distance:
                    min_distance=cur_distance
                    num_cluster=k
            cluster_result[j]=num_cluster
            
        #check if assignment no longer change
        print np.sum(pre_cluster_result==cluster_result)
        print np.all(pre_cluster_result==cluster_result)
        if pre_cluster_result is not None and np.all(pre_cluster_result==cluster_result):
            break
        
        #update centroids
        for k in range(n_clusters):
            II = (cluster_result==k)
            centroids[k]= np.mean(X[II],axis=0)
        #deep copy cluster_result to pre_cluster_result     
            pre_cluster_result = np.copy(cluster_result)
        i+=1
        cur_WCSS=WCSS(X,centroids,cluster_result)
        print "The %d's iterative with WCSS: %f "%(i,cur_WCSS)
            
    final_WCSS=WCSS(X,centroids,cluster_result)
    
            
    kmeans_result={"cluster_centers_":centroids,
                    "labels_":cluster_result,
                    "WCSS_":final_WCSS,
                    "max_iter_":i}       
    return kmeans_result     

if __name__ == '__main__':
    in_file_name = "SCLC_study_output_filtered.csv"
    dataIn = pd.read_csv(in_file_name)
    X = dataIn.drop(dataIn.columns[0], axis=1)
    k=2
    myKmeansResults = kMeans(X.values,n_clusters=k)
    labels=myKmeansResults['labels_']
    
      
        
    