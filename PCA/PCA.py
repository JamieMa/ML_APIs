# -*- coding: utf-8 -*-
"""
@author: Junqing Ma
"""
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



def my_PCA(my_data,mean_flag,normalize_flag):
    '''
    :type my_data: 2D np array,each row is a record
          mean_flag: Boolean
          normalize_flag:Boolean        
    :rtype: dictionary
    
    '''
    #-----I. calculate the variances of each dimension
    variances = np.var(my_data,axis=0,ddof=1)
    #-----II. PCA
    #----- mean centered the data-----#
    #1.get mean of each column,substract the mean for each column
    if mean_flag:
        my_data_mean = my_data.mean(axis=0)
        dataMeanCentered = my_data-my_data_mean
    else:
        dataMeanCentered = my_data
        
    #----- normalize the data-----#
    if normalize_flag==1: # normalize the data by standard deviation
        dataForPca = dataMeanCentered/np.std(dataMeanCentered,axis=0,ddof=1)
    else:
        dataForPca = dataMeanCentered
    
    dataForPca_var = np.var(dataForPca,axis=0,ddof=1)       
    #2 calculate Cov(X)
    my_data_Cov = np.cov(dataForPca,rowvar=False)
    
    #3 calculate the eigenvalue/eigenvector of Cov
    eigenValues,eigenVectors = LA.eig(my_data_Cov) # w:eigenvalue; v:eigenvector
    
    #4 sort the eigenvector according to the order of eigenvalue
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors= eigenVectors[:,idx] # eigenvector per column
    
    #5 projection
    pcaScores = np.dot(dataForPca,eigenVectors)
    
    #6 collect PCA results
    pcaResults = {'original_data':my_data,
                  'original_data_var':variances,
                   'mean_centered_data': dataMeanCentered,
                   'dataForPca': dataForPca,
                   'dataForPca_var':dataForPca_var,
                   'PC_variance': eigenValues,
                   'loadings': eigenVectors,
                   'scores': pcaScores}
    
    return pcaResults

if __name__ == '__main__':
    # read the raw data
    in_file_name = "SCLC_study_output_filtered.csv"
    dataIn = pd.read_csv(in_file_name)
    dataIn = dataIn.drop(dataIn.columns[0], axis=1)
    my_data = np.copy(dataIn)
    # PCA
    myPCAResults=my_PCA(my_data,mean_flag=1,normalize_flag=0)
    #total variance of original variables
    var_original_variables = sum(myPCAResults['original_data_var'])
    #total variance of PCS
    var_PCs_variables = sum(myPCAResults['PC_variance'])   
    # covariance between PC1 and PC2  
    cov_PCs = np.cov(myPCAResults['scores'],rowvar=False)
    cov_PC1_PC2 = cov_PCs[0,1]
    
    
    #screen plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('PC')
    ax1.set_ylabel(u'\u03BB')
    ax1.set_title('screen plot')
    x = range(1,len(myPCAResults['PC_variance'])+1)
    y = myPCAResults['PC_variance']
    ax1.scatter(x,y)  
    plt.xticks(np.arange(min(x), max(x)+1, 1.0)) # set x_ticks
    plt.show()
    
    #score plot   
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('score plot')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2') 
    
    x1 = myPCAResults['scores'][:20,0]
    y1 = myPCAResults['scores'][:20,1]   
    x2 = myPCAResults['scores'][21:,0]
    y2 = myPCAResults['scores'][21:,1]   
    ax1.scatter(x1,y1,color='red')
    ax1.scatter(x2,y2,color='blue')
    plt.show()
    
    # percentage of PC1
    percentVarianceExplained = myPCAResults['PC_variance'][0]/sum(myPCAResults['PC_variance'])*1.0
    print "PC1 explains: "+"%.2f%%"%percentVarianceExplained+" variance\n"
    # percentage of PC1+PC2
    percentVarianceExplained = (myPCAResults['PC_variance'][0]+myPCAResults['PC_variance'][1])/sum(myPCAResults['PC_variance'])*1.0
    print "PC1+PC2 explains: "+"%.2f%%"%percentVarianceExplained+" variance\n"
    # percentage of PC1+PC2+pc3
    percentVarianceExplained = (myPCAResults['PC_variance'][0]+myPCAResults['PC_variance'][1]+myPCAResults['PC_variance'][2])/sum(myPCAResults['PC_variance'])*1.0
    print "PC1+PC2+pc3 explains: "+"%.2f%%"%percentVarianceExplained+" variance\n"
    # percentage of PC1+PC2+pc3+pc4
    percentVarianceExplained = (myPCAResults['PC_variance'][0]+myPCAResults['PC_variance'][1]+myPCAResults['PC_variance'][2]+myPCAResults['PC_variance'][3])/sum(myPCAResults['PC_variance'])*1.0
    print "PC1+PC2+pc3+pc4 explains: "+"%.2f%%"%percentVarianceExplained+" variance\n"
    
    #loading plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('loading plot')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
   
    x = myPCAResults['loadings'][:,0]
    y = myPCAResults['loadings'][:,1]    
    ax1.scatter(x,y)  
    plt.show()
    
    #6 PCA with standard
    myPCAResults_standard=my_PCA(my_data,mean_flag=1,normalize_flag=1)
    #total variance of PCS
    var_PCs_variables_standard = sum(myPCAResults_standard['dataForPca_var']) 
    data_for_pca = myPCAResults_standard['dataForPca']
    data_for_pca.std(axis=0)
    
    
    
    
    
    
    
    
    
    
    