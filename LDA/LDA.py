# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 22:03:08 2017

@author: mjq
"""
import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import linalg as LA

class LDA(object):
    @staticmethod
    def my_LDA_twoClass(x1,x2):
        '''
        x1:n1*m1 np.array, contain n1 m1-dimension data
        x2:n2*m2 np.array, contain n2 m2-dimension data
        '''
        d = x1.shape[1]
        mu1 = np.mean(x1, axis=0)
        mu2 = np.mean(x2, axis=0) 
        s1 = np.dot(np.transpose(x1-mu1),x1-mu1)
        s2 = np.dot(np.transpose(x2-mu2),x2-mu2)
        s_within = s1+s2
        s_between = np.dot(np.reshape(mu1-mu2,(d,1)),np.reshape(mu1-mu2,(1,d)))
        W = np.dot(inv(s_within),mu1-mu2)
        W = W/np.linalg.norm(W)
        theta = math.atan(W[1] / W[0])
        mu1_tilde = np.dot(W.transpose(), mu1)
        mu2_tilde = np.dot(W.transpose(), mu2)
        projection_1 = np.dot(W.transpose(), x1.transpose())
        projection_2 = np.dot(W.transpose(), x2.transpose()) 
        
        res = {"W":W,
               "slope_of_W":theta,
               "mu1_tilde":mu1_tilde,
               "mu2_tilde":mu2_tilde,
               "projection_1":projection_1,
               "projection_2":projection_2}
        return res
    
    @staticmethod
    def my_LDA_multiClass(data,label,n_components=1):
        '''
        data:2D np.array, with each row represent a record
        label: class label corresponding to each record
        '''
        d = len(data[0])
        s_within = np.zeros((d,d))
        s_between = np.zeros((d,d))
        unique_label = np.unique(label)
        k = len(unique_label)
        if n_components>k-1:
            print "n_components should not larger than number of class"
            assert False
        mu = np.mean(data, axis=0)
        for c in unique_label:
            II_c = np.where(label==c)[0]
            n_c = len(II_c)
            X_c = data[II_c]
            mu_c = np.mean(X_c, axis=0)
            s_c = np.dot(np.transpose(X_c-mu_c),X_c-mu_c)
            s_within += s_c
            s_between += n_c*np.dot(np.reshape(mu_c-mu,(d,1)),np.reshape(mu_c-mu,(1,d)))
            
        eigenValues,eigenVectors = LA.eig(np.dot(inv(s_within),s_between))
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors= eigenVectors[:,idx] # eigenvector per column
        W = eigenVectors[:n_components]
        eigenValues = eigenValues[:n_components]
        projection = np.dot(W, data.transpose())
        projection = projection.transpose()
        
        res = {"W":W,
               "eigenValues":eigenValues,
               "projection":projection
               }
        
        return res
        
            
if __name__=="__main__":
    
    
    # -----------------------------------------------------
    # 1. Do LDA on toy data
    # -----------------------------------------------------
    #original data
    x1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
    x2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])   
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)    
    ax.set_title('Apply LDA to a toy dataset')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    
    ax.scatter(x1[:, 0], x1[:, 1], color='blue')
    ax.scatter(x2[:, 0], x2[:, 1], color='red')
      
    #LDA
    res = LDA.my_LDA_twoClass(x1,x2)
    W = res["W"]
    theta = res["slope_of_W"]
    projection_1 = res["projection_1"]
    projection_2 = res["projection_2"]
    W_scaled = W * 14.0 *(-1)
    ax.plot([0, W_scaled[0]], [0, W_scaled[1]], color='green')
    ax.scatter(-projection_1*math.cos(theta), -projection_1*math.sin(theta), color='blue', marker='x')
    ax.scatter(-projection_2*math.cos(theta), -projection_2*math.sin(theta), color='red', marker='x')
    fig.show()
    
    # -----------------------------------------------------
    # 2. Do LDA on cell line data
    # -----------------------------------------------------
    in_file_name = "SCLC_study_output_filtered.csv"
    dataIn = pd.read_csv(in_file_name)
    X = dataIn.drop(dataIn.columns[0], axis=1)
    x1 = X[:20]
    x2 = X[20:]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)    
    ax.set_title('Apply LDA to cell line data')
    ax.set_xlabel('projection')
    ax.set_ylabel('')
      
    #LDA
    res = LDA.my_LDA_twoClass(x1,x2)
    W = res["W"]
    theta = res["slope_of_W"]
    mu1_tilde = res["mu1_tilde"]
    mu2_tilde = res["mu2_tilde"]
    projection_1 = res["projection_1"]
    projection_2 = res["projection_2"] 
    
    ax.plot(projection_1, np.zeros(20), linestyle='None', marker='o', markersize=2, color='blue', label='NSCLC')
    ax.plot(projection_2, np.zeros(20), linestyle='None', marker='o', markersize=2, color='red', label='SCLC')
    ax.plot(mu1_tilde, 0.0, linestyle='None', marker='*', markersize=10, color='magenta',label='mu1_tilde')
    ax.plot(mu2_tilde, 0.0, linestyle='None', marker='*', markersize=10, color='green',label='mu2_tilde')
    ax.legend()
    fig.show()
    
    #--------compared with sklearn LDA
    # apply sklearn LDA to cell line data
    sklearn_LDA = LinearDiscriminantAnalysis()
    y = np.concatenate((np.zeros(20), np.ones(20)))
    II_0 = np.where(y==0)[0]
    II_1 = np.where(y==1)[0]
    sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
    
    sklearn_LDA_projection = -sklearn_LDA_projection    
    # plot the projections
    fig = plt.figure()   
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Results from applying sklearn LDA to cell line data')
    ax.set_xlabel('projection')
    ax.set_ylabel('')
    ax.plot(sklearn_LDA_projection[II_0], np.zeros(len(II_0)), linestyle='None', marker='o', markersize=2, color='blue', label='NSCLC')
    ax.plot(sklearn_LDA_projection[II_1], np.zeros(len(II_1)), linestyle='None', marker='o', markersize=2, color='red', label='SCLC')
    ax.legend()
    
    fig.show()

	# -----------------------------------------------------
    # 3. Do LDA on iris
    # -----------------------------------------------------
    iris = load_iris()
    data = iris.data
    label = iris.target
    res = LDA.my_LDA_multiClass(data,label,n_components=2)
    W = res["W"]
    eigenValues = res["eigenValues"]
    projection = res["projection"]
   
    # plot the projections
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Results from applying LDA to iris')
    ax.set_xlabel(r'$W_1$')
    ax.set_ylabel(r'$W_2$')
    ax.plot(projection[0:50, 0], projection[0:50, 1], linestyle='None', marker='o', markersize=2, color='blue', label='setosa')
    ax.plot(projection[50:100, 0], projection[50:100, 1], linestyle='None', marker='o', markersize=2, color='red', label='versicolor')
    ax.plot(projection[100:150, 0], projection[100:150, 1], linestyle='None', marker='o', markersize=2, color='green', label='setosa')
    ax.legend()
    fig.show()