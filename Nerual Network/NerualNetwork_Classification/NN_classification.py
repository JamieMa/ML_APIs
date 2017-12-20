# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:48:13 2017

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

class NN_classification(object):
    '''
    This nerual network is fixed to have one input layer, one hidden layer and one output layer
    The input layer has two units, the hidden layer has two units and the output layer has one unit.
    '''
    def __init__(self):
        self.theta_1 = None
        self.theta_2 = None
    
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    
    def fit(self,x,y,alpha=0.1,max_iter=1000):
        m = x.shape[1]
        n = x.shape[0]
        bias_x =np.ones((1,m))
        x = np.concatenate((bias_x,x.reshape((n,m))),axis=0)
    
        theta_1 = np.random.random_sample(size=(2, 3))
        theta_2 = np.random.random_sample(size=(1, 3))
    
        alpha=1
        max_iter=1000
        E_total_history = np.zeros(max_iter)
        theta10_1_history = np.zeros(max_iter)
        theta11_1_history = np.zeros(max_iter)
        theta12_1_history = np.zeros(max_iter)
        theta20_1_history = np.zeros(max_iter)
        theta21_1_history = np.zeros(max_iter)
        theta22_1_history = np.zeros(max_iter)
        
        theta10_2_history = np.zeros(max_iter)
        theta11_2_history = np.zeros(max_iter)
        theta12_2_history = np.zeros(max_iter)
        
        for i in range(max_iter):
            #----forward propagation----#
            z1_2 = np.dot(theta_1[0],x)
            z2_2 = np.dot(theta_1[1],x)
            a1_2 = np.array([self.sigmoid(e) for e in z1_2])
            a2_2 = np.array([self.sigmoid(e) for e in z2_2])
            a0_1 =np.ones((1,m))
            a_2 = np.concatenate((a0_1,a1_2.reshape((1,m)),a2_2.reshape((1,m))),axis=0)
            z1_3 = np.dot(theta_2[0],a_2)
            a1_3 = np.array([self.sigmoid(e) for e in z1_3])
            
            E_total_history[i] = -np.mean(y*np.log(a1_3)+(1-y)*np.log(1-a1_3))
            theta10_1_history[i] = theta_1[0,0]
            theta11_1_history[i] = theta_1[0,1]
            theta12_1_history[i] = theta_1[0,2]
            theta20_1_history[i] = theta_1[1,0]
            theta21_1_history[i] = theta_1[1,1]
            theta22_1_history[i] = theta_1[1,2]   
            theta10_2_history[i] = theta_2[0,0]
            theta11_2_history[i] = theta_2[0,1]
            theta12_2_history[i] = theta_2[0,2]
            
            #----backward propatation----#
            #partial derivative(pd) of theta
            pd_theta10_2 = -np.mean((y-a1_3)*a_2[0,:])
            pd_theta11_2 = -np.mean((y-a1_3)*a_2[1,:])
            pd_theta12_2 = -np.mean((y-a1_3)*a_2[2,:])
            
            pd_theta10_1 = -np.mean((y-a1_3)*theta_2[0,1]*(a1_2*(1-a1_2))*x[0,:])
            pd_theta11_1 = -np.mean((y-a1_3)*theta_2[0,1]*(a1_2*(1-a1_2))*x[1,:])
            pd_theta12_1 = -np.mean((y-a1_3)*theta_2[0,1]*(a1_2*(1-a1_2))*x[2,:])
            
            pd_theta20_1 = -np.mean((y-a1_3)*theta_2[0,2]*(a1_2*(1-a1_2))*x[0,:])
            pd_theta21_1 = -np.mean((y-a1_3)*theta_2[0,2]*(a1_2*(1-a1_2))*x[1,:])
            pd_theta22_1 = -np.mean((y-a1_3)*theta_2[0,2]*(a1_2*(1-a1_2))*x[2,:])
            
            #update theta
            theta_1 -= alpha*np.array([[pd_theta10_1,pd_theta11_1,pd_theta12_1],[pd_theta20_1,pd_theta21_1,pd_theta22_1]])
            theta_2 -= alpha*np.array([[pd_theta10_2,pd_theta11_2,pd_theta12_2]])
         
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        
        iteration = range(max_iter)  
        res = {"E_total_history":E_total_history,
               "theta10_1_history":theta10_1_history,
               "theta11_1_history":theta11_1_history,
               "theta12_1_history":theta12_1_history,
               "theta20_1_history":theta20_1_history,
               "theta21_1_history":theta21_1_history,
               "theta22_1_history":theta22_1_history,
               "theta10_2_history":theta10_2_history,
               "theta11_2_history":theta11_2_history,
               "theta12_2_history":theta12_2_history,
               "iteration":iteration,
               }
        return res
    
    def predict(self,x):
        bias_x =np.array([1]) 
        x = np.concatenate((bias_x.reshape((1,1)),x.reshape((len(x),1))),axis=0)    
        z1_2 = np.dot(self.theta_1[0],x)
        z2_2 = np.dot(self.theta_1[1],x)
        a1_2 = self.sigmoid(z1_2)
        a2_2 = self.sigmoid(z2_2)
        a_2 = np.array([1,a1_2,a2_2])
        z1_3 = np.dot(self.theta_2[0],a_2)
        a1_3 = self.sigmoid(z1_3)       
        return 1 if a1_3>0.5 else 0


if __name__=="__main__":
    iris = load_iris()
    data = iris.data
    label = iris.target
    np.random.seed(9)
    x1 = data[50:,2]
    x2 = data[50:,3]    
    x1 = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
    x2 = (x2-np.min(x2))/(np.max(x2)-np.min(x2))
    x = np.concatenate((x1.reshape((1,len(x1))),x2.reshape(1,len(x2))),axis=0)
    
    #leave-one-out analysis
    y = np.array([0 if e==1 else 1 for e in label[50:]])
    error = 0
    for i in range(100):   
        idx = i
        print "the %d th iteration,select the %d th data as test data "%(i,idx)+"\n"
        test_idx = np.array([idx])
        test_x = x[:,test_idx]
        test_y = y[test_idx]
        train_idx = np.setdiff1d(np.arange(100) , test_idx)
        train_x = x[:,train_idx]
        train_y = y[train_idx]
        A = NN_classification()
        res = A.fit(train_x,train_y,alpha=0.5,max_iter=100)
        predict_res = A.predict(test_x)
        if predict_res!=test_y:
            error+=1
            print "false prediction"
    error_rate = error/100.0
    
    print "average error rate is %0.3f"%error_rate
        