# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 17:02:58 2017

@author: mjq
"""

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA

class NN_regression(object):
    '''
    This nerual network is fixed to have one input layer, one hidden layer and one output layer
    The input layer has two units, the hidden layer has two units and the output layer has two units.
    '''
    def __init__(self):
        self.theta_1 = None
        self.theta_2 = None
        
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    
    def fit(self,x,y,alpha,max_iter=1000):
        
        
        bias_x =np.array([1]) 
        x = np.concatenate((bias_x,x),axis=0)    
        np.random.seed(1)
        theta_1 = np.random.random_sample(size=(2, 3))
        theta_2 = np.random.random_sample(size=(2, 3))
        
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
        theta20_2_history = np.zeros(max_iter)
        theta21_2_history = np.zeros(max_iter)
        theta22_2_history = np.zeros(max_iter)
        
        for i in range(max_iter):
            #----forward propagation----#
            
            z1_2 = np.dot(theta_1[0],x)
            z2_2 = np.dot(theta_1[1],x)
            a1_2 = self.sigmoid(z1_2)
            a2_2 = self.sigmoid(z2_2)
            a_2 = np.array([1,a1_2,a2_2])
            z1_3 = np.dot(theta_2[0],a_2)
            z2_3 = np.dot(theta_2[1],a_2)
            a1_3 = self.sigmoid(z1_3)
            a2_3 = self.sigmoid(z2_3)
            E_total_history[i] = 0.5*(y[0]-a1_3)**2+0.5*(y[1]-a2_3)**2
            theta10_1_history[i] = theta_1[0,0]
            theta11_1_history[i] = theta_1[0,1]
            theta12_1_history[i] = theta_1[0,2]
            theta20_1_history[i] = theta_1[1,0]
            theta21_1_history[i] = theta_1[1,1]
            theta22_1_history[i] = theta_1[1,2]
            
            theta10_2_history[i] = theta_2[0,0]
            theta11_2_history[i] = theta_2[0,1]
            theta12_2_history[i] = theta_2[0,2]
            theta20_2_history[i] = theta_2[1,0]
            theta21_2_history[i] = theta_2[1,1]
            theta22_2_history[i] = theta_2[1,2]
            
            #----backward propatation----#
            #partial derivative(pd) of theta
            pd_theta10_2 = (a1_3-y[0])*(a1_3*(1-a1_3))*a_2[0]
            pd_theta11_2 = (a1_3-y[0])*(a1_3*(1-a1_3))*a_2[1]
            pd_theta12_2 = (a1_3-y[0])*(a1_3*(1-a1_3))*a_2[2] 
            
            pd_theta20_2 = (a2_3-y[1])*(a2_3*(1-a2_3))*a_2[0]
            pd_theta21_2 = (a2_3-y[1])*(a2_3*(1-a2_3))*a_2[1]
            pd_theta22_2 = (a2_3-y[1])*(a2_3*(1-a2_3))*a_2[2]
            
            pd_E1_theta10_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,1]*(a1_2*(1-a1_2))*x[0]
            pd_E1_theta11_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,1]*(a1_2*(1-a1_2))*x[1]
            pd_E1_theta12_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,1]*(a1_2*(1-a1_2))*x[2]
            pd_E2_theta10_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,1]*(a1_2*(1-a1_2))*x[0]
            pd_E2_theta11_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,1]*(a1_2*(1-a1_2))*x[1]
            pd_E2_theta12_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,1]*(a1_2*(1-a1_2))*x[2]
            pd_theta10_1 = pd_E1_theta10_1+pd_E2_theta10_1
            pd_theta11_1 = pd_E1_theta11_1+pd_E2_theta11_1
            pd_theta12_1 = pd_E1_theta12_1+pd_E2_theta12_1
            
            pd_E1_theta20_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,2]*(a2_2*(1-a2_2))*x[0]
            pd_E1_theta21_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,2]*(a2_2*(1-a2_2))*x[1]
            pd_E1_theta22_1 = (a1_3-y[0])*(a1_3*(1-a1_3))*theta_2[0,2]*(a2_2*(1-a2_2))*x[2]
            pd_E2_theta20_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,2]*(a2_2*(1-a2_2))*x[0]
            pd_E2_theta21_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,2]*(a2_2*(1-a2_2))*x[1]
            pd_E2_theta22_1 = (a2_3-y[1])*(a2_3*(1-a2_3))*theta_2[1,2]*(a2_2*(1-a2_2))*x[2]
            pd_theta20_1 = pd_E1_theta20_1+pd_E2_theta20_1
            pd_theta21_1 = pd_E1_theta21_1+pd_E2_theta21_1
            pd_theta22_1 = pd_E1_theta22_1+pd_E2_theta22_1
            
            #update theta
            theta_1 -= alpha*np.array([[pd_theta10_1,pd_theta11_1,pd_theta12_1],[pd_theta20_1,pd_theta21_1,pd_theta22_1]])
            theta_2 -= alpha*np.array([[pd_theta10_2,pd_theta11_2,pd_theta12_2],[pd_theta20_2,pd_theta21_2,pd_theta22_2]])
            
        self.theta_1 =  theta_1
        self.theta_2 =  theta_2
        
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
               "theta20_2_history":theta20_2_history,
               "theta21_2_history":theta21_2_history,
               "theta22_2_history":theta22_2_history,
               "iteration":iteration,
               }
        return res
    
    def predict(self,x):   
        bias_x =np.array([1]) 
        x = np.concatenate((bias_x,x),axis=0)    
        z1_2 = np.dot(self.theta_1[0],x)
        z2_2 = np.dot(self.theta_1[1],x)
        a1_2 = self.sigmoid(z1_2)
        a2_2 = self.sigmoid(z2_2)
        a_2 = np.array([1,a1_2,a2_2])
        z1_3 = np.dot(self.theta_2[0],a_2)
        z2_3 = np.dot(self.theta_2[1],a_2)
        a1_3 = self.sigmoid(z1_3)
        a2_3 = self.sigmoid(z2_3)
        return a1_3,a2_3

    
if __name__=="__main__":
	
    x = np.array([0.05,0.1])
    y = np.array([0.01,0.99])
 
    max_iter = 1000
    alpha=0.1
    A = NN_regression()
    res = A.fit(x,y,alpha,max_iter)
    y_hat = A.predict(x)
    
    
    E_total_history = res["E_total_history"]
    theta10_1_history = res["theta10_1_history"]
    theta11_1_history = res["theta11_1_history"]
    theta12_1_history = res["theta12_1_history"]
    theta20_1_history = res["theta20_1_history"]
    theta21_1_history = res["theta21_1_history"]
    theta22_1_history = res["theta22_1_history"]
    
    theta10_2_history = res["theta10_2_history"]
    theta11_2_history = res["theta11_2_history"]
    theta12_2_history = res["theta12_2_history"]
    theta20_2_history = res["theta20_2_history"]
    theta21_2_history = res["theta21_2_history"]
    theta22_2_history = res["theta22_2_history"]
    iteration = res["iteration"]
    
    #plot total cost
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('E_total')
    ax1.set_xlabel('iteration')
    ax1.set_title('total cost vs iteration')
    
    ax1.plot(iteration, E_total_history, linestyle='None', marker='o', markersize=1, color='blue')
    plt.show()
    
    #plot theta10_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta10_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta10_1 vs iteration')
    
    ax1.plot(iteration, theta10_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta11_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta11_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta11_1 vs iteration') 
    ax1.plot(iteration, theta11_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta12_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta12_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta12_1 vs iteration') 
    ax1.plot(iteration, theta12_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta20_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta20_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta20_1 vs iteration') 
    ax1.plot(iteration, theta20_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta21_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta21_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta21_1 vs iteration') 
    ax1.plot(iteration, theta21_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta22_1 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta22_1')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta22_1 vs iteration') 
    ax1.plot(iteration, theta21_1_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta10_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta10_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta10_2 vs iteration') 
    ax1.plot(iteration, theta10_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta11_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta11_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta11_2 vs iteration') 
    ax1.plot(iteration, theta11_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta12_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta12_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta12_2 vs iteration') 
    ax1.plot(iteration, theta12_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta20_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta20_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta20_2 vs iteration') 
    ax1.plot(iteration, theta20_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta21_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta21_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta21_2 vs iteration') 
    ax1.plot(iteration, theta21_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()
    
    #plot theta22_2 history
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('theta22_2')
    ax1.set_xlabel('iteration')
    ax1.set_title('theta22_2 vs iteration') 
    ax1.plot(iteration, theta22_2_history, linestyle='None', marker='o', markersize=1, color='green')
    plt.show()