# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:48:15 2017

@author: mjq
"""
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.stats as stats

def my_lm(x,y,max_iter=1000,alpha=0.1):
    '''
    :type x: 1D np array
          y: 1D np array
      max_iter:positive int
      alpha: double
    :rtype: dictionary
    '''
    m = len(x)   
    # To get theta_hat using gradient descent
    theta = np.zeros((max_iter, 2))
    theta[0, :] = np.array([2.5, 2.5])   
    J = np.zeros(max_iter)
    
    #J(theta)= 1/(2*m)*sum((y_hat-y)^2)
    #partial_derivative_theta_0 = 1/m*sum(y_hat-y)
    #partial_derivative_theta_1 = 1/m*sum[(y_hat-y)*x]
    for i in range(max_iter):
        y_hat = theta[i,0]+theta[i,1]*x
        residual = y_hat-y
        J[i] = 0.5 / m * np.sum(residual**2)
        if i<(max_iter-1):
            partial_derivative = np.array([0.0, 0.0])            
            partial_derivative[0] = 1.0/m*np.sum(residual)
            partial_derivative[1] = 1.0/m*np.sum(residual*x)
            theta[i+1, 0] = theta[i, 0] - alpha * partial_derivative[0] / m
            theta[i+1, 1] = theta[i, 1] - alpha * partial_derivative[1] / m
    
    theta_hat = theta[-1]
    y_hat =theta_hat[0] + theta_hat[1]*x   

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    
    sigma_hat = np.sqrt(np.sum((y - y_hat)**2) / (m-2))
    # R2
    # total sum of squares
    SS_total = np.sum((y - y_bar)**2)
    # regression sum of squares
    SS_reg = np.sum((y_hat - y_bar)**2)
    #residual sum of squares
    SS_err = np.sum((y - y_hat)**2)
    R2 = SS_reg / SS_total
    
    #F-statistic
    MS_total = SS_total/(m-1)
    MS_reg = SS_reg / 1.0
    MS_err = SS_err / (m-2)
    F=MS_reg/MS_err
    F_test_p_value = 1 - stats.f._cdf(F, dfn=1, dfd=m-2)
    
    #theta statistic
    #--theta_1
    theta_1_hat_var = SS_err / ((m-1) * np.var(x))
    theta_1_hat_sd = np.sqrt(theta_1_hat_var)
    # confidence interval
    z = stats.t.ppf(q=0.975, df=m-2)
    print z
    theta_1_hat_CI_lower_bound = theta_hat[1] - z * theta_1_hat_sd
    theta_1_hat_CI_upper_bound = theta_hat[1] + z * theta_1_hat_sd
    # hypothesis tests for beta_1_hat
    # H0: theta_1 = 0
    # H1: theta_1 != 0
    theta_1_hat_t_statistic = theta_hat[1] / theta_1_hat_sd
    theta_1_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(theta_1_hat_t_statistic), df=m-2))
    
    #--theta_0
    theta_0_hat_var = theta_1_hat_var * np.sum(x**2) / m
    theta_0_hat_sd = np.sqrt(theta_0_hat_var)
    # confidence interval
    theta_0_hat_CI_lower_bound = theta_hat[0] - z * theta_0_hat_sd
    theta_0_hat_CI_upper_bound = theta_hat[0] + z * theta_0_hat_sd
    theta_0_hat_t_statistic = theta_hat[0] / theta_0_hat_sd
    theta_0_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(theta_0_hat_t_statistic), df=m-2))
    
    # confidence interval for the regression line
    sigma_i = 1.0/m * (1 + ((x - x_bar) / np.std(x))**2)
    y_hat_sd = sigma_hat * sigma_i

    y_hat_CI_lower_bound = y_hat - z * y_hat_sd
    y_hat_CI_upper_bound = y_hat + z * y_hat_sd
       
    lmResults = {'x':x,
                 'y':y,
                 'J': J,
                 'theta': theta,
                 'final_theta':theta_hat,
                 'y_hat':y_hat,
                 'R2':R2,
                 'F_test_p_value':F_test_p_value,
                 'theta_0_hat_CI':(theta_0_hat_CI_lower_bound,theta_0_hat_CI_upper_bound),
                 'theta_1_hat_CI':(theta_1_hat_CI_lower_bound,theta_1_hat_CI_upper_bound),
                 "theta_0_hat_t_test_p_value":theta_0_hat_t_test_p_value,
                 "theta_1_hat_t_test_p_value":theta_1_hat_t_test_p_value,
                 "y_hat_CI":(y_hat_CI_lower_bound,y_hat_CI_upper_bound)
                 
                 }
    return lmResults
        
        
        
if __name__ == '__main__':
    # read the raw data
    in_file_name = "linear_regression_test_data.csv"
    dataIn = pd.read_csv(in_file_name,delimiter=',',index_col=0)
    dataIn = dataIn.values
    x=dataIn[:,0]
    y=dataIn[:,1]
    y_theroretical = dataIn[:,2]
    #linear regression
    lm_result = my_lm(x,y)
    y_hat = lm_result['y_hat']
    
    print "R2 is %.5f"%lm_result['R2']
    print "theta_hat_0 is %.5f,theta_hat_1 is %.5f"%(lm_result['final_theta'][0],lm_result['final_theta'][1])
    print "F_test_p_value is %.10f"%lm_result['F_test_p_value']
    print "theta_0_hat_confidence interval is [%f,%f]"%(lm_result['theta_0_hat_CI'][0],lm_result['theta_0_hat_CI'][1])
    print "theta_1_hat_confidence interval is [%f,%f]"%(lm_result['theta_1_hat_CI'][0],lm_result['theta_1_hat_CI'][1])
    print "theta_0_hat_t_test_p_value is %f"%lm_result['theta_0_hat_t_test_p_value']
    print "theta_1_hat_t_test_p_value is %f"%lm_result['theta_1_hat_t_test_p_value']
    
    #plot
    fig_width = 8
    fig_height = 6
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y/y_theroretical')
    
    ax1.scatter(x,y,color='red',label='y')
    ax1.scatter(x,y_theroretical,color='blue',label='y_theroretical')
    ax1.plot(x,y_hat,label='lm')
    plt.legend()
    
    plt.show()
        
        
    