#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 01 13:50:17 2020

@author: BenjaminSuter
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.optimize import root
from IterCompRelEnt import MonteCarloENTROPY, LoadAndComputeEntropy
from surprise import Surprise


#Define the fixed parameters
mean_1 = np.zeros(2)
cov_1 = np.identity(2)

mean_q = np.zeros(2)
cov_q = np.identity(2)*10
icov_q = np.linalg.inv(cov_q)

F_0 = np.array([0,1,0,2])
M = np.matrix([[1,0],[0,2],[1,1],[1,1]])

cov_C = np.identity(4)
icov_C = np.linalg.inv(cov_C)


#Define all the distributions
posterior_1 = multivariate_normal(mean_1, cov_1)
prior_q = multivariate_normal(mean_q, cov_q)

Model = lambda theta: np.asarray(F_0 + M@theta)[0]

L_2 =  lambda theta: multivariate_normal(Model(theta), cov_C)

#Calculate posterior 2 : p_2
D = np.array([1.29469631, 1.06716929, 0.58396572, 1.26835481])
cov_2 = np.linalg.inv(icov_q + M.T@icov_C@M)
mean_2 = lambda D: np.asarray(cov_2@np.asarray((icov_q@mean_q + M.T@icov_C@(D-F_0)))[0])[0]

'''posterior_2 = multivariate_normal(mean_2(D), cov_2)

total = []
avrg = 15
for i in range(avrg):
    #Sample theta_i from posterior_1 and D_i from L_2
    steps = 5000
    theta_i = []
    D_i = []
    for i in range(steps):
        tmp_theta = posterior_1.rvs(size=1)
        tmp_D = L_2(tmp_theta).rvs(size=1)
        
        theta_i.append(tmp_theta)
        D_i.append(tmp_D)
    #Create likelihood matrix
    mat_L = []
    for i in range(steps):
        mat_L.append(L_2(theta_i[i]).pdf(D_i))
    mat_L = np.asarray(mat_L)
    sum1 = 0
    for i in range(steps):
        sum1 -= np.log(sum(mat_L[:,i])/steps)/(steps*np.log(2))
    sum2 = 0
    sum2 = np.trace(np.log(mat_L))/(steps*np.log(2))
    total.append(sum1 + sum2)
print('<D> after average of %s calculations' %avrg)
print(sum(total)/avrg)
print('Error: ')
print(np.sqrt(np.var(total)))'''

#------------------------------------------------------------------------------
def DimensionTest(start, end, avrg, MCsteps=10000):
    """
    Will compare the entropy results from MCentropy and Surprise() for the same 
    normal distributed priors and posteriors when climing in diemsionality of
    the distributions.
    
    Args:
        (start, end) (int): The start dimension and end dimension of the test
        avrg (int): The number of computations made inorder to take the average from
        MCsteps (int): How many randome samples are used in computing the MCentropy
    """
    sc = Surprise()
    
    Dimensions = []
    MCentropy = []
    MCerror = []
    GaussEntropy = []
    Gausserror = []
    GaussSTD = []
    RelativeDiff = []
    
    EEE = []
    EEEerror = []
    GaussEE = []
    GaussEEerror = []
    RelativeDiffE = []
    
    for dim in range(start, end+1):
        posterior_1 = multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim))
        
        #Build the model, choose F_0 = 0
        F_0 = np.ones(9)
        M = []
        for i in range(9):
            tmp = np.ones(dim)
            if i < dim:
                tmp[i] = 4
            M.append(tmp)
        M = np.matrix(M)
        #Define the prior for posterior_2
        mean_q = np.zeros(dim)
        cov_q = np.identity(dim)*5
        icov_q = np.linalg.inv(cov_q)
        
        cov_C = np.identity(9)*2
        icov_C = np.linalg.inv(cov_C)
        
        Model = lambda theta: np.asarray(F_0 + M@theta)[0]
        
        L_2 =  lambda theta: multivariate_normal(Model(theta), cov_C)
        
        #Calculate posterior 2 : p_2
        D = L_2(np.zeros(dim)).rvs(size=1)
        cov_2 = np.linalg.inv(icov_q + M.T@icov_C@M)
        mean_2 = lambda D: np.asarray(cov_2@np.asarray((icov_q@mean_q + M.T@icov_C@(D-F_0)))[0])[0]
        
        posterior_2 = multivariate_normal(mean_2(D), cov_2)
        
        posterior_1_sample = posterior_1.rvs(size=400000)
        posterior_2_sample = posterior_2.rvs(size=400000)
        
        
        
        def EstimatedExpectedEntropy(likelihood, posterior_1, steps):
            """
            """
            warnings.simplefilter("error", RuntimeWarning)
            #Sample theta_i from posterior_1 and D_i from L_2
            theta_i = posterior_1.rvs(size=steps)
            D_i = []
            for i in range(steps):
                tmp_D = likelihood(theta_i[i]).rvs(size=1)
                D_i.append(tmp_D)
            
            #Create likelihood matrix
            mat_L = []
            for i in range(steps):
                mat_L.append(likelihood(theta_i[i]).pdf(D_i))
            mat_L = np.asarray(mat_L)
            sum1 = 0
            for i in range(steps):
                sum1 -= np.log(sum(mat_L[:,i])/steps)/(steps*np.log(2))
            sum2 = 0
            try: 
                sum2 = np.trace(np.log(mat_L))/(steps*np.log(2))
            except RuntimeWarning:
                count = 0
                for i in range(steps):
                    if abs(mat_L[i,i]) < 1e-100:
                        count += 1
                        continue
                    else:
                        sum2 += np.log(mat_L[i,i])
                steps = steps - count
                sum2 /= (steps*np.log(2))
            entropy = sum1 + sum2
            
            return entropy
    
        MCsum = []
        GaussSum = []
        tmpGaussSTD = []
        EEEsum = []
        GaussEEsum = []
        
        for i in range(avrg):
            MCsum.append(MonteCarloENTROPY(posterior_1_sample,posterior_2_sample, MCsteps))
            EEEsum.append(EstimatedExpectedEntropy(L_2, posterior_1, 5000))
            surprise = sc(posterior_1_sample, posterior_2_sample, mode='add')
            GaussSum.append(surprise[0])
            GaussEEsum.append(surprise[1])
            tmpGaussSTD.append(surprise[-2])
        
        Msum = sum(MCsum)
        EEsum = sum(EEEsum)
        Gsum = sum(GaussSum)
        GEsum = sum(GaussEEsum)
        
        Dimensions.append(dim)
        MCentropy.append(Msum/avrg)
        MCerror.append(np.sqrt(np.var(MCsum)))
        EEE.append(EEsum/avrg)
        EEEerror.append(np.sqrt(np.var(EEEsum)))
        GaussEntropy.append(Gsum/avrg)
        GaussEE.append(GEsum/avrg)
        Gausserror.append(np.sqrt(np.var(GaussSum)))
        GaussEEerror.append(np.sqrt(np.var(GaussEEsum)))
        GaussSTD.append(sum(tmpGaussSTD)/avrg)
        if Msum > Gsum:
            tmp = (1 - (Gsum/Msum))*100
            RelativeDiff.append(tmp)
        else:
            tmp = (1 - (Msum/Gsum))*100
            RelativeDiff.append(tmp)
        if EEsum > GEsum:
            tmp = (1 - (GEsum/EEsum))*100
            RelativeDiffE.append(tmp)
        else:
            tmp = (1 - (EEsum/GEsum))*100
            RelativeDiffE.append(tmp)
        
    fig = plt.figure() 
    ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.6],
                       ylabel='Relative Entropy [Bits]',
                       title='Dimension Test')
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.2],
                       sharex=ax1,
                       xlabel='Number of Dimensions',
                       ylabel='Diff. in %')
    
    ax1.errorbar(Dimensions, GaussEntropy, yerr=Gausserror, marker='o', ls='none',
                 color='orange', label='Gauss Entropy')
    ax1.errorbar(Dimensions, MCentropy, yerr=MCerror, marker='*', ls='none',
                 label='Monte Carlo Entropy')
    ax2.bar(Dimensions, RelativeDiff, width=0.2, label='Rel. Diff')
    ax1.legend()
    plt.savefig('./Plots/DimensionTestAnalyticalNew.pdf', format='pdf')
    
    
    fig = plt.figure() 
    ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.6],
                       ylabel='Expected Relative Entropy [Bits]',
                       title='Dimension Test')
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.2],
                       sharex=ax1,
                       xlabel='Number of Dimensions',
                       ylabel='Diff. in %')
    
    ax1.errorbar(Dimensions, GaussEE, yerr=GaussEEerror, marker='o', ls='none',
                 color='orange', label='Gauss Expected Entropy')
    ax1.errorbar(Dimensions, EEE, yerr=EEEerror, marker='*', ls='none',
                 label='MC Expected Entropy')
    ax2.bar(Dimensions, RelativeDiffE, width=0.2, label='Rel. Diff')
    ax1.legend()
    plt.savefig('./Plots/DimensionTestExpectedEntNew.pdf', format='pdf')



DimensionTest(2, 8, 10, MCsteps=25000)
    