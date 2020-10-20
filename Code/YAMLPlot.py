#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:50:17 2020

@author: BenjaminSuter
"""
import yaml
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal, skewnorm, norm, boxcox_normmax, boxcox
from surprise import Surprise
from IterCompRelEnt import MonteCarloENTROPY, LoadAndComputeEntropy



def OneDimIterativeEntropy(pri_data, post_data, iterations, mode='add'):
    """
    Algorithm used to iterativly compute the relative entropy based on
    gaussianisation of both the prior and the posterior distribution.
    
    Args:
        pri_data (ndarray): Data array of shape (n, m) where n is #variats and 
                           m is #observations
        post_data (ndarray): Data array of shape (n, m) where n is #variats
                            and m is #observations
        iterations (int): Number of iterations
        mode (str): relation between dist1 and dist2; either
                    * 'add' if dist1 is the prior of dist2
                    * 'replace' if dist1 and dist2 are independently derived 
                       posteriors and the prior is much wider than the constraints
                    * 'partial' if dist1 and dist2 are independently derived 
                       using the same prior dist3
                    Default: 'add'
        ....
    
    Returns:
        ....
    """
    if not isinstance(mode, str):
        raise Exception(
                'Invalid kind value %s. Allowed format:'
                'str', mode)
    
    sc = Surprise()
    
    prior = pri_data
    posterior = post_data
    
    cach_rel_ent = []
    cach_exp_ent = []
    cach_S = []
    cach_sD = []
    
    BoxCoxError = False
    for steps in range(iterations):
        if steps == 0:
            
            prior_mu = np.mean(prior, axis=0)
            prior_std = np.std(prior, axis=0)
            
            #Standardize data
            prior = ((prior.T-prior_mu)/prior_std).T
            posterior = ((posterior.T-prior_mu)/prior_std).T
        
        #Transform to positive parameter values
        prior_a = np.amin(prior)
        post_a = np.amin(posterior)
        if prior_a < post_a and prior_a < 0:
            prior -= prior_a
            posterior -= prior_a
        elif post_a < 0:
            prior -= post_a
            posterior -= post_a
        prior_a = np.amin(prior)
        post_a = np.amin(posterior)
        while prior_a <= 0 or post_a <= 0:
            prior += 5.0E-6
            posterior += 5.0E-6
            prior_a = np.amin(prior)
            post_a = np.amin(posterior)
                
        #Find optimal one-parameter Box_Cox transformation
        try:
            lambd = boxcox_normmax(prior, brack=(-1.9, 2.0), method='mle')    
            box_cox_prior = boxcox(prior, lmbda=lambd)
            prior = box_cox_prior
                
            box_cox_post = boxcox(posterior, lmbda=lambd)
            posterior = box_cox_post
        except RuntimeWarning:
            print('Something went wrong with BoxCox')
            BoxCoxError = True
            break
            
        if BoxCoxError:
            break
        prior_mu = np.mean(prior, axis=0)
        prior_std = np.std(prior, axis=0)
            
        #Standardize data
        prior = ((prior.T-prior_mu)/prior_std).T
        posterior = ((posterior.T-prior_mu)/prior_std).T
        
        #Compute D, <D>, S and sigma(D)
        try:
            rel_ent, exp_rel_ent, S, sD, p = sc(prior.T, posterior.T, mode=mode)
        except:
            print('Error with Suprise()')
            rel_ent = None
            exp_rel_ent = None
            S = None
            sD = None
        
        cach_rel_ent.append(rel_ent)
        cach_exp_ent.append(exp_rel_ent)
        cach_S.append(S)
        cach_sD.append(sD)
        convergence_flag = 0
        if steps > 3 and not None in cach_rel_ent[-4:]:
            if abs(cach_rel_ent[-1] - cach_rel_ent[-2]) < 0.005:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] - cach_rel_ent[-3]) < 0.0075:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] - cach_rel_ent[-4]) < 0.01:
                convergence_flag += 1
        if convergence_flag == 3:
            print('Convergence reached')
            break
    
    return prior, posterior, cach_rel_ent, cach_exp_ent, cach_S, cach_sD, BoxCoxError



def RelativeDiff(a,b):
    if np.isnan(a) or np.isnan(b):
        res = np.nan
    elif a < b:
        res = (1-a/b)*100
    else:
        res = (1-b/a)*100
    return res

def PrettyString(x):
    tmp = x.replace('samples', '')
    tmp = tmp.replace('nonuisance', 'N')
    tmp = tmp.replace('Gaussian', 'g')
    tmp = tmp.replace('sim', 's')
    tmp = tmp.replace('tt', 'T')
    tmp = tmp.replace('dd', 'D')
    tmp = tmp.replace('kk', 'K')
    tmp = tmp.replace('gg', 'G')
    tmp = tmp.replace('_', '')
    return tmp

'''target = 'ThreeProb_avrg10_MCsteps50000.yml'
with open(target, 'r') as yaml_file:
    data = yaml.load(yaml_file)
data.pop('Names')

keys = []
MCresults = []
MCerr = []
Gresults = []
Gconv = []
noBCent = []
for key in data:
    keys.append(PrettyString(key))
    res = [np.nan if val=='None' else val for val in data[key]]
    MCresults.append(float(res[0]))
    MCerr.append(float(res[1]))
    Gresults.append(float(res[2]))
    Gconv.append(float(res[7]))
    noBCent.append(float(res[-2]))
#Compute the relative difference between results
MC_G_diff = []
MC_C_diff = []
MC_B_diff = []
for i in range(len(keys)):
    MC_G_diff.append(RelativeDiff(MCresults[i],Gresults[i]))
    MC_C_diff.append(RelativeDiff(MCresults[i],Gconv[i]))
    MC_B_diff.append(RelativeDiff(MCresults[i],noBCent[i]))
    
fig = plt.figure(figsize=(16,6)) 
ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.6],
                   ylabel='Relative Entropy [Bits]',
                   xticklabels=[],
                   title='Entropy Comparison')
ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.2],
                   ylabel='Diff. in %')
    
ax1.errorbar(keys, Gresults, marker='o', color=(1,0,0,0.6), ls='none',
             label='Gauss Entropy')
ax1.errorbar(keys, Gconv, marker='o', color=(0,1,0,0.6), ls='none',
             label='Gauss Entropy Converged')
ax1.errorbar(keys, noBCent, marker='o', color=(0,0,1,0.6), ls='none',
             label='no BoxCox Entropy')
ax1.errorbar(keys, MCresults, yerr=MCerr, marker='*', color='black', ls='none',
             label='Monte Carlo Entropy')
ax2.bar(keys, MC_G_diff, width=0.2, color=(1,0,0,0.6),label='MC to Gauss')
ax2.bar(keys, MC_C_diff, width=0.2, color=(0,1,0,0.6), align='edge',label='MC to Conv.')
ax2.bar(keys, MC_B_diff, width=-0.2, color=(0,0,1,0.6), align='edge',label='MC to no BC')
ax1.legend()
ax2.legend()
ax1.grid(True, axis='y')
ax2.grid(True, axis='y')
#plt.show()
plt.savefig('./Plots/ThreeProbs.pdf', format='pdf')'''


    

sc = Surprise()
'''cov = np.matrix([[2,0.1,0.1,0.1,0.1],
                 [0.1,2,0.1,0.1,0.1],
                 [0.1,0.1,2,0.1,0.1],
                 [0.1,0.1,0.1,2,0.1],
                 [0.1,0.1,0.1,0.1,2]])'''
'''#cov = cov*0.32
cov = np.identity(5)*(0.32)

prior_sample1 = multivariate_normal.rvs(mean=np.ones(5)*16,
                                       cov=np.identity(5)*5, size=400000)
post_sample1 = multivariate_normal.rvs(mean=np.ones(5)*20,
                                      cov=cov, size=400000)

prior_sample = np.square(np.log(multivariate_normal.rvs(mean=np.ones(5)*16,
                                       cov=np.identity(5)*5, size=900000)))
post_sample = np.square(np.log(multivariate_normal.rvs(mean=np.ones(5)*20,
                                      cov=cov, size=900000)))'''
'''skewness = range(0,12,2)
MC_res = []
Gauss_res = []
OneBC_res = []
Conv_res = []
priors = []
for skew in skewness:
    #skew_sample = norm.rvs(scale=0.8,size=250000)
    norm_sample = skewnorm.rvs(2,loc=0.8,scale=1,size=250000)
    skew_sample = skewnorm.rvs(skew, scale=3,size=250000)
    x = np.linspace(-10,10,1000)
    priors.append(skewnorm.pdf(x,skew,scale=3))
    avrg = 5
    MCent = []
    Gent = []
    OneBC = []
    Conv = []
    
    count1 = avrg
    count2 = avrg
    for step in range(avrg):
        MCent.append(MonteCarloENTROPY(norm_sample, skew_sample, 8000))
        Gent.append(sc(norm_sample, skew_sample, mode='add')[0])
        tmp = OneDimIterativeEntropy(norm_sample, skew_sample, 1)
        if not tmp[-1] and not tmp[2][-1]==None and tmp[2][-1] < 3.5:
            OneBC.append(tmp[2][-1])
        elif not tmp[-1] and not tmp[2][-1]==None and MCent[-1]/tmp[2][-1] > 0.5:
            OneBC.append(tmp[2][-1])
        else:
            count1 -= 1
        tmp = OneDimIterativeEntropy(norm_sample, skew_sample, 300)
        if not tmp[-1] and not tmp[2][-1]==None and tmp[2][-1] < 0.5:
            Conv.append(tmp[2][-1])
        elif not tmp[-1] and not tmp[2][-1]==None and MCent[-1]/tmp[2][-1] > 0.5:
            Conv.append(tmp[2][-1])
        else:
            count2 -= 1
    
    MC_res.append(sum(MCent)/avrg)
    Gauss_res.append(sum(Gent)/avrg)
    if count1 > 0:
        OneBC_res.append(sum(OneBC)/avrg)
    else:
        OneBC_res.append(np.nan)
    if count2 > 0:
        Conv_res.append(sum(Conv)/avrg)
    else:
        Conv_res.append(np.nan)
    

plt.figure()
plt.plot(skewness, MC_res, '*', color='black', label='MC Entropy')
plt.plot(skewness, Gauss_res, 'x', color=(0,0,1,0.6), label='no BC')
plt.plot(skewness, OneBC_res, 'v', color=(1,0,0,0.6), label='Gauss Entropy')
plt.plot(skewness, Conv_res, 'o', color=(0,1,0,0.6), label='Conv')
plt.xlabel('Skew value')
plt.ylabel('Entropy')
plt.legend()
plt.show()

plt.figure()
x = np.linspace(-10,10,1000)
nn = skewnorm.pdf(x, 2, loc=0.8,scale=1)
plt.plot(x,nn, label='prior')
for skew in priors:
    plt.plot(x,skew)
plt.legend()
plt.show()'''




'''print('Skew is: ', skew)
print('MCentropy: ', sum(MCent)/avrg)
print('Gent: ', sum(Gent)/avrg)'''
'''print('BCent: ', sum(BCent)/avrg)'''
'''print('MCentropySQr: ', sum(MCentSQr)/avrg)
print('GentSQr: ', sum(GentSQr)/avrg)'''
'''print('BCentSQr: ', sum(BCentSQr)/avrg)'''


'''h1 = np.histogram(post_sample1[:,3], bins=100, density=True)
h2 = np.histogram(post_sample[:,3], bins=100, density=True)
p1 = np.histogram(prior_sample1[:,3], bins=100, density=True)
p2 = np.histogram(prior_sample[:,3], bins=100, density=True)
plt.figure()
plt.step(h1[1][:-1], h1[0], label='post')
plt.step(h2[1][:-1], h2[0], label='post squared')
plt.step(p1[1][:-1], p1[0], label='prior')
plt.step(p2[1][:-1], p2[0], label='prior squared')
plt.ylim(0,1.5)
#plt.xlim(10,60)
plt.legend()
plt.show()'''

    
    


'''prior_sample = np.random.normal(0,1,250000)
post_sample = np.random.normal(0,0.32,250000)'''

sum1 = 0
sum2 = 0
'''for i in range(5):
    sum1 += sc(prior_sample, post_sample, mode='replace')[0]'''

'''print(sum1/5)'''

'''print('MC Entropy 100000 samples')

for i in range(5):
    sum2 += MonteCarloENTROPY(prior_sample,post_sample, 100000)
print(sum2/5)'''


'''fig = plt.figure(figsize=(15,6))


Names = ['DGgauss', 'DGsim', 'DKgauss', 'DKgaussNoN', 'DKsim',
         'DKsimNon', 'TDgauss', 'TDgaussNon', 'TDsimNon', 'TGgaussNon',
         'TGsimNon', 'TKgaussNon', 'TKsimNon', 'TestGauss']
MCresults.append(5.11)
Eresults.append(gauss_entropy[0])
sDresults.append(gauss_entropy[-2])

ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                       ylabel='$Relative Entropy [Bits]$',
                       title='Comparison Plot',
                       ylim=[0,8])

ax1.plot(Names, MCresults, 'o', label='MC Entropy')
ax1.errorbar(Names, Eresults, yerr=sDresults, marker='*', ls='none', label='Gauss Entropy')
plt.legend()
plt.savefig('./Plots/ConsistencyPlot2.pdf', format='pdf')'''



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
    Dimensions = []
    MCentropy = []
    MCerror = []
    GaussEntropy = []
    Gausserror = []
    GaussSTD = []
    RelativeDiff = []
    
    for dim in range(start, end+1):
        if dim == 1:
            prior_sample = np.random.normal(0,1,400000)
            post_sample = np.random.normal(0,0.32,400000)
        else:
            prior_sample = multivariate_normal.rvs(mean=np.zeros(dim),
                                                   cov=np.identity(dim),
                                                   size=400000)
            post_sample = multivariate_normal.rvs(mean=np.zeros(dim),
                                                  cov=np.identity(dim)*(0.32),
                                                  size=400000)
        MCsum = []
        GaussSum = []
        tmpGaussSTD = []
        for i in range(avrg):
            MCsum.append(MonteCarloENTROPY(prior_sample,post_sample, MCsteps))
            surprise = sc(prior_sample, post_sample, mode='add')
            GaussSum.append(surprise[0])
            tmpGaussSTD.append(surprise[-2])
        
        Msum = sum(MCsum)
        Gsum = sum(GaussSum)
        
        Dimensions.append(dim)
        MCentropy.append(Msum/avrg)
        MCerror.append(np.sqrt(np.var(MCsum)))
        GaussEntropy.append(Gsum/avrg)
        Gausserror.append(np.sqrt(np.var(GaussSum)))
        GaussSTD.append(sum(tmpGaussSTD)/avrg)
        if Msum > Gsum:
            tmp = (1 - (Gsum/Msum))*100
            RelativeDiff.append(tmp)
        else:
            tmp = (1 - (Msum/Gsum))*100
            RelativeDiff.append(tmp)
    
    fun = lambda x: (x/2)*(0.32-1-np.log(0.32))/np.log(2)
    
    theory = []
    steps = []
    for i in range(2,end+1):
        theory.append(fun(i))
        steps.append(i)
        
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
    ax1.plot(steps,theory, label='Analytical values')
    ax2.bar(Dimensions, RelativeDiff, width=0.2, label='Rel. Diff')
    ax1.legend()
    plt.savefig('./Plots/DimensionTestAnalytical.pdf', format='pdf')

DimensionTest(1, 8, 20, MCsteps=25000)

def FiveDimDifferenceTest(steps, avrg, MCsteps=100000):
    MC = []
    Gauss = []
    RelDiff = []
    for i in range(steps):
        values = np.random.random(size=2)*100
        if values[0] > values[1]:
            prior_sample = multivariate_normal.rvs(mean=np.zeros(5),
                                                   cov=np.identity(5)*values[0],
                                                   size=250000)
            post_sample = multivariate_normal.rvs(mean=np.zeros(5),
                                                  cov=np.identity(5)*values[1],
                                                  size=250000)
        else:
            prior_sample = multivariate_normal.rvs(mean=np.zeros(5),
                                                   cov=np.identity(5)*values[1],
                                                   size=250000)
            post_sample = multivariate_normal.rvs(mean=np.zeros(5),
                                                  cov=np.identity(5)*values[0],
                                                  size=250000)
        MCtmp = []
        GAUSStmp = []
        for i in range(avrg):
            MCtmp.append(MonteCarloENTROPY(prior_sample,post_sample, MCsteps))
            GAUSStmp.append(sc(prior_sample, post_sample, mode='add')[0])
        MCtmp = sum(MCtmp)
        GAUSStmp = sum(GAUSStmp)
        MC.append(MCtmp/avrg)
        Gauss.append(GAUSStmp/avrg)
        
        if MCtmp > GAUSStmp:
            tmp = (1 - (GAUSStmp/MCtmp))*100
            RelDiff.append(tmp)
        else:
            tmp = (1 - (MCtmp/GAUSStmp))*100
            RelDiff.append(tmp)
    
    print(RelDiff)
    print('')
    print(Gauss)
    print('')
    print(MC)

'''FiveDimDifferenceTest(10, 10, MCsteps=10000)'''
    
    