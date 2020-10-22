#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:50:33 2020

@author: BenjaminSuter
"""
import warnings
import numpy as np
import multiprocessing as mp

from p_tqdm import p_map
#from tqdm import tqdm
from scipy.stats import gaussian_kde
from datetime import datetime


def MonteCarloENTROPY(prior, post, steps, error=False,
                      pri_burn=None, post_burn=None):
    """
    Will approximate the PDF of the prior and posterior distributions using
    the gaussian_kde tool
    Then D(f||g) ≈ (1/n)sum(log(f/g)) where n is the number of steps,
    f is the posterior distribution and g is the prior distribution. We sample
    n i.i.d samples from the f PDF
    
    Args:
        prior (str): String containing the path to the data samples
                     Or pre-loaded data as (ndarray) with shape (N_samples, n_dim)
        post (str): String containing the path to the data samples
                    Or pre-loaded data as (ndarray) with shape (N_samples, n_dim)
        steps (int): Number of iteration steps
        error (bool): If True will return an error estimate
        pri_burn (int): Number of data points cut off at begining of prior
                        sample data; default is 50%
        post_burn (int): Number of data points cut off at begining of post
                         sample data; default is 50%
        
    Returns:
        entropy (float): Approximation of the relative entropy
    """
    #tart = datetime.now()
    
    #pool = mp.Pool(mp.cpu_count())
    warnings.simplefilter("error", RuntimeWarning)
    
    #Load data
    if isinstance(prior, str) and not pri_burn:
        tmp = np.load(prior)
        pri_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        prior_data = np.load(prior)[pri_burn:,:5]
    elif isinstance(prior, str):
        prior_data = np.load(prior)[pri_burn:,:5]
    elif isinstance(prior, np.ndarray):
        prior_data = prior
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    if isinstance(post, str) and not post_burn:
        tmp = np.load(post)
        post_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        post_data = np.load(post)[post_burn:,:5]
    elif isinstance(post, str):
        post_data = np.load(post)[post_burn:,:5]
    elif isinstance(post, np.ndarray):
        post_data = post
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    
    #Compute mean and standard deviation of the prior
    prior_mu = np.mean(prior_data.T, axis=1)
    prior_std = np.std(prior_data.T, axis=1)
            
    #Standardize data
    prior = ((prior_data-prior_mu)/prior_std).T
    posterior = ((post_data-prior_mu)/prior_std).T
            
    #Rotate data into eigenbasis of the covar matrix
    pri_cov = np.cov(prior)
            
    #Compute eigenvalues 
    eVa, eVe = np.linalg.eig(pri_cov)
            
    #Compute transformation matrix from eigen decomposition
    R, S = eVe, np.diag(np.sqrt(eVa))
    T = np.matmul(R,S).T
            
    #Transform data with inverse transformation matrix T^-1
    inv_T = np.linalg.inv(T)
            
    prior_data = np.matmul(prior.T, inv_T)
    post_data = np.matmul(posterior.T, inv_T)
    
    
    #Generate a PDF for prior and post data
    prior_kernel = gaussian_kde(prior_data.T)
    post_kernel = gaussian_kde(post_data.T)
    
    #Generate i.i.d sampling set from posterior data set     --- not used
    #max_val = len(post_data[:,0])-1
    #float_samples = np.random.random(steps)*max_val
    #converte to int
    #int_samples = np.around(float_samples).astype(int)
    
    #sample_points = post_data[int_samples,:].T
    #sample_points = post_data[int_samples,:]
    
    #Generate new sample set for MC evaluation
    sample_points = post_kernel.resample(size=steps).T
    
    #Generate set of evaluated g_i and f_i
    #prior_prob = prior_kernel.evaluate(sample_points)
    #post_prob = post_kernel.evaluate(sample_points)'''
    
    #Parallel compute g_i and f_i
    #prior_prob = pool.map(prior_kernel.evaluate, [row for row in sample_points])
    #post_prob = pool.map(post_kernel.evaluate, [row for row in sample_points])
    print("Using p_map()")
    prior_prob = p_map(prior_kernel.evaluate, [row for row in sample_points])
    post_prob = p_map(post_kernel.evaluate, [row for row in sample_points])
    
    
    #Compute log(f_i/g_i), using log_2 for bit iterpretation
    try:
        quotient = np.divide(post_prob,prior_prob)
    #Catch 'divide by zero' and ajust steps for invalide probes
    except RuntimeWarning:
        quotient = []
        count = 0
        for i in range(steps):
            if prior_prob[i] == 0:
                count += 1
                continue
            else:
                quotient.append(post_prob[i]/prior_prob[i])
        steps = steps-count
    
    #temp_res = pool.map(np.log2, quotient)    
    temp_res = np.log2(quotient)
    
    entropy = sum(temp_res)/steps
    
    #time = datetime.now()-start
    #pool.close()
    
    if error:
        error_estimate = np.var(temp_res)/steps
        return entropy, error_estimate
    else:
        return entropy
    #return time