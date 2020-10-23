#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  20 14:00:14 2020

@author: BenjaminSuter
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.stats import boxcox, boxcox_normmax
from scipy.stats import gaussian_kde, multivariate_normal
from p_tqdm import p_map
from surprise import Surprise
from datetime import datetime



def IterativeEntropy(pri_data, post_data, iterations, mode='add'):
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
    
    axis = 1
    
    try:
        dim = len(prior[:,0])
    except IndexError:
        dim = 1
        axis = 0
    
    cach_rel_ent = []
    cach_exp_ent = []
    cach_S = []
    cach_sD = []
    
    BoxCoxError = False
    for steps in range(iterations):
        if steps == 0:
            
            prior_mu = np.mean(prior, axis=axis)
            prior_std = np.std(prior, axis=axis)
            
            #Standardize data
            prior = ((prior.T-prior_mu)/prior_std).T
            posterior = ((posterior.T-prior_mu)/prior_std).T
            
            if dim > 1:
                #Rotate data into eigenbasis of the covar matrix
                pri_cov = np.cov(prior)
                
                #Compute eigenvalues 
                eVa, eVe = np.linalg.eig(pri_cov)
                
                #Compute transformation matrix from eigen decomposition
                R, S = eVe, np.diag(np.sqrt(eVa))
                T = np.matmul(R,S).T
                
                #Transform data with inverse transformation matrix T^-1
                inv_T = np.linalg.inv(T)
                
                prior = np.matmul(prior.T, inv_T).T
                posterior = np.matmul(posterior.T, inv_T).T
        
        #Transform to positive parameter values
        for k in range(dim):
            prior_a = np.amin(prior[k,:])
            post_a = np.amin(posterior[k,:])
            if prior_a < post_a and prior_a < 0:
                prior[k,:] -= prior_a
                posterior[k,:] -= prior_a
            elif post_a < 0:
                prior[k,:] -= post_a
                posterior[k,:] -= post_a
            prior_a = np.amin(prior[k,:])
            post_a = np.amin(posterior[k,:])
            while prior_a <= 0 or post_a <= 0:
                prior[k,:] += 5.0E-6
                posterior[k,:] += 5.0E-6
                prior_a = np.amin(prior[k,:])
                post_a = np.amin(posterior[k,:])
                
            #Find optimal one-parameter Box_Cox transformation
            try:
                lambd = boxcox_normmax(prior[k,:], brack=(-1.9, 2.0), method='mle')    
                box_cox_prior = boxcox(prior[k,:], lmbda=lambd)
                #print(box_cox_prior)
                #print(lambd)
                prior[k,:] = box_cox_prior
                
                box_cox_post = boxcox(posterior[k,:], lmbda=lambd)
                posterior[k,:] = box_cox_post
            except RuntimeWarning:
                print('Something went wrong with BoxCox')
                BoxCoxError = True
                break
            
        if BoxCoxError:
            break
        prior_mu = np.mean(prior, axis=1)
        prior_std = np.std(prior, axis=1)
            
        #Standardize data
        prior = ((prior.T-prior_mu)/prior_std).T
        posterior = ((posterior.T-prior_mu)/prior_std).T
        
        if dim > 1:
            #Rotate data into eigenbasis of the covar matrix
            pri_cov = np.cov(prior)
            
            #Compute eigenvalues 
            eVa, eVe = np.linalg.eig(pri_cov)
            
            #Compute transformation matrix from eigen decomposition
            R, S = eVe, np.diag(np.sqrt(eVa))
            T = np.matmul(R,S).T
            
            #Transform data with inverse transformation matrix T^-1
            
            try:
                inv_T = np.linalg.inv(T)
                prior = np.matmul(prior.T, inv_T).T
                posterior = np.matmul(posterior.T, inv_T).T
                
            except:
                print('Singular Matrix in inversion')
                print('Stopping BoxCox')
                cach_rel_ent.append(None)
                cach_exp_ent.append(None)
                cach_S.append(None)
                cach_sD.append(None)
                break
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
            if abs(cach_rel_ent[-1] - cach_rel_ent[-2])/cach_rel_ent[-2] < 0.001:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] - cach_rel_ent[-3])/cach_rel_ent[-3] < 0.005:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] - cach_rel_ent[-4])/cach_rel_ent[-4] < 0.005:
                convergence_flag += 1
        if convergence_flag == 3:
            print('Convergence reached')
            break
    
    return prior, posterior, cach_rel_ent, cach_exp_ent, cach_S, cach_sD, BoxCoxError



def LoadAndComputeEntropy(prior, post, steps=300,
                          pri_burn=None, post_burn=None,
                          params=[0,1,2,3,4], mode='add'):
    """
    Load prior and posterior data, then compute the relative entropy using 
    IterativeEntropy()
    
    Args:
        prior (str): String containing the path to the data samples
                     Or pre-loaded data as (ndarray)
        post (str): String containing the path to the data samples
                    Or pre-loaded data as (ndarray)
        steps (int): Number of iteration steps used in IterativeEntropy()
                     default is 300
        pri_burn (int): Number of data points cut off at begining of prior
                        sample data; default is 50%
        post_burn (int): Number of data points cut off at begining of post
                         sample data; default is 50%
        params (list): List which indicates what varied parameters should be
                       used when computing the relative entropy
                       h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                       m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
                       Default is [0,1,2,3,4]
    Returns:
        results (list): A list containing following results in order
                        - Gaussianised prior (narray)
                        - Gaussianised posterior (narray)
                        - Relative entropy at each iteration step (list)
                        - Expected Entropy at each iteration step (list)
                        - Surprise at each iteration step (list)
                        - Standard deviation of expected entropy at each
                          iteration step (list)
    """
    if isinstance(prior, str) and not pri_burn:
        tmp = np.load(prior)
        pri_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        prior_data = np.load(prior)[pri_burn:,:]
    elif isinstance(prior, str):
        prior_data = np.load(prior)[pri_burn:,:]
    elif isinstance(prior, np.ndarray):
        prior_data = prior
        pri_burn = 0
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    
    if isinstance(post, str) and not post_burn:
        tmp = np.load(post)
        post_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        post_data = np.load(post)[post_burn:,:]
    elif isinstance(post, str):
        post_data = np.load(post)[post_burn:,:]
    elif isinstance(post, np.ndarray):
        post_data = post
        post_burn = 0
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    
    results = ['','','','','','','']
    
    pri, post, rel_ent, exp_ent, S, sD, err = IterativeEntropy(prior_data.T[params,:],
                                                               post_data.T[params,:],
                                                               steps, mode=mode)
    results[0] = pri
    results[1] = post
    results[2] = rel_ent
    results[3] = exp_ent
    results[4] = S
    results[5] = sD
    results[6] = err
    
    del(prior_data)
    del(post_data)
    
    return results



def RemoveScatter(data_path, parameters, cut_off, burning=200000):
    """
    Will remove unwanted scatter from the data
    
    Args:
        data_path (str): String containing the path to the data samples
        parameters (list): List containing the parameter index where the
                           scatter should be removed
                           h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                           m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
        cut_off (list): List of the cut off for each parameter where the 
                        scatter should be returned. 
                        If cut off is an int it will be the upper bouned and if
                        cut off is a touple of ints it will be the upper and
                        lower bound (upper bound, lower bound)
        burning (int): Number of data points cut off at begining of
                       sample data; default is 200000
    Returns:
        data (ndarray): New data set where scatter has been removed
    """
    
    data = np.load(data_path)[burning:,:]
    cut_off_index = 0
    
    for param in parameters:
        if isinstance(cut_off[cut_off_index], float):
            scatter_index = np.argwhere(data[:,param] > cut_off[cut_off_index])
            replace_val = np.mean(np.delete(data[:,param], scatter_index))
            
            #Replace scatter values with mean value computed without scatter
            data[:,param] = np.where(data[:,param] < cut_off[cut_off_index],
                                     data[:,param], replace_val)
            cut_off_index += 1
        elif isinstance(cut_off[cut_off_index], tuple):
            scatter_index = np.argwhere(data[:,param] > cut_off[cut_off_index][0])
            temp = np.delete(data[:,param], scatter_index)
            scatter_index = np.argwhere(temp < cut_off[cut_off_index][1])
            replace_val = np.mean(np.delete(temp, scatter_index))
            
            #Replace scatter values with mean value computed without scatter
            data[:,param] = np.where(data[:,param] < cut_off[cut_off_index][0],
                                     data[:,param], replace_val)
            data[:,param] = np.where(data[:,param] > cut_off[cut_off_index][1],
                                     data[:,param], replace_val)
            cut_off_index += 1
        
    return data
    
   
def MonteCarloENTROPY(prior, post, steps, error=False,
                      pri_burn=None, post_burn=None,
                      params=[0,1,2,3,4]):
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
        params (list): List which indicates what varied parameters should be
                       used when computing the relative entropy
                       h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                       m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
                       Default is [0,1,2,3,4]
        
    Returns:
        entropy (float): Approximation of the relative entropy
    """
    cpu_count = mp.cpu_count()
    
    #Assert that steps > cpu_count to prevent empty arrays in parallel compute
    if steps < cpu_count:
        cpu_count = steps

    warnings.simplefilter("error", RuntimeWarning)
    
    
    if isinstance(prior, str) and not pri_burn:
        tmp = np.load(prior)
        pri_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        prior_data = np.load(prior)[pri_burn:,params]
    elif isinstance(prior, str):
        prior_data = np.load(prior)[pri_burn:,params]
    elif isinstance(prior, np.ndarray):
        prior_data = prior
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    if isinstance(post, str) and not post_burn:
        tmp = np.load(post)
        post_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        post_data = np.load(post)[post_burn:,params]
    elif isinstance(post, str):
        post_data = np.load(post)[post_burn:,params]
    elif isinstance(post, np.ndarray):
        post_data = post
    else:
        raise Exception(
                'Invalide type for prior! Use str or np.ndarray')
    
    '''#Compute mean and standard deviation of the prior
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
    post_data = np.matmul(posterior.T, inv_T)'''
    
    
    
    prior_kernel = gaussian_kde(prior_data.T)
    post_kernel = gaussian_kde(post_data.T)
    
    #Generate new sample set for MC evaluation
    sample_points = post_kernel.resample(size=steps).T
    
    #Parallel compute g_i and f_i
    prior_prob = p_map(prior_kernel.evaluate,
                       [sample_points[i::cpu_count].T for i in range(cpu_count)],
                       desc="Evaluating prior probability")
    post_prob = p_map(post_kernel.evaluate,
                      [sample_points[i::cpu_count].T for i in range(cpu_count)],
                      desc="Evaluating posterior probability")
    
    
    #Compute log(f_i/g_i), using log_2 for bit interpretation
    try:
        quotient = np.divide(post_prob,prior_prob)
    #Catch 'divide by zero' and adjust steps for invalide probes
    except RuntimeWarning:
        print('In exception...')
        quotient = []
        count = 0
        index_i = 0
        for prior_vec in prior_prob:
            index_j = 0
            try:
                for prior_val in prior_vec:
                    if abs(prior_val) < 1e-30:
                        count += 1
                        index_j += 1
                        continue
                    else:
                        post_val = post_prob[index_i][index_j]
                        quotient.append(post_val/prior_val)
                        index_j += 1
                index_i += 1
            except TypeError:
                print('Iteration failed, in second exception...')
                prior_val = prior_vec
                if abs(prior_val) < 1e-30:
                    count += 1
                    continue
                else:
                    post_val = post_prob[index_i][index_j]
                    quotient.append(post_val/prior_val)
                index_i += 1
        steps = steps - count
        if steps == 0:
            print('All f_i/g_i failed!! Can not estimate relative entropy.')
            quotient = [np.nan]
            steps = 1
       
    temp_res = []
    Type_Error = False
    for value in quotient:
        temp_res.append(np.log2(value))
    tot_sum = 0
    for value in temp_res:
        try:
            tot_sum += sum(value)
        except TypeError:
            Type_Error = True
            tot_sum += value
    if Type_Error and not np.isnan(tot_sum):
        print('Something was wrong with iterating log2 values.\n', 
              'MC Entropy was probably estimated with a low count number...')
    entropy = tot_sum/steps
    
    if error:
        error_estimate = np.var(temp_res)/steps
        return entropy, error_estimate
    else:
        return entropy


def LogarithmicScore(data_sample, steps=200000, burn=200000):
    """
    Will compute the logarithmic score given a gaussian distribution
    
    Args:
        data_sample (str): String containing the path to the data samples
                           Or pre-loaded data as (ndarray) with shape (N_samples, n_dim)
        steps (int): Number of steps performed to compute the logarithmic score
                     default is 100000
        burn (int): Number of data points cut off at begining of data_sample
                    default is 200000
    """
    if isinstance(data_sample, str):
        data_sample = np.load(data_sample)[burn:,:5]
    
    data_sample_trans = data_sample.T
    mu = np.mean(data_sample_trans, axis=1)
    std = np.std(data_sample_trans, axis=1)
    
    data = ((data_sample-mu)/std).T
    cov = np.cov(data)

    eVa, eVe = np.linalg.eig(cov)
    R, S = eVe, np.diag(np.sqrt(eVa))
    T = np.matmul(R,S).T

    inv_T = np.linalg.inv(T)
    data = np.matmul(data.T, inv_T).T
    
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    gauss_pdf = multivariate_normal(mu, cov)
    
    #Generate i.i.d sampling set
    max_val = len(data[:,0])-1
    float_samples = np.random.random(steps)*max_val
    #converte to int
    int_samples = np.around(float_samples).astype(int)
    
    sample_points = data[:,int_samples].T
    
    result = sum(gauss_pdf.logpdf(sample_points))/steps
    
    return result



def EntropyConsistencyTest(prior_mu, prior_cov, post_mu, post_cov,
                           size=250000):
    """
    Will generate a gaussian prior and posterior distribution and compare the 
    entropy result from the Monte Carlo simulation to the analythical result
    for gaussian distributions.
    
    Args:
        prior_mu (ndarray): The mean of the prior distribution
        prior_cov (ndarray): The covariance of the prior distribution
        post_mu (ndarray): The mean of the posterior distribution
        post_cov (ndarray): The covariance of the posterior distribution
        size (int): Size of test data
    Returns:
        diff (float): Absolute difference between Monte Carlo Entropy and 
                      analythical entropy |E_MC - E_gauss|
    """
    start = datetime.now()
    sc = Surprise()
    fig = plt.figure()
    
    prior_sample = multivariate_normal.rvs(mean=prior_mu, cov=prior_cov,
                                           size=size)
    post_sample = multivariate_normal.rvs(mean=post_mu, cov=post_cov,
                                          size=size)
    mc_entropy = []
    for i in range(2000):
        mc_entropy.append(MonteCarloENTROPY(prior_sample, post_sample, 300))
    gauss_entropy = sc(prior_sample, post_sample)[0]
    
    entropy_hist = np.histogram(mc_entropy,bins=20, density=True)
    mean = np.around(sum(mc_entropy)/len(mc_entropy), decimals=3)
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                       xlabel='$D_{MC}(f||g)$',
                       ylabel='$Probability$',
                       title='EntropyConsistencyTest')
    ax1.step(entropy_hist[1][:-1],entropy_hist[0],
             label='%d its. of $D_{MC}(%d)$' %(2000, 300))
    ax1.axvline(x=mean, color='r', ls='--', label='Mean value of %s [Bits]' % mean)
    ax1.axvline(x=gauss_entropy, color='g',
                label='Gauss Entropy of %s [Bits]' % round(gauss_entropy,3))
    ax1.legend()
    plt.savefig('./Plots/EntropyConsistencyTest2.pdf', format='pdf')
    
    diff = abs(sum(mc_entropy)/len(mc_entropy) - gauss_entropy)
    print('Time: ', datetime.now()-start)
    
    return diff
    