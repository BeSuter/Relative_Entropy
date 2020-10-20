#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:44:14 2020

@author: BenjaminSuter
"""
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
from scipy.stats import entropy, gaussian_kde, multivariate_normal, norm
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
from IterCompRelEnt import IterativeEntropy



#----------------------------------------------------------------------------

def GaussApprox(data):
    """
    Aproximate mean and cov from given data set
    
    Args:
        data (narray): array of form (n,m) where n is #variats and
                       m is #observations
    
    Returns:
        mean (array): mean value of variats
        covc(narray): Covariance matrix of variates
        
    """
    mu = np.mean(data, axis=1)
    cov = np.cov(data)
    
    return mu, cov



def GaussPDF(data):
    """
    Will compute the Probability Density Function of a given Data set
    
    Args:
        data (narray): array of (n,m) where n is #variats and
                       m is #observations
        
    Returns:
        PDF (scipy.stats._multivariate.multivariate_normal_gen object)
            evaluate PDF using the methode PDF.pdf(x), x is array_like
            
    """
    mu, cov = GaussApprox(data)
    
    return multivariate_normal(mu, cov)



def GaussRelEnt(mu_1, mu_2, cov_1, cov_2):
    """
    Relative Entropy between two Gaussian Distributions
    
    Args:
        mu_1 (array): Mean value of posterior Distribution
        mu_2 (array): Mean of prior Distribution
        cov_1 (narray): Covariance of posterior Distribution
        cov_2 (narray): Covariance of prior Distribution
    
    Returns:
        relative_entropy (int): Relative entropy between prior and posterior
                                distribution
    
    """
    #Compute dimensionality of the distributions
    d = len(mu_1)
    assert d == len(mu_2), "Unequal dimensionality between prior and posterior"
    
    #Compute inverse of cov_2
    try:
        inv_cov_2 = np.linalg.inv(cov_2)
    except 'LinAlgError':
        raise Exception(
                'Unable to invert the covariance matrix of the prior'
                'distribution')
    
    #Compute the product of cov_1 and inv_cov_2
    cov_product = np.matmul(inv_cov_2,cov_1)
    
    #Compute difference in mean values of posterior and prior
    mean_diff = mu_2 - mu_1
    
    res = 0.5*(np.trace(cov_product) - d - np.log(np.linalg.det(cov_1)/np.linalg.det(cov_2)))
    res += 0.5*(np.matmul(mean_diff,np.matmul(inv_cov_2,mean_diff)))
    
    return res/np.log(2)

print(GaussRelEnt(np.zeros(1), np.zeros(1), np.identity(1)*0.32, np.identity(1)))

def GaussExpectedRelEnt(cov_1, cov_2):
    """
    Expected Relative Entropy <D> between two Gaussian Distributions
    and the standard deviation of <D>
    
    Args:
        cov_1 (narray): Covariance of posterior Distribution
        cov_2 (narray): Covariance of prior Distribution
    
    Returns:
        exp_entropy (int): Expected relative entropy
        var_ent (int): Standard deviation of the expected relative entropy
    """
    #Compute dimensionality of the distributions
    d = len(cov_1[0,:])
    assert d == len(cov_1[:,0]), "New covariance matrix is not square"
    assert d == len(cov_2[0,:]), "Unequal dimensionality between prior and posterior"
    assert d == len(cov_2[:,0]), "Old covariance matrix is not square"
    
    try:
        inv_cov_2 = np.linalg.inv(cov_2)
    except 'LinAlgError':
        raise Exception(
                'Unable to invert the covariance matrix of the old '
                'distribution')
    
    det_cov_1 = np.linalg.det(cov_1)
    det_cov_2 = np.linalg.det(cov_2)
    
    cov_product = np.matmul(inv_cov_2, cov_1)
    
    #Compute the expected relative entropy
    exp_entropy = (-0.5)*(np.log2(det_cov_1/det_cov_2)) + np.trace(cov_product)
    
    mat = cov_product + np.eye(d)
    square_mat = np.matmul(mat,mat)
    
    #Compute the standard deviation of the expected entropy
    var_ent = 0.5*np.trace(square_mat)
    
    
    
    return exp_entropy, var_ent



def MonteCarloInt(func, V, x, N):
    """
    Crude Monte Carlo integration
    
    Args:
        func: Integration function
        V (int): Volume of integration domain
        x (narray): Set of random vectors in integration domain with shape
                    (n,m) where n = dimension of integration domain and 
                    m = number of samples; m >= N
        N: Number of Monte Carlo steps
    
    Returns:
        tralalala...
    """
    assert len(x[0,:]) >= N, "Sample size too small for MC steps N"
    
    #Compute expectation E[func] of func
    E_func = func(x[:,0])
    
    for i in range(N-1):
        E_func += func(x[:,i+1])
    
    #Compute estimation of integration
    Q_N = V*E_func/N
    
    #Compute variance of func
    temp_var_func = (func(x[:,0]) - E_func) **2
    for i in range(N-1):
        temp_var_func += (func(x[:,i+1]) - E_func) **2
    var_func = temp_var_func/(N-1)
    
    #Compute variance of Q_N
    var_Q_N = (var_func/N)*(V**2)
    
    return Q_N, var_Q_N
        
#---------------------------------------------------------------------------
#
#Read Data

plt.switch_backend("TkAgg")

'''SamplePath = "./tt_dd_gg_kk_samples.npy"
Auto_GaussianPath = "./4probes_auto_samples_Gaussian.npy"
Auto_SimPath = "./4probes_auto_samples_sim.npy"
Cross_GaussianPath = "./4probes_cross_samples_Gaussian.npy"
Cross_SimPath = "./4probes_cross_samples_sim.npy"

BURNIN = 200000

Samples = np.load(SamplePath)[BURNIN:,:]
Auto_Gaussian = np.load(Auto_GaussianPath)[BURNIN:,:]
Auto_Sim = np.load(Auto_SimPath)[BURNIN:,:]
Cross_Gaussian = np.load(Cross_GaussianPath)[BURNIN:,:]
Cross_Sim = np.load(Cross_SimPath)[BURNIN:,:]

NAMES = ["h", "\Omega_m", "\Omega_b", "n_s",
         "\sigma_8", "m_1", "m_2", "m_3", "m_4"]
LABELS = ["h", "\Omega_m", "\Omega_b", "n_s", 
          "\sigma_8", "m_1", "m_2", "m_3", "m_4"]'''

'''print('Test')
print(GaussRelEnt(np.zeros(5),np.zeros(5),
                  np.identity(5),np.identity(5)*2))
print(GaussExpectedRelEnt(np.identity(5), np.identity(5)*2))'''

#----------------------------------------------------------------------------


'''getdist_Cross_Gaussian = MCSamples(samples = Cross_Gaussian,
                                         names = NAMES, labels = LABELS,
                                         label="auto Gaussian")'''



#-----------------------------------------------------------------------------
#
#Compute multivariat PDF using gaussian kernels
#
#-----------------------------------------------------------------------------

'''#Build prior with uniform sampling
h_sample = np.random.uniform(0.2,1.2,650400)
Om_sample = np.random.uniform(0.1,0.7,650400)
Ob_sample = np.random.uniform(0.01,0.09,650400)
Ns_sample = np.random.uniform(0.1,1.8,650400)
S8_sample = np.random.uniform(0.4,1.5,650400)
m1_sample = np.random.uniform(-0.2,0.2,650400)
m2_sample = np.random.uniform(-0.2,0.2,650400)
m3_sample = np.random.uniform(-0.2,0.2,650400)
m4_sample = np.random.uniform(-0.2,0.2,650400)

full_prior_samples = np.array([h_sample, Om_sample, Ob_sample, Ns_sample,
                               S8_sample, m1_sample, m2_sample, m3_sample,
                               m4_sample])'''


'''kernel = gaussian_kde(Samples.T)
full_prior_kernel = gaussian_kde(full_prior_samples)'''

'''def EntropyIntegrand(x):
    """
    Evaluates the function p(x)*log(p(x)/q(x)) at the point x
    
    Args:
        x (array): Vector at which function will be evaluated
    
    Returns:
        value (int): Value of evaluated integrand
    """
    p_x = kernel.evaluate(x)
    q_x = full_prior_kernel.evaluate(x)
    
    if p_x == 0:
        return 0
    else:
        return p_x*np.log2(p_x/q_x)


RelativeEntropy, Var_Entropy = MonteCarloInt(EntropyIntegrand, 0.143616,
                                             full_prior_samples, 10000)
print("Monte Carlo Integration")
print("Relative Entropy: ", RelativeEntropy)
print("Error of Relative Entropy", Var_Entropy**0.5)'''
#----------------------------------------------------------------------------
#Approximate Sample Data and Prior Data as Gaussian Distribution

'''sample_mu, sample_cov = GaussApprox(Samples.T)
prior_mu, prior_cov = GaussApprox(full_prior_samples)

auto_gaussian_mu, auto_gaussian_cov = GaussApprox(Auto_Gaussian.T)
auto_sim_mu, auto_sim_cov = GaussApprox(Auto_Sim.T)
cross_gaussian_mu, cross_gaussian_cov = GaussApprox(Cross_Gaussian.T)
cross_sim_mu, cross_sim_cov = GaussApprox(Cross_Sim.T)'''


#-----------------------------------------------------------------------------
#Gaussianity Test using Mahalanobis distance

def GaussianityTest(data, chi_squared=True):
    """
    Will compute the Mahalanobis distance for the gauss approximation and the 
    chi-squared distribution
    
    Args:
        data (narray): array of form (m,n) where n is #variats and
                       m is #observations
    
    Returns:
        gauss_hist (np.histogram): Normalized histogram of gauss distances
    """
    data_mu = np.mean(data.T, axis=1)
    data_std = np.std(data.T, axis=1)
            
    #Standardize data
    data = ((data-data_mu)/data_std)
    
    gauss_mean, gauss_var = GaussApprox(data.T)
    inv_gauss_var = np.linalg.inv(gauss_var)
        
    gauss_distance = []
        
    for p in data:
        gauss_distance.append(mahalanobis(p, gauss_mean, inv_gauss_var))
        
    gauss_hist = np.histogram(gauss_distance, bins=np.arange(0,30,0.1), density=True)
        
    return gauss_hist


'''auto_gauss_hist = GaussianityTest(Auto_Gaussian) 
cross_gauss_hist = GaussianityTest(Cross_Gaussian)
auto_sim_hist = GaussianityTest(Auto_Sim)
cross_sim_hist = GaussianityTest(Cross_Sim)

plt.figure()
rv = chi2(9)
plt.plot(np.arange(0,30,0.2), rv.pdf(np.arange(0,30,0.2)))
plt.plot(auto_gauss_hist[1][:-1],auto_gauss_hist[0])
plt.plot(cross_gauss_hist[1][:-1],cross_gauss_hist[0])
plt.plot(auto_sim_hist[1][:-1],auto_sim_hist[0])
plt.plot(cross_sim_hist[1][:-1],cross_sim_hist[0])
plt.show()'''


#----------------------------------------------------------------------------
#Compute the Relative Entropy and Expected Entropy along with the variance


'''auto_entropy = GaussRelEnt(auto_gaussian_mu, auto_sim_mu,
                           auto_gaussian_cov, auto_sim_cov)
cross_entropy = GaussRelEnt(cross_gaussian_mu, cross_sim_mu,
                            cross_gaussian_cov, cross_sim_cov)
auto_cross_gauss_entropy = GaussRelEnt(auto_gaussian_mu, cross_gaussian_mu,
                                       auto_gaussian_cov, cross_gaussian_cov)
auto_cross_sim_entropy = GaussRelEnt(auto_sim_mu, cross_sim_mu,
                                     auto_sim_cov, cross_sim_cov)


exp_auto_ent, var_exp_auto_ent = GaussExpectedRelEnt(auto_gaussian_cov,
                                                     auto_sim_cov)
exp_cross_ent, var_exp_cross_ent = GaussExpectedRelEnt(cross_gaussian_cov,
                                                       cross_sim_cov)
exp_auto_cross_gauss, var_exp_auto_cross_gauss = GaussExpectedRelEnt(auto_gaussian_cov,
                                                                     cross_gaussian_cov)
exp_auto_cross_sim, var_exp_auto_cross_sim = GaussExpectedRelEnt(auto_sim_cov, cross_sim_cov)'''



'''print("Full multi-variat Entropy computed via gaussian approximation: ")
print(GaussRelEnt(sample_mu, prior_mu, sample_cov, prior_cov))
print("Mean from samples: ")
print("Mean: ", sample_mu)
print("Mean from prior: ")
print("Mean: ", prior_mu)

print("\n")
print("Full multi-variat Entropy computed via gaussian approximation: ")
print("\n", "Update from Auto_Sim to Auto_Gaussian")
print("", "Entropy: ", auto_entropy)
print("", "Surprise: ", auto_entropy - exp_auto_ent)
print("", "Error in expected entropy: ", (var_exp_auto_ent)**0.5)

print("\n", "Update from Cross_Sim to Cross_Gaussian")
print("", "Entropy: ", cross_entropy)
print("", "Surprise: ", cross_entropy - exp_cross_ent)
print("", "Error in expected entropy: ", (var_exp_cross_ent)**0.5)

print("\n", "Update from Auto_Gaussian to Cross_Gaussian")
print("", "Entropy: ", auto_cross_gauss_entropy)
print("", "Surprise: ", auto_cross_gauss_entropy - exp_auto_cross_gauss)
print("", "Error in expected entropy: ", (var_exp_auto_cross_gauss)**0.5)

print("\n", "Update from Auto_Sim to Cross_Sim")
print("", "Entropy: ", auto_cross_sim_entropy)
print("", "Surprise: ", auto_cross_sim_entropy - exp_auto_cross_sim)
print("", "Error in expected entropy: ", (var_exp_auto_cross_sim)**0.5)'''


'''print("Iterative Entropy for auto-gaussian-sim")
IterativeEntropy(Auto_Sim.T, Auto_Gaussian.T, 5)
print(' ')
print("Iterative Entropy for cross-gaussian-sim")
IterativeEntropy(Cross_Sim.T, Cross_Gaussian.T, 1)
print('')'''
#print("Iterative Entropy for auto-cross-gaussian")
#IterativeEntropy(Auto_Gaussian.T, Cross_Gaussian.T, 5)
#print('')
'''print("Iterative Entropy for auto-cross-sim")
IterativeEntropy(Auto_Sim.T, Cross_Sim.T, 1)'''


'''Ent_auto_gauss_prior = GaussRelEnt(auto_gaussian_mu, prior_mu,
                                   auto_gaussian_cov, prior_cov)
Ent_auto_sim_prior = GaussRelEnt(auto_sim_mu, prior_mu,
                                 auto_sim_cov, prior_cov)
delta_auto_ent = Ent_auto_gauss_prior - Ent_auto_sim_prior


Ent_cross_gauss_prior = GaussRelEnt(cross_gaussian_mu, prior_mu,
                                   cross_gaussian_cov, prior_cov)
Ent_cross_sim_prior = GaussRelEnt(cross_sim_mu, prior_mu,
                                 cross_sim_cov, prior_cov)
delta_cross_ent = Ent_cross_gauss_prior - Ent_cross_sim_prior

print("\n")
print("Difference in update from prior to Auto_Gaussian and from prior to Auto_Sim")
print("Delta Auto: ", delta_auto_ent)

print("\n")
print("Difference in update from prior to Cross_Gaussian and from prior to Cross_Sim")
print("Delta Auto: ", delta_cross_ent)'''






'''print(entropy(kernel.evaluate(full_prior_samples[:,:2000]),
              full_prior_kernel.evaluate(full_prior_samples[:,:2000]),base=2))'''

'''#-----------------------------------------------------------------------------
#PLots
y_val = [auto_entropy, exp_auto_ent, cross_entropy, exp_cross_ent]
y_err = [0, (var_exp_auto_ent)**0.5, 0, (var_exp_cross_ent)**0.5]
entropy_names = ["$D_{auto}$", "$<D_{auto}>$", "$D_{cross}$", "$<D_{cross}>$", ]

plt.figure()
plt.bar(entropy_names, y_val, yerr=y_err, capsize=30)
plt.ylabel("Relative Entropy in [bits]")
plt.savefig('./Plots/Auto_and_CrossEntropy.pdf', format='pdf')
plt.show()'''

#-----------------------------------------------------------------------------
'''g = plots.getSubplotPlotter(width_inch=14)

g.triangle_plot([getdist_Cross_Gaussian], filled=True, alpha=1)

g.settings.axes_fontsize = 14
g.settings.lab_fontsize = 20
g.settings.legend_fontsize = 14
g.settings.figure_legend_frame = False

g.settings.alpha_filled_add=1
g.export()'''

'''plt.figure()
plt.scatter(Auto_Gaussian[::2,0],Auto_Gaussian[::2,1])
plt.show()

plt.figure()
plt.scatter(Auto_Sim[::2,0],Auto_Sim[::2,1])
plt.show()

eVa, eVe = np.linalg.eig(auto_gaussian_cov)
            
#Compute transformation matrix from eigen decomposition
R, S = eVe, np.diag(np.sqrt(eVa))
T = np.matmul(R,S).T
            
#Transform data with inverse transformation matrix T^-1
inv_T = np.linalg.inv(T)
            
test = np.matmul(Auto_Gaussian, inv_T)
test2 = np.matmul(Auto_Sim, inv_T)

dim = len(test[0,:])
for k in range(dim):
    print(np.amin(test[:,k]))

plt.figure()
plt.scatter(test[::2,0],test[::2,1])
plt.show()


plt.figure()
plt.scatter(test2[::2,0],test2[::2,1])
plt.show()

test_hist = GaussianityTest(test)
test2_hist = GaussianityTest(test2)

plt.figure()
rv = chi2(9)
plt.plot(np.arange(0,30,0.2), rv.pdf(np.arange(0,30,0.2)))
plt.plot(test_hist[1][:-1],test_hist[0])
plt.plot(test2_hist[1][:-1],test2_hist[0])
plt.show()'''

