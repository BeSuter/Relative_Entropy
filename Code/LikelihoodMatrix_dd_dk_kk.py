#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: BenjaminSuter
"""
import sys, os
import numpy as np
import PyCosmo
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.stats import invgamma, multivariate_normal
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

dd_gg_gauss = "./Samples/dd_gg_samples_Gaussian.npy"
log = open("ExpectedEntropyLog.txt", 'w')
log.write('Start\n')


'''Combinations = [[0.7,0.3,0.05,0.961,0.8]]'''

#Will compute the Cl_i
def Compute_Cl_theory(Combinations):
    #M_L = []
    log.write('start of Compute_Cl_theory\n')
    
    w_bin = 100
    l_min = 100
    l_max = 1000
    no_real = 252
    ells = np.arange(1,3080)
    
    param_array_dd_dg_gg = (('custom', 'custom', 'deltag', 'deltag', None, 'nonlinear'),
                            ('custom', 'custom', 'gamma', 'deltag', None, 'nonlinear'),
                            ('custom', 'custom', 'gamma', 'gamma', None, 'nonlinear'))
    ############# set up PyCosmo #############
    path = '/cluster/home/besuter/data_vec_and_cov'
    
    cosmo = PyCosmo.Cosmo()
    
    nz_smail_txt = '/cluster/home/besuter/DataSim/nz_smail'	# used redshift-distribution
    
    # set values for fiducial cosmology (data vector and covariance matrix is computed for this cosmology)
    
    Omega_m_fiducial = 0.3
    Omega_l_fiducial = 'flat'
    h_fiducial = 0.7
    Omega_b_fiducial = 0.05
    omega_suppress = 'False'
    suppress_rad = 'False'
    sigma8 = 0.8
    ns = 0.961
    
    log.write('Fiducial cosmology\n')
    
    cosmo.set(suppress_rad=omega_suppress)
    cosmo.set(h=h_fiducial)
    cosmo.set(omega_m = Omega_m_fiducial)
    cosmo.set(omega_b = Omega_b_fiducial)
    cosmo.set(pk_norm = sigma8)
    cosmo.set(n=ns)
    
    ############# import data and covariance matrix #############
    #Changed to dd_dk_kk, don't trust the var. names...
    cl_obs_dd_dg_gg = np.load(path + '/cl_dd_dg_gg_100_1000_w100_fullsky.npy')
    
    cov_gaussian_dd_dg_gg = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky.npy')
    
    ############# compute PyCosmo C_ells once at fiducial cosmology for normalization #############
    obs = PyCosmo.Obs()
    
    ells = np.arange(1,3080)
    log.write('Calculate normalizations\n')
    for (nz1, nz2, probe1, probe2, zrecomb, perturb) in param_array_dd_dg_gg:
        #print('-----------------------')
        #print('probes:', probe1, probe2)
        
        clparams = {'nz': [nz1, nz2],
                    'path2zdist': [nz_smail_txt, nz_smail_txt],
                    'probes': [probe1, probe2],
                    'zrecomb': zrecomb,
                    'perturb': perturb,
                    'normalised': False,
                    'bias': 1.,
                    'm': 0,
                    'ngauss': 700,
                    'z_grid': np.array([0.001, 1.75, 1000])
                    }
        cl_temp = obs.cl(ells, cosmo, clparams)
        
        if probe1 == 'deltag' and probe2 == 'deltag':
            norm_dd = 0.2 / np.mean(cl_temp[l_min:l_max])
        if probe1 == 'gamma' and probe2 == 'deltag':
            norm_gd = 0.2 / np.mean(cl_temp[l_min:l_max])
        if probe1 == 'gamma' and probe2 == 'gamma':
            norm_gg = 0.2 / np.mean(cl_temp[l_min:l_max])
    log.write('Start iteration of combinations\n')
            
    #Iterate through all the parameter combinations
    Cl_theory = []
    count_num = 0
    for theta in Combinations:
        cl_start = datetime.now()
        enablePrint()
        log.write('Count number: %s \n' %count_num)
        log.write('Param. combination %s \n' %theta)
        count_num += 1
        blockPrint()
        h = theta[0]
        Om = theta[1]
        Ob = theta[2]
        ns = theta[3]
        s8 = theta[4]
        
        cosmo.set(h=h)
        cosmo.set(omega_m=Om)
        cosmo.set(omega_b = Ob)
        cosmo.set(n=ns)
        cosmo.set(pk_norm=s8)
        
        obs = PyCosmo.Obs()
        
        for (nz1, nz2, probe1, probe2, zrecomb, perturb) in param_array_dd_dg_gg:
            clparams = {'nz': [nz1, nz2],
                        'path2zdist': [nz_smail_txt, nz_smail_txt],
                        'probes': [probe1, probe2],
                        'zrecomb': zrecomb,
                        'perturb': perturb,
                        'normalised': False,
                        'bias': 1.,
                        'm': 0,
                        'ngauss': 700,
                        'z_grid': np.array([0.001, 1.75, 1000])
                        }
            cl_temp = obs.cl(ells, cosmo, clparams)
            
            if probe1 == 'deltag' and probe2 == 'deltag':
                cl_multi_pycosmo_arr = cl_temp[l_min:l_max] * norm_dd
            elif probe1 == 'gamma' and probe2 == 'gamma':
                cl_multi_pycosmo_arr = np.hstack((cl_multi_pycosmo_arr, cl_temp[l_min:l_max] * norm_gg))
            else:
                cl_multi_pycosmo_arr = np.hstack((cl_multi_pycosmo_arr, cl_temp[l_min:l_max] * norm_gd))
            cl_pycosmo = cl_multi_pycosmo_arr
            
            # bin pycosmo-vector (this is needed since the data-vector and covariance matrix are binned in this way)
            n_bin = int(len(cl_pycosmo)/w_bin)
            
            cl_pycosmo_binned = np.zeros(n_bin)
            for i in range(n_bin):
                cl_pycosmo_binned[i] = np.mean(cl_pycosmo[i * w_bin : i * w_bin + w_bin])
        
        # compute likelihood with pycosmo-vector (for this set of parameters) and imported data-vector, covariance matrix
        cl_obs = cl_obs_dd_dg_gg
        cov = cov_gaussian_dd_dg_gg
        
        l_obs = np.arange(len(cl_obs))
        assert cl_pycosmo_binned.shape == cl_obs.shape
        
        #log-likelihood (the data-vector and covariance are multiplied by a large no. to ensure stable invers of the covariance matrix
        # this factor drops out)
        #cl_diff = cl_obs*1e4 - cl_pycosmo_binned*1e4
        #part1 = (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(no_real - 1.)
        #print(part1)
        #part2 = (1 + part1)**(-no_real / 2.)
        #print(part2)
        #print(part2*(np.linalg.det(cov)**(-0.5)))
        #print('')
        #lnL     = (-no_real / 2.)*np.log(1 + (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(no_real - 1.) )
        #likelihood = np.exp(lnL)	#if you need the likelihood instead of log-likelihood
        
        #M_L.append(likelihood)
        Cl_theory.append(cl_pycosmo_binned)
        enablePrint()
        log.write('Cl done\n')
        log.write('Time: %s \n' %(datetime.now() - cl_start))
        blockPrint()
    return Cl_theory
#-----------------------------------------------------------------------------
#Main script starts here
#-----------------------------------------------------------------------------



#path = '/cluster/home/besuter/DataSim'
#cov_gaussian_dd_dg_gg = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky.npy')
#cl_obs_dd_dg_gg = np.load(path + '/cl_dd_dg_gg_100_1000_w100_fullsky.npy')
#cov = cov_gaussian_dd_dg_gg

name = dd_gg_gauss[10:].replace('.npy', '')
path = '/cluster/home/besuter/data_vec_and_cov'
cov_gaussian_dd_dk_kk = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky.npy')
cl_obs_dd_dk_kk = np.load(path + '/cl_dd_dg_gg_100_1000_w100_fullsky.npy')
cov = cov_gaussian_dd_dk_kk

#Some values used later to generate the mock data
p = len(cl_obs_dd_dk_kk)
N = 252
nu = N-p
lam = N-1

alpha = nu/2
beta = lam/2
size=1300

log.write("Pobe: %s \n" %name )
log.write("Step size: %s \n" %size)

start1 = datetime.now()

tmp = np.load(dd_gg_gauss)
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
data = np.load(dd_gg_gauss)[pri_burn:,:5]
log.write('Loaded data\n')
kernel = gaussian_kde(data.T)
log.write('Kernel made\n')
Combinations = kernel.resample(size=size).T
log.write('Samples made\n')


#Stop PyCosmo from Spamming!!
blockPrint()

#PyCosmo part, compute Cl_i for every theta_i
Cl_theory = np.asarray(Compute_Cl_theory(Combinations))

enablePrint()



log.write('Estimate Entropy\n')


#Generate mock data
v_sim = invgamma.rvs(alpha, scale=beta, size=size)
Y_sim = multivariate_normal.rvs(mean=np.zeros(p), cov=cov, size=size)

Cl_sim = []
for i in range(size):
    Cl_sim.append(Cl_theory[i] + np.sqrt(v_sim[i])*Y_sim[i])
Cl_sim = np.asarray(Cl_sim)

icov_scal = np.linalg.inv(cov * 1e8)


#Generate Likelihood matrix
M_tot = []
for i in range(size):
    M_cl = (Cl_sim[i]*1e4 - Cl_theory*1e4).T
    M_i = M_cl.T@icov_scal@M_cl
    M_i = np.diag(M_i)
    M_tot.append(M_i)

    
#Comute the numerical estimate of the expected entropy
part1 = 0
part2 = 0
for i in range(size):
    val1 = M_tot[i][i]
    tmp1 = (1 + val1/(N-1))**(-N/2.)
    part1 += np.log(tmp1)
    
    val2 = sum((1 + M_tot[i]/(N-1))**(-N/2.))/size
    part2 -= np.log(val2)

ent = (part1 + part2)/(size*np.log(2))
log.write('Entropy done\n')
log.close()
time = datetime.now()-start1

#Append results so a file
f = open("ExpectedEntropyResults.txt", 'a')
f.write("Pobe: %s \n" %name )
f.write("Time: %s \n" %time)
f.write("Step size: %s \n" %size)
f.write("Expected Entropy: %s \n" %ent)
f.write('\n \n \n')
f.close()  