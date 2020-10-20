#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: BenjaminSuter
"""
import numpy as np
import PyCosmo
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.stats import invgamma, multivariate_normal
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')

dd_gg_gauss = "./Samples/dd_dk_kk_samples_Gaussian_nonuisance.npy"

'''Combinations = [[0.7,0.3,0.05,0.961,0.8]]'''


def LikelihoodMatrix(Combinations):
    #M_L = []
    
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
    
    cosmo.set(suppress_rad=omega_suppress)
    cosmo.set(h=h_fiducial)
    cosmo.set(omega_m = Omega_m_fiducial)
    cosmo.set(omega_b = Omega_b_fiducial)
    cosmo.set(pk_norm = sigma8)
    cosmo.set(n=ns)
    
    ############# import data and covariance matrix #############
    #Changed to dd_dk_kk, don't trust the var. names...
    cl_obs_dd_dg_gg = np.load(path + '/cl_dd_dk_kk_100_1000_w100_fullsky.npy')
    
    cov_gaussian_dd_dg_gg = np.load(path + '/cov_dd_dk_kk_100_1000_w100_n252_fullsky.npy')
    
    ############# compute PyCosmo C_ells once at fiducial cosmology for normalization #############
    obs = PyCosmo.Obs()
    
    ells = np.arange(1,3080)
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
            
    Cl_theory = []
    for theta in Combinations:
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
            else:
                cl_multi_pycosmo_arr = np.hstack((cl_multi_pycosmo_arr, cl_temp[l_min:l_max] * norm_gg))
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
    #return M_L
    return Cl_theory

#path = '/cluster/home/besuter/DataSim'
#cov_gaussian_dd_dg_gg = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky.npy')
#cl_obs_dd_dg_gg = np.load(path + '/cl_dd_dg_gg_100_1000_w100_fullsky.npy')
#cov = cov_gaussian_dd_dg_gg

path = '/cluster/home/besuter/data_vec_and_cov'
cov_gaussian_dd_dk_kk = np.load(path + '/cov_dd_dk_kk_100_1000_w100_n252_fullsky.npy')
cl_obs_dd_dk_kk = np.load(path + '/cl_dd_dk_kk_100_1000_w100_fullsky.npy')
cov = cov_gaussian_dd_dk_kk

p = len(cl_obs_dd_dk_kk)
N = 252
nu = N-p
lam = N-1

alpha = nu/2
beta = lam/2
size=2

start1 = datetime.now()

tmp = np.load(dd_gg_gauss)
pri_burn = int(np.shape(tmp)[0]/2)
del(tmp)
data = np.load(dd_gg_gauss)[pri_burn:,:5]

kernel = gaussian_kde(data.T)

Combinations = kernel.resample(size=size).T

Cl_theory = np.asarray(LikelihoodMatrix(Combinations))


v_sim = invgamma.rvs(alpha, scale=beta, size=size)
Y_sim = multivariate_normal.rvs(mean=np.zeros(p), cov=cov, size=size)

Cl_sim = []
for i in range(size):
    Cl_sim.append(Cl_theory[i] + np.sqrt(v_sim[i])*Y_sim[i])
Cl_sim = np.asarray(Cl_sim)

icov_scal = np.linalg.inv(cov * 1e8)

M_tot = []
for i in range(size):
    M_cl = (Cl_sim[i]*1e4 - Cl_theory*1e4).T
    M_i = M_cl.T@icov_scal@M_cl
    M_i = np.diag(M_i)
    M_tot.append(M_i)
part1 = 0
part2 = 0
for i in range(size):
    val1 = M_tot[i][i]
    tmp1 = (1 + val1/(N-1))**(-N/2.)
    part1 += np.log(tmp1)
    
    val2 = sum((1 + M_tot[i]/(N-1))**(-N/2.))/size
    part2 -= np.log(val2)

ent = (part1 + part2)/(size*np.log(2))
print('Time1 : ', datetime.now()-start1)
print(ent)    
print('')

'''M_L = []
start2 = datetime.now()
for i in range(size):
    M_Li = []
    for j in range(size):
        cl_diff = Cl_sim[i]*1e4 - Cl_theory[j]*1e4
        tmp = (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(N - 1.)
        M_Li.append((1+tmp)**(-N/2.))
    M_L.append(M_Li)

M_L = np.asarray(M_L)

sum1 = 0
for i in range(size):
    sum1 -= np.log(sum(M_L[:,i])/size)/(size*np.log(2))
sum2 = 0
for i in range(size):
    sum2 += np.log(M_L[i,i])/(size*np.log(2))
    
entropy = (sum1 + sum2)
print('Time2 : ', datetime.now()-start2)
print(entropy)'''
    

'''Cl_sim = []
Cl_diff = []

for i in range(size):
    tmp = np.sqrt(v_sim[i])*Y_sim[i]
    Cl_sim.append(Cl_theory + tmp)
    Cl_diff.append(tmp)

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                   xlabel='Bins',
                   ylabel='$C_l$',
                   title='C_l sampling')
x = range(1,p+1)
ax1.plot(x,Cl_theory, color='black', linewidth=2, label='Cl_theory')

for C_l in Cl_sim:
    ax1.plot(x,C_l, linestyle='dashed')

for C_l in Cl_diff:
    ax1.plot(x,C_l, 'o')
plt.legend()
plt.savefig('/cluster/home/besuter/Cl_plot.pdf', format='pdf')

#Evaluate the likelihood
L = []

for C_l in Cl_sim:
    cl_diff = C_l*1e4 - Cl_theory*1e4
    part1 = (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(N - 1.)
    det = np.linalg.det(cov)
    L.append(((1+part1)**(-N/2)))
print(L)'''

    