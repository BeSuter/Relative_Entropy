#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:24:10 2020

@author: BenjaminSuter
"""

import yaml
import numpy as np

from RelativeEntropyFunctions import MonteCarloENTROPY, LoadAndComputeEntropy
from surprise import Surprise


def EntropyComparison(data, avrg, target, MCsteps=100000):
    """
    Will compute the relative entropy via monte carlo integration and the
    Surprise package and write the results to a YAML file.
    
    Args:
        data (list): A list of tuples containing the path to the prior and 
                     posterior distributions
        avrg (int): Number of computations over which to average the results
                    of the MC entropy and the gauss approximated entropy
        target (str): Target to where to save the YAML file
        MCsteps (int): Number of sample points used when computing the entropy
                       via Monte Carlo integration
    """
    EntropyValues = {'Names': ['MC Entropy', 'MC Error', 'Gauss Entropy',
                               'Gauss Error', 'Expected Entropy','Surprise',
                               'sD', 'Gauss Entropy converged',
                               'Gauss Error converged',
                               'Expected Entropy converged','Surprise converged',
                               'sD converged', 'Entropy no BC',
                               'Expected Entropy no BC']}
    for names in data:
        tmp_mc = []
        tmp_gauss_ent = []
        tmp_gauss_exp = []
        tmp_gauss_S = []
        tmp_gauss_sD = []
        tmp_gauss_conv_ent = []
        tmp_gauss_conv_exp = []
        tmp_gauss_conv_S = []
        tmp_gauss_conv_sD = []
        
        size1 = avrg
        size2 = avrg
        for i in range(avrg):
            tmp_mc.append(MonteCarloENTROPY(names[0],names[1], MCsteps))
            gauss_approx = LoadAndComputeEntropy(names[0], names[1], steps=1)
            tmp = [gauss_approx[2][-1], gauss_approx[3][-1],
                   gauss_approx[-3][-1], gauss_approx[-4][-1]]
            if not gauss_approx[-1] and not None in tmp:
                tmp_gauss_ent.append(gauss_approx[2][-1])
                tmp_gauss_exp.append(gauss_approx[3][-1])
                tmp_gauss_S.append(gauss_approx[-3][-1])
                tmp_gauss_sD.append(gauss_approx[-2][-1])
            else:
                size1 -= 1
            gauss_approx_conv = LoadAndComputeEntropy(names[0], names[1], steps=300)
            tmp = [gauss_approx_conv[2][-1], gauss_approx_conv[3][-1], 
                   gauss_approx_conv[-3][-1], gauss_approx_conv[-4][-1]]
            if not gauss_approx_conv[-1] and not None in tmp:
                tmp_gauss_conv_ent.append(gauss_approx_conv[2][-1])
                tmp_gauss_conv_exp.append(gauss_approx_conv[3][-1])
                tmp_gauss_conv_S.append(gauss_approx_conv[-3][-1])
                tmp_gauss_conv_sD.append(gauss_approx_conv[-2][-1])
            else:
                size2 -= 1
            
        mc_error = np.around(np.sqrt(np.var(tmp_mc)), decimals=3)
        try:
            gauss_error = np.around(np.sqrt(np.var(tmp_gauss_ent)), decimals=3)
        except:
            gauss_error = None
        try:
            gauss_conv_error = np.around(np.sqrt(np.var(tmp_gauss_conv_ent)), decimals=3)
        except:
            gauss_conv_error = None
        
        mc_ent = np.around(sum(tmp_mc)/avrg, decimals=3)
        if size1 > 0:
            gauss_ent = np.around(sum(tmp_gauss_ent)/size1, decimals=3)
            gauss_exp = np.around(sum(tmp_gauss_exp)/size1, decimals=3)
            gauss_S = np.around(sum(tmp_gauss_S)/size1, decimals=3)
            gauss_sD = np.around(sum(tmp_gauss_sD)/size1, decimals=3)
        else:
            gauss_ent = None
            gauss_exp = None
            gauss_S = None
            gauss_sD = None
        if size2 > 0:
            gauss_conv_ent = np.around(sum(tmp_gauss_conv_ent)/size2, decimals=3)
            gauss_conv_exp = np.around(sum(tmp_gauss_conv_exp)/size2, decimals=3)
            gauss_conv_S = np.around(sum(tmp_gauss_conv_S)/size2, decimals=3)
            gauss_conv_sD = np.around(sum(tmp_gauss_conv_sD)/size2, decimals=3)
        else:
            gauss_conv_ent = None
            gauss_conv_exp = None
            gauss_conv_S = None
            gauss_conv_sD = None
        
        
        tmp = np.load(names[0])
        pri_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        data1 = np.load(names[0])[pri_burn:,:5]
        
        tmp = np.load(names[1])
        pri_burn = int(np.shape(tmp)[0]/2)
        del(tmp)
        data2 = np.load(names[1])[pri_burn:,:5]
        sc = Surprise()
        try:
            vals = sc(data1, data2, mode='add')[0:2]
            vals = np.around(vals, decimals=3)
        except AssertionError:
            vals = [None, None]
        ent_noBC = vals[0]
        EXPent_noBC = vals[1]
        del(data1)
        del(data2)
        
        name = names[0][10:].replace('.npy', '')
        EntropyValues[name] = ['%s' % mc_ent, 
                               '%s' % mc_error,
                               '%s' % gauss_ent,
                               '%s' % gauss_error,
                               '%s' % gauss_exp,
                               '%s' % gauss_S,
                               '%s' % gauss_sD, 
                               '%s' % gauss_conv_ent,
                               '%s' % gauss_conv_error,
                               '%s' % gauss_conv_exp,
                               '%s' % gauss_conv_S,
                               '%s' % gauss_conv_sD,
                               '%s' % ent_noBC,
                               '%s' % EXPent_noBC]
    
    with open(target, 'w') as yaml_file:
        yaml.dump(EntropyValues, stream=yaml_file, default_flow_style=False)


#------------------------------------------------------------------------------
#Example will compare the MC entropy to single Box-Cox, converged Box-Cox and
#no Box-Cox and write all results to a YAML file.

#Replace 'path_to_auto_corr_data' and 'path_to_cross_corr_data'!!
auto_data = 'path_to_auto_corr_data'
cross_data = 'path_to_cross_corr_data'

ExampleData = [(auto_data, cross_data)]

EntropyComparison(ExampleData, 2, './ExampleResults.yml',
                  MCsteps=10000)

#Example of using the Box-Cox algorithm
Results = LoadAndComputeEntropy(auto_data, cross_data, steps=300)
print('Results from Box-Cox')
print('Entropy: ', Results[2][-1])
print('Expected Entropy: ', Results[3][-1])

#Example of using Monte Carlo Entropy
MCResults = MonteCarloENTROPY(auto_data, cross_data, 10000)
print('Monte Carlo Entropy')
print('Entropy: ', MCResults)

