#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  27 11:44:14 2020

@author: BenjaminSuter
"""
import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2
from getdist import MCSamples, plots
from sklearn.covariance import EmpiricalCovariance
from IterCompRelEnt import MonteCarloENTROPY


def GaussianityTest(data):
    """
    Will compute the Mahalanobis distance for the gauss approximation and the 
    chi-squared distribution
    
    Args:
        data (narray): array of form (m,n) where n is #variats and
                       m is #observations
    
    Returns:
        gauss_hist (np.histogram): Normalized histogram of gauss distances
    """
    emp_cov = EmpiricalCovariance().fit(data)
    mds = emp_cov.mahalanobis(data)
        
    gauss_hist = np.histogram(mds, bins=np.arange(0,30,0.2), density=True)
        
    return gauss_hist



def Chi2ComparisonPlot(data, target, labels=[], dim=5, form='pdf'):
    """
    Will plot the normalised histogram of the Mahalanobis distance compared
    to the expected chi-squared distribution
    
    Args:
        data (list): List containing Mahalanobis distance histogramms
        labels (list): List of strings containing the labels of the plotted
                        data in the same order as 'data'
        target (str): Target string to where the plot will be saved to
        dim (int): Degrees of freedom for the chi-squared distribution, default
                   is 5
        form (str): Format in which to save the figure, default is 'pdf'
    """
    
    fig = plt.figure()
    
    legend = True
    if len(labels) == 0:
        legend = False
        #Fill labels list with dummy strings
        for i in range(len(data)):
            labels.append('')
    
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                       xlabel='$d_{Mahalanobis}$',
                       title='Gaussianity test')
    rv = chi2(dim)
    label_index = 0
    
    #Check if multiple histogramms are plotted
    if len(data) > 1:
        for hist in data:
            ax1.step(hist[1][:-1],hist[0], label=labels[label_index])
            label_index += 1
    else:
        ax1.step(data[0][1][:-1],data[0][0], label=labels[label_index])
    
    ax1.plot(np.arange(0,30,0.2), rv.pdf(np.arange(0,30,0.2)),
             color='black', linestyle='dashed',
             linewidth=2, label='chi2')
    
    if legend:
        ax1.legend()
    plt.savefig(target, format=form)



def ConvergencePlot(data, target, labels=[], form='pdf'):
    """
    Will plot the computed relative entropy [Bits] against the iteration step
    showing the convergence behavior
    
    Args:
        data (list): List of the data arrays to be plotted, where data arrays
                     containe the relative entropy points
        target (str): Target string to where the plot will be saved to
        labels (list): List of strings containing the labels of the plotted
                       data in the same order as 'data'
        form (str): Format in which to save the figure, default is 'pdf'
    """
    fig = plt.figure()
    
    n = len(data)
    legend = True
    if len(labels) == 0:
        legend = False
        #Fill labels with dummy values
        for i in range(n):
            labels.append('')
    
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                       xlabel='$N_{Iterations}$',
                       ylabel='Relative Entropy [Bits]',
                       title='Relative Entropy Convergence')
    label_index = 0
    if n > 1:
        for data_set in data:
            ax1.plot(range(len(data_set)),data_set, 'o',
                     label=labels[label_index])
            label_index += 1
    else:
        ax1.plot(range(len(data[0])),data[0],'o',
                 label=labels[0])
    
    if legend:
        ax1.legend()
    plt.savefig(target, format=form)



def RelativeEntropyBarPlot(data, labels, target, surprise=[], form='pdf'):
    """
    Will make a bar plot of the given data. If surprise is provided, will
    additionally plot the surprise on top of the relative entropy.
    
    Args:
        data (list): List of the relative entropy (int) for each data set
        labels (list): List of the data set names for which the relative
                       entropy is provided
        target (str): Target string to where the plot will be saved to
        surprise (list): List of the surprise (int) for each data set,
                         default is an empty list where no surprise will be 
                         plotted
        form (str): Format in which to save the figure, default is 'pdf'
    """
    fig = plt.figure(figsize=(14,6))
    
    ax1 = fig.add_axes([0.2, 0.15, 0.8, 0.75])
    ax1.set_xlabel('Information gains [Bits]')
    
    ax1.barh(labels, data, label='Relative Entropy')
    
    if not len(surprise) == 0:
        ax1.barh(labels,surprise, label='Surprise')
    
    ax1.legend()
    plt.savefig(target, format=form)



def GetDistContourPlot(data_path, labels, title, target,burning=200000):
    """
    Will load data from path and create a triangle plot using getdist
    
    Args:
        data_path (str): String containing the path to where the data is located
                         Or pre-loaded data as (ndarray)
        labels (list): List of strings containing the labels for the parameters
                       shown in the plot
        title (str): Title of the plot
        target (str): String indicating target path to where the plot will be 
                      saved
                      e.g ./example.pdf
        burning (int): Number of data points cut off at begining of sample 
                       data; default is 200000
    """
    if isinstance(data_path, str):
        data = np.load(data_path)[burning:,:]
    elif isinstance(data_path, np.ndarray):
        data = data_path
        burning = 0
    getdist_data = MCSamples(samples = data,
                             names = labels, labels = labels,
                             label=title)
    del(data)
    
    g = plots.getSubplotPlotter(width_inch=14)
    
    g.triangle_plot([getdist_data], filled=True, alpha=1)
    
    g.settings.axes_fontsize = 14
    g.settings.lab_fontsize = 20
    g.settings.legend_fontsize = 14
    g.settings.figure_legend_frame = False
    
    g.settings.alpha_filled_add=1
    g.export(target)



def MonteCarloHistogram(prior_data, post_data, iterations, target=None,
                        MCsteps=500, pri_burn=None, post_burn=None, 
                        bins=30, form='pdf', entropy_str=False):
    """
    Will create a histogramm plot of the MonteCarlo results
    
    Args:
        prior_data (str): String containing the path to the data samples
                          Or pre-loaded data as (ndarray) with shape (N_samples, n_dim)
        post_data (str): String containing the path to the data samples
                         Or pre-loaded data as (ndarray) with shape (N_samples, n_dim)
        iterations(int): Number of iteration steps
        target (str): Target string to where the image will be saved
                      default is None
        MCsteps (int): Number of sample points used in each MC
        pri_burn (int): Number of data points cut off at begining of prior
                        sample data; default is 50%
        post_burn (int): Number of data points cut off at begining of post
                         sample data; default is 50%
        form (str): Format in which the image will be saved
        save_file (bool): If True will return the mean entropy and estimated
                          error as a string in order to be written to a file
    """
    fig = plt.figure()
    
    if target:
        target = target
    elif isinstance(prior_data, str):
        name = prior_data[10:].replace('.npy', '')
        target = '../Plots/MonteCarloHistogram/AutoCross_%s.pdf' % name
    else:
        if not os.path.isfile('./Plots/MonteCarloHistogram/NotSpecified.pdf'):
            target = '../Plots/MonteCarloHistogram/NotSpecified.pdf'
        else:
            i = 1
            while os.path.isfile('../Plots/MonteCarloHistogram/NotSpecified%s.pdf' % i):
                i += 1
            target = '../Plots/MonteCarloHistogram/NotSpecified%s.pdf' % i
    
    entropy = []
    
    if entropy_str:
        error = []
        for i in range(iterations):
            ent, err = MonteCarloENTROPY(prior_data, post_data, MCsteps,
                                         error=True, pri_burn=pri_burn,
                                         post_burn=post_burn)
            entropy.append(ent)
            error.append(err)
        error = np.around(sum(error)/len(error), decimals=3)
    else:
        for i in range(iterations):
            entropy.append(MonteCarloENTROPY(prior_data, post_data, MCsteps,
                                             pri_burn=pri_burn,
                                             post_burn=post_burn))
    
    mean = np.around(sum(entropy)/len(entropy), decimals=3)
    entropy_hist = np.histogram(entropy,bins=bins, density=True)
    
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75],
                       xlabel='$D_{MC}(f||g)$',
                       ylabel='$Probability$',
                       title='MonteCarloEntropy')
    ax1.step(entropy_hist[1][:-1],entropy_hist[0],
             label='%d its. of $D_{MC}(%d)$' %(iterations, MCsteps))
    ax1.axvline(x=mean, color='r', label='Mean value of %s [Bits]' % mean)
    ax1.legend()
    plt.savefig(target, format=form)
    
    if entropy_str:
        return '{} +/- {}'.format(mean, error)



def PrettyName(name):
    """
    """
    tmp = name.replace('tt', 'T')
    tmp = tmp.replace('gg', 'G')
    tmp = tmp.replace('kk', 'K')
    tmp = tmp.replace('dd', 'D')
    tmp = tmp.replace('samples', '')
    tmp = tmp.replace('Gaussian', 'gauss')
    tmp = tmp.replace('_', '')
    tmp = tmp.replace('/','')
    final = tmp.replace('nonuisance', '_noN')
    return final


def EntropyComparisonPlot(data_path, target):
    """
    
    ToDo:
        Make YAML file optional. Take directly the dictionary from
        EntropyComparison()...
    
    
    
    Will load values from a YAML file and plot the relative entropy as well
    as the relative difference between the Monte Carlo integration and the 
    gauss approximation
    
    Args:
        data_path (str): String to the YAML file
        target (str): String to where the plot will be saved
    """
    with open(data_path, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    data.pop('Names')
    
    keys = []
    MCresults = []
    MCerror = []
    GaussResults = []
    GaussError = []
    RelDiff = []
    for key in data:
        tmp = []
        keys.append(PrettyName(key))
        for string in data[key]:
            if string == 'None':
                tmp.append(np.NAN)
            else:
                test = re.findall(r'(-?\d+\.\d+)', string)
                res = [test[0]]
                tmp.append(float(res[0]))
        MCresults.append(tmp[0])
        MCerror.append(tmp[1])
        GaussResults.append(tmp[2])
        GaussError.append(tmp[3])
    for i in range(len(MCresults)):
        if MCresults[i] > GaussResults[i]:
            tmp = (1 - (GaussResults[i]/MCresults[i]))*100
            RelDiff.append(tmp)
        else:
            tmp = (1 - (MCresults[i]/GaussResults[i]))*100
            RelDiff.append(tmp)
    
    fig = plt.figure(figsize=(14,6)) 
    ax1 = fig.add_axes([0.1, 0.35, 0.8, 0.6],
                       ylabel='Relative Entropy [Bits]',
                       title='Comparison of Entropy')
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.2],
                       sharex=ax1,
                       ylabel='Diff. in %')
    
    ax1.errorbar(keys, MCresults, yerr=MCerror, marker='*', ls='none',
                 label='Monte Carlo Entropy')
    ax1.errorbar(keys, GaussResults, yerr=GaussError, marker='o', ls='none',
                 label='Gauss Entropy')
    ax2.bar(keys, RelDiff, width=0.2, label='Rel. Diff')
    plt.savefig(target, format='pdf')
    ax1.legend()



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
    tmp = tmp.replace('/','')
    if tmp[-1] == "g":
        tmp = "Gaussian"
    if tmp[-1] == "s":
        tmp = "Sim"
    if tmp[-2:] == "gN":
        tmp = "Gaussian no nuisance"
    if tmp[-2:] == "sN":
        tmp = "Sim no nuisance"
    return tmp

def YAMLplot(file_path, target_path):
    with open(file_path, 'r') as yaml_file:
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
    plt.savefig(target_path, format='pdf')
    
