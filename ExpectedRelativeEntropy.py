# Author: Benjamin Suter
# Created: May 2020

import sys, os

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import invgamma, multivariate_normal


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def ExpectedEntropy(Compute_Cl_theory,
                    cov_matrix,
                    Cl_observed,
                    auto_corr_PDF,
                    iterations,
                    auto_corr_burn=None,
                    params=[0, 1, 2, 3, 4],
                    N=252):
    """
    Will estimate the expected relative entropy via a monte carlo simulation.
    Important, the PyCosmo function "Compute_Cl_theory" that returns the 
    theoretical values of a Cl-vector given a spesific parameter value sampled
    from the auto_corr_PDF has to be implemented!!!
    Args:
        Compute_Cl_theory (function): PyCosmo function that takes a list/array
                                      of parameter values as an input and returns
                                      a list/array of the corresponding Cl_data
                                      vectors.
        cov_matrix (str or ndarray): covariance matrix of the multi-probe
                                     power spectrum.
        Cl_observed (str or ndarray): Observed data vector of the multi-probe
                                      power spectrum.
        auto_corr_PDF (str or ndarray): Full posterior distribution of the 
                                        parameters obtained from the auto-correlated
                                        probes.
        iterations (int): Number of iterations performed to estimate the 
                          eastimated relative entropy
        auto_corr_burn (int): Number of data points cut off at begining of auto_corr_PDF
                              data.
                              Default is 50%
        params (list): List which indicates what varied parameters should be
                       used when computing the relative entropy
                       h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                       m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
                       Default is [0,1,2,3,4]
        N (int): Number of realizations
                 Default set to N = 252
    Returns:
        ent (float): Numerical estimation of the expected relative entropy
    """

    #Load the covariance matrix, the observed Cl-vector and the posterior
    #probability distribution of the auto-correlated probe parameters
    if isinstance(cov_matrix, str):
        cov_matrix = np.load(cov_matrix)

    if isinstance(Cl_observed, str):
        Cl_observed = np.load(Cl_observed)

    if isinstance(auto_corr_PDF, str) and not auto_corr_burn:
        tmp = np.load(auto_corr_PDF)
        burn = int(np.shape(tmp)[0] / 2)
        del (tmp)
        auto_corr_PDF = np.load(auto_corr_PDF)[burn:, params]
    elif isinstance(auto_corr_PDF, str):
        auto_corr_PDF = np.load(auto_corr_PDF)[auto_corr_burn:, params]

    #Generate kernel from auto-correlated PDF in order to sample new parameters
    kernel = gaussian_kde(auto_corr_PDF.T)
    Combinations = kernel.resample(size=iterations).T

    #Stop PyCosmo from Spamming!!
    blockPrint()

    #PyCosmo part, compute Cl_i for every sampled parameter theta_i
    #The function "Compute_Cl_theory" has to be implemented!!
    #--------------------------------------------------------------------------
    #Implement the function "Compute_Cl_theory()" and pass it as an argument to
    #this function "ExpectedEntropy()"
    Cl_theory = np.asarray(Compute_Cl_theory(Combinations))
    #--------------------------------------------------------------------------
    enablePrint()

    #Generate mock data
    #Parameters used in the inverse gamma distribution
    p = len(Cl_observed)
    nu = N - p
    lam = N - 1
    alpha = nu / 2
    beta = lam / 2

    v_sim = invgamma.rvs(alpha, scale=beta, size=iterations)
    Y_sim = multivariate_normal.rvs(mean=np.zeros(p),
                                    cov=cov_matrix,
                                    size=iterations)

    #Sample the mock data
    Cl_sim = []
    for i in range(iterations):
        Cl_sim.append(Cl_theory[i] + np.sqrt(v_sim[i]) * Y_sim[i])
    Cl_sim = np.asarray(Cl_sim)

    icov_scal = np.linalg.inv(cov_matrix * 1e8)

    #Generate Likelihood matrix
    M_tot = []
    for i in range(iterations):
        M_cl = (Cl_sim[i] * 1e4 - Cl_theory * 1e4).T
        M_i = M_cl.T @ icov_scal @ M_cl
        M_i = np.diag(M_i)
        M_tot.append(M_i)

    #Comute the numerical estimate of the expected entropy
    part1 = 0
    part2 = 0
    for i in range(iterations):
        val1 = M_tot[i][i]
        tmp1 = (1 + val1 / (N - 1))**(-N / 2.)
        part1 += np.log(tmp1)

        val2 = sum((1 + M_tot[i] / (N - 1))**(-N / 2.)) / iterations
        part2 -= np.log(val2)
    ent = (part1 + part2) / (iterations * np.log(2))

    return ent
