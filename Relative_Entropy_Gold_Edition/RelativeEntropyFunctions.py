# Author: Benjamin Suter
# Created: May 2020

import sys
import logging
import warnings
import numpy as np
import multiprocessing as mp

from scipy.stats import boxcox
from scipy.stats import boxcox_normmax
from scipy.stats import gaussian_kde
from p_tqdm import p_map

from surprise import Surprise
from utils import standardise_data
from utils import load_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def IterativeEntropy(pri_data, post_data, iterations, mode='add'):
    """
    Algorithm used to iteratively compute the relative entropy based on
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
    Returns:
        prior (ndarray): Resulting data from iterative Box-Cox transformation
        posterior (ndarray): Resulting data from iterative Box-Cox transformation
        cach_rel_ent (list): Entropy computed after each Box-Cox transformation
        cach_exp_ent (list): Expected Entropy computed after each Box-Cox transformation
        cach_S (list): Surprise computed after each Box-Cox transformation
        cach_sD (list): Standard deviation computed after each Box-Cox transformation
        BoxCoxError (bool): True if Box-Cox failed
    """
    if not isinstance(mode, str) and mode in ["add", "replace"]:
        raise Exception(
            "Invalid kind value %s. Allowed format:"
            "'add' or 'replace'", mode)

    sc = Surprise()

    cach_rel_ent = []
    cach_exp_ent = []
    cach_S = []
    cach_sD = []

    BoxCoxError = False
    for steps in range(iterations):
        if steps == 0:
            prior, posterior, dim = standardise_data(pri_data.T,
                                                     post_data.T,
                                                     return_dim=True)
            prior = prior.T
            posterior = posterior.T

        # Transform to positive parameter values
        for k in range(dim):
            prior_a = np.amin(prior[k, :])
            post_a = np.amin(posterior[k, :])
            if prior_a < post_a and prior_a < 0:
                prior[k, :] -= prior_a
                posterior[k, :] -= prior_a
            elif post_a < 0:
                prior[k, :] -= post_a
                posterior[k, :] -= post_a
            prior_a = np.amin(prior[k, :])
            post_a = np.amin(posterior[k, :])
            while prior_a <= 0 or post_a <= 0:
                prior[k, :] += 5.0E-6
                posterior[k, :] += 5.0E-6
                prior_a = np.amin(prior[k, :])
                post_a = np.amin(posterior[k, :])

            # Find optimal one-parameter Box_Cox transformation
            try:
                lmbda = boxcox_normmax(prior[k, :],
                                       brack=(-1.9, 2.0),
                                       method='mle')
                box_cox_prior = boxcox(prior[k, :], lmbda=lmbda)
                prior[k, :] = box_cox_prior

                box_cox_post = boxcox(posterior[k, :], lmbda=lmbda)
                posterior[k, :] = box_cox_post
            except RuntimeWarning:
                logger.warning(
                    f"Box Cox transformation failed during step {steps}")
                BoxCoxError = True
                break

        if BoxCoxError:
            break
        prior_mu = np.mean(prior, axis=1)
        prior_std = np.std(prior, axis=1)

        # Standardize data
        prior = ((prior.T - prior_mu) / prior_std).T
        posterior = ((posterior.T - prior_mu) / prior_std).T

        if dim > 1:
            # Rotate data into eigenbasis of the covar matrix
            pri_cov = np.cov(prior)

            # Compute eigenvalues
            eVa, eVe = np.linalg.eig(pri_cov)

            # Compute transformation matrix from eigen decomposition
            R, S = eVe, np.diag(np.sqrt(eVa))
            T = np.matmul(R, S).T

            # Transform data with inverse transformation matrix T^-1
            try:
                inv_T = np.linalg.inv(T)
                prior = np.matmul(prior.T, inv_T).T
                posterior = np.matmul(posterior.T, inv_T).T
            except np.linalg.LinAlgError:
                logger.warning(
                    f"Singular matrix, inversion failed! Setting all output values for step {steps} to None"
                )
                cach_rel_ent.append(None)
                cach_exp_ent.append(None)
                cach_S.append(None)
                cach_sD.append(None)
                break
        # Compute D, <D>, S and sigma(D)
        try:
            rel_ent, exp_rel_ent, S, sD, p = sc(prior.T,
                                                posterior.T,
                                                mode=mode)
        except:
            logger.warning(
                f"Suprise() failed to compute the entropy values. Setting all output values for step {steps} to None"
            )
            rel_ent = None
            exp_rel_ent = None
            S = None
            sD = None

        cach_rel_ent.append(rel_ent)
        cach_exp_ent.append(exp_rel_ent)
        cach_S.append(S)
        cach_sD.append(sD)
        convergence_flag = 0
        """
         Very empirical convergence criterions. Idee is, that the true entropy value of the probe is either found after
         very vew transformations 1-3. First few transformations do not alter the computed entropy value by much, later
         on the transformations push the computed entropy away from the true value. Or the probe gets truly gaussianised
         by the transformation and the computed entropy value slowly converges to the true value i.e after 10+
         transformations.
        """
        if 6 > steps >= 1 and not None in cach_rel_ent[-2:]:
            if cach_rel_ent[-1] > cach_rel_ent[-2] and abs(
                    cach_rel_ent[-1] -
                    cach_rel_ent[-2]) / cach_rel_ent[-1] < 0.035:
                convergence_flag = 2
        elif steps == 2 and not None in cach_rel_ent[-2:]:
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-2]) / cach_rel_ent[-1] < 0.001:
                convergence_flag += 1
            if abs(cach_rel_ent[-2] -
                   cach_rel_ent[-3]) / cach_rel_ent[-1] < 0.001:
                convergence_flag += 1
        elif steps == 3 and not None in cach_rel_ent[-3:]:
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-2]) / cach_rel_ent[-1] < 0.002:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-3]) / cach_rel_ent[-1] < 0.008:
                convergence_flag += 1
        if steps > 3 and not None in cach_rel_ent[-4:]:
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-2]) / cach_rel_ent[-1] < 0.0002:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-3]) / cach_rel_ent[-1] < 0.0005:
                convergence_flag += 1
            if abs(cach_rel_ent[-1] -
                   cach_rel_ent[-4]) / cach_rel_ent[-1] < 0.0008:
                convergence_flag += 1
        if convergence_flag >= 2:
            logger.info(f"Convergence reached at step {steps}")
            break

    return prior, posterior, cach_rel_ent[-1], cach_exp_ent[-1], cach_S[
        -1], cach_sD[-1], BoxCoxError


def LoadAndComputeEntropy(prior,
                          post,
                          steps=300,
                          pri_burn=None,
                          post_burn=None,
                          params=[0, 1, 2, 3, 4],
                          mode='add'):
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
                        now can also be percentages (0,1)
        post_burn (int): Number of data points cut off at begining of post
                         sample data; default is 50%
                         now can also be percentages (0,1)
        params (list): List which indicates what varied parameters should be
                       used when computing the relative entropy
                       h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                       m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
                       Default is [0,1,2,3,4]
    Returns:
        results (dict): A dictionary containing following results
                        - "prior": Gaussianised prior (narray)
                        - "posterior": Gaussianised posterior (narray)
                        - "rel_ent": Relative entropy (float)
                        - "exp_ent": Expected Entropy (float)
                        - "S": Surprise (float)
                        - "sD": Standard deviation of expected entropy (float)
                        - "err": True if Box Cox failed (bool)
    """
    prior_data, post_data = load_data(prior,
                                      post,
                                      pri_burn,
                                      post_burn,
                                      params=params)

    results = {}

    logger.info("Starting IterativeEntropy")
    pri, post, rel_ent, exp_ent, S, sD, err = IterativeEntropy(prior_data.T,
                                                               post_data.T,
                                                               steps,
                                                               mode=mode)
    results["prior"] = pri
    results["posterior"] = post
    results["rel_ent"] = rel_ent
    results["exp_ent"] = exp_ent
    results["S"] = S
    results["sD"] = sD
    results["err"] = err

    del (prior_data)
    del (post_data)

    return results


def MonteCarloENTROPY(prior,
                      post,
                      steps,
                      error=False,
                      pri_burn=None,
                      post_burn=None,
                      params=[0, 1, 2, 3, 4]):
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
                        now can also be percentages (0,1)
        post_burn (int): Number of data points cut off at begining of post
                         sample data; default is 50%
                         now can also be percentages (0,1)
        params (list): List which indicates what varied parameters should be
                       used when computing the relative entropy
                       h = 0, Omega_m = 1, Omega_b = 2, N_s = 3, sigma_8 = 4
                       m_1 = 5, m_2 = 6, m_3 = 7, m_4 = 8
                       Default is [0,1,2,3,4]
        
    Returns:
        entropy (float): Approximation of the relative entropy
    """
    logger.info(f"Starting MonteCarloENTROPY with {steps} steps")
    cpu_count = mp.cpu_count()
    logger.info(f"Using {cpu_count} CPUs")

    # Assert that steps > cpu_count to prevent empty arrays in parallel compute
    if steps < cpu_count:
        cpu_count = steps

    pool = mp.Pool(cpu_count)
    warnings.simplefilter("error", RuntimeWarning)

    prior_data, post_data = load_data(prior,
                                      post,
                                      pri_burn,
                                      post_burn,
                                      params=params)
    #prior_data, posterior_data = standardise_data(prior_data, post_data)

    prior_kernel = gaussian_kde(prior_data.T)
    post_kernel = gaussian_kde(post_data.T)

    # Generate new sample set for MC evaluation
    sample_points = post_kernel.resample(size=steps).T

    # Parallel compute g_i and f_i
    prior_prob = p_map(
        prior_kernel.evaluate,
        [sample_points[i::cpu_count].T for i in range(cpu_count)],
        desc="Evaluating prior probability")
    post_prob = p_map(
        post_kernel.evaluate,
        [sample_points[i::cpu_count].T for i in range(cpu_count)],
        desc="Evaluating posterior probability")

    # Compute log(f_i/g_i), using log_2 for bit interpretation
    try:
        quotient = np.divide(post_prob, prior_prob)
    # Catch 'divide by zero' and adjust steps for invalid probes
    except RuntimeWarning:
        logger.warning(
            "RuntimeWarning: divide by zero encountered! Adjusting steps and filtering out invalid probes."
        )
        quotient = []
        count = 0
        for ii, prior_vec in enumerate(prior_prob):
            try:
                for jj, prior_val in enumerate(prior_vec):
                    if abs(prior_val) < 1e-30:
                        count += 1
                        continue
                    else:
                        post_val = post_prob[ii][jj]
                        quotient.append(post_val / prior_val)
            except TypeError:
                logger.debug("Iteration failed in second exception")
                prior_val = prior_vec
                if abs(prior_val) < 1e-30:
                    count += 1
                    continue
                else:
                    post_val = post_prob[ii][jj]
                    quotient.append(post_val / prior_val)
        if (steps - count) == 0:
            logger.critical(
                "Divide by zero encountered in all f_i/g_i! Can not estimate relative entropy"
            )
            quotient = [np.nan]
            steps = 1
        else:
            logger.info(
                f"Divide by zero encountered in {count}/{steps} of the f_i/g_i"
            )

    temp_res = []
    for value in quotient:
        temp_res.append(np.log2(value))
    tot_sum = 0
    for ii, value in enumerate(temp_res):
        try:
            tot_sum += sum(value)
        except TypeError:
            if not np.isnan(value):
                tot_sum += value
                logger.critical(
                    "MonteCarloENTROPY estimated with a low count number." +
                    f"Estimation vector {ii} of {cpu_count} only contained one estimation value"
                )
    entropy = tot_sum / steps
    pool.close()

    if error:
        error_estimate = np.var(temp_res) / steps
        return entropy, error_estimate
    else:
        return entropy
