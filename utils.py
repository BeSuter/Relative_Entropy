# Author: Benjamin Suter
# Created: May 2020

import numpy as np


def load_data(prior, posterior, pri_burn, post_burn, params=None):
    data_set = [prior, posterior]
    burn_set = [pri_burn, post_burn]
    return_set = [None, None]
    if not params:
        param_dim = len(data_set[0][0, :])
        params = np.arange(param_dim)

    for index, data in enumerate(data_set):
        if isinstance(data, str) and not burn_set[index]:
            tmp = np.load(data)
            burn = int(np.shape(tmp)[0] / 2)
            del (tmp)
            return_set[index] = np.load(data)[burn:, params]
        elif isinstance(data, str) and burn_set[index] < 1:
            tmp = np.load(data)
            burn = int(np.shape(tmp)[0] * burn_set[index])
            del (tmp)
            return_set[index] = np.load(prior)[burn:, params]
        elif isinstance(data, str):
            return_set[index] = np.load(data)[burn_set[index]:, params]
        elif isinstance(data, np.ndarray):
            return_set[index] = data
        else:
            raise Exception(
                "Invalide type for prior! Use 'str' or 'np.ndarray'")

    return return_set[0], return_set[1]


def standardise_data(prior_data, post_data, return_dim=False):
    axis = 1
    try:
        dim = len(prior_data[0, :])
    except IndexError:
        dim = 1
        axis = 0

    # Compute mean and standard deviation of the prior
    prior_mu = np.mean(prior_data.T, axis=axis)
    prior_std = np.std(prior_data.T, axis=axis)

    # Standardize data
    prior = ((prior_data - prior_mu) / prior_std).T
    posterior = ((post_data - prior_mu) / prior_std).T

    if dim > 1:
        # Rotate data into eigenbasis of the covar matrix
        pri_cov = np.cov(prior)

        # Compute eigenvalues
        eVa, eVe = np.linalg.eig(pri_cov)

        # Compute transformation matrix from eigen decomposition
        R, S = eVe, np.diag(np.sqrt(eVa))
        T = np.matmul(R, S).T

        # Transform data with inverse transformation matrix T^-1
        inv_T = np.linalg.inv(T)

        prior = np.matmul(prior.T, inv_T)
        post = np.matmul(posterior.T, inv_T)
    if return_dim:
        return prior, post, dim
    else:
        return prior, post
