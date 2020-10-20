import numpy as np
import PyCosmo

w_bin = 100
l_min = 100
l_max = 1000
no_real = 252
ells = np.arange(1,3080)

param_array_dd_gg = (('custom', 'custom', 'deltag', 'deltag', None, 'nonlinear'), ('custom', 'custom', 'gamma', 'gamma', None, 'nonlinear'))

param_array_dd_dg_gg = (('custom', 'custom', 'deltag', 'deltag', None, 'nonlinear'), ('custom', 'custom', 'gamma', 'deltag', None, 'nonlinear'),\
    ('custom', 'custom', 'gamma', 'gamma', None, 'nonlinear'))

############# set up PyCosmo #############

path = '/cluster/home/besuter/DataSim'

#cosmo = PyCosmo.Cosmo('/cluster/home/rsgier/python_3.6_venv/lib/python3.6/site-packages/PyCosmo/config/default_v2_norecfast.py')
cosmo = PyCosmo.Cosmo()

nz_smail_txt = path + '/nz_smail'	# used redshift-distribution

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

cl_obs_dd_dg_gg = np.load(path + '/cl_dd_dg_gg_100_1000_w100_fullsky.npy')
cl_obs_dd_gg = np.load(path + '/cl_dd_gg_100_1000_w100_fullsky.npy')

#cov_sim_dd_dg_gg = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky_sim.npy')
cov_gaussian_dd_dg_gg = np.load(path + '/cov_dd_dg_gg_100_1000_w100_n252_fullsky.npy')

#cov_sim_dd_gg = np.load(path + '/cov_dd_gg_100_1000_w100_n252_fullsky_sim.npy')
cov_gaussian_dd_gg = np.load(path + '/cov_dd_gg_100_1000_w100_n252_fullsky.npy')

############# compute PyCosmo C_ells once at fiducial cosmology for normalization #############

obs = PyCosmo.Obs()

ells = np.arange(1,3080)

for (nz1, nz2, probe1, probe2, zrecomb, perturb) in param_array_dd_dg_gg:
   print('-----------------------')
   print('probes:', probe1, probe2)
   
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

############# compute PyCosmo for different parameter-values #############
############# here you can "loop" over parameter values #############
# draw h, Om, Ob, ns, s8 values for priors-ranges (for auto and auto+cross)

#p.add("h", (0.4, 1.2))
#p.add("Om", (0.1, 0.45))
#p.add("Ob", (0.01, 0.09))
#p.add("ns", (0.75, 1.2))
#p.add("s8", (0.7, 0.9))
combinations = [[0.3, 0.65, 0.04, 0.8, 0.8]]

for theta in combinations:
    
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
        print(n_bin)

        cl_pycosmo_binned = np.zeros(n_bin)
        for i in range(n_bin):
            cl_pycosmo_binned[i] = np.mean(cl_pycosmo[i * w_bin : i * w_bin + w_bin])
        
    print('cl_pycosmo_binned: ')
    print(cl_pycosmo_binned)
    
    # compute likelihood with pycosmo-vector (for this set of parameters) and imported data-vector, covariance matrix
    cl_obs = cl_obs_dd_dg_gg
    cov = cov_gaussian_dd_dg_gg
        
    l_obs = np.arange(len(cl_obs))
    
    assert cl_pycosmo_binned.shape == cl_obs.shape
    
    #log-likelihood (the data-vector and covariance are multiplied by a large no. to ensure stable invers of the covariance matrix
    # this factor drops out)
    cl_diff = cl_obs*1e4 - cl_pycosmo_binned*1e4
    #lnL     = -0.5*(cl_diff.dot(np.linalg.inv(cov*1e8))).dot(cl_diff)	# Gaussian likelihood
    
    lnL     = (-no_real / 2.)*np.log(1 + (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(no_real - 1.) )
    likelihood = np.exp(lnL)	#if you need the likelihood instead of log-likelihood
    print('Likelihood: ')
    print(likelihood)



def pycosmo_auto():

        """
        returns array, which has the same shape as cl_obs_auto
        """

        for (nz1, nz2, probe1, probe2, zrecomb, perturb) in param_array_dd_gg:
               
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
                    if probe1 == 'gamma' and probe2 == 'deltag':
                        cl_multi_pycosmo_arr = np.hstack((cl_multi_pycosmo_arr, cl_temp[l_min:l_max] * norm_gd))
                        if probe1 == 'gamma' and probe2 == 'gamma':
                            cl_multi_pycosmo_arr = np.hstack((cl_multi_pycosmo_arr, cl_temp[l_min:l_max] * norm_gg))
                    
        return cl_multi_pycosmo_arr

def pycosmo_cross():

        """
        returns array, which has the same shape as cl_obs_cross
        """

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
        
        return cl_multi_pycosmo_arr


