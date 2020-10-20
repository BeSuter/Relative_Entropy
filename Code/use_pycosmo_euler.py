import numpy as np
import PyCosmo

w_bin = 100
l_min = 100
l_max = 1000
ells = np.arange(1,3080)

############# set up PyCosmo #############

path = SET PATH WHERE FILES ARE LOCATED

cosmo = PyCosmo.Cosmo('/cluster/home/rsgier/python_3.6_venv/lib/python3.6/site-packages/PyCosmo/config/default_v2_norecfast.py')

nz_smail_txt = path + 'nz_smail'	# used redshift-distribution

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

# gamma-gamma auto-correlations only

cl_obs_gg = np.load(path + '/cl_gg_100_1000_w100_fullsky.npy')
cov_sim_gg= np.load(path + '/cov_gg_100_1000_w100_n252_fullsky_sim.npy')
cov_gaussian_gg = np.load(path + '/cov_gg_100_1000_w100_n252_fullsky.npy')

############# compute PyCosmo C_ells once at fiducial cosmology for normalization #############

obs = PyCosmo.Obs()

nz_smail_txt = path + 'nz_smail'

clparams_g_g = {'nz': ['custom', 'custom'],
         'path2zdist': [nz_smail_txt, nz_smail_txt],
         'probes': ['gamma', 'gamma'],
         'perturb': 'nonlinear',
         'normalised': False,
         'bias': 1.,
         'm': 0,
         'ngauss': 700,
         'z_grid': np.array([0.001, 1.75, 1000])
         }

ells = np.arange(1,3080)                              
cl_pycosmo_norm = obs.cl(ells, cosmo, clparams_g_g)[l_min : l_max]
norm = 0.2 / np.mean(cl_pycosmo_norm)

############# compute PyCosmo for different parameter-values #############
############# here you can "loop" over parameter values #############
# draw h, Om, Ob, ns, s8 values for priors-ranges
# p.add("Om", (0.2, 0.6))
# p.add("s8", (0.6, 1.0))
# p.add("h", (0.2, 1.2))
# p.add("ns", (0.5, 1.4))
# p.add("Ob", (0.01, 0.09))

cosmo.set(h=h)
cosmo.set(omega_m=Om)
cosmo.set(omega_b = Ob)
cosmo.set(n=ns)
cosmo.set(pk_norm=s8)

obs = PyCosmo.Obs()

nz_smail_txt = path + 'nz_smail'

clparams_g_g = {'nz': ['custom', 'custom'],
        'path2zdist': [nz_smail_txt, nz_smail_txt],
        'probes': ['gamma', 'gamma'],
        'perturb': 'nonlinear',
        'normalised': False,
        'bias': 1.,
        'm': 0,
        'ngauss': 700,
        'z_grid': np.array([0.001, 1.75, 1000])
        }

ells = np.arange(1,3080)                              
cl_pycosmo = obs.cl(ells, cosmo, clparams_g_g)[l_min : l_max]
cl_pycosmo *= norm


# bin pycosmo-vector (this is needed since the data-vector and covariance matrix are binned in this way)

n_bin = int(len(cl_pycosmo)/w_bin)
print(n_bin)

cl_pycosmo_binned = np.zeros(n_bin)
for i in range(n_bin):
 cl_pycosmo_binned[i] = np.mean(cl_pycosmo[i * w_bin : i * w_bin + w_bin])

# compute likelihood with pycosmo-vector (for this set of parameters) and imported data-vector, covariance matrix

l_obs = np.arange(len(cl_obs))

assert cl_pycosmo_binned.shape == cl_obs.shape

#log-likelihood (the data-vector and covariance are multiplied by a large no. to ensure stable invers of the covariance matrix
# this factor drops out)
cl_diff = cl_obs*1e4 - cl_pycosmo_binned*1e4
#lnL     = -0.5*(cl_diff.dot(np.linalg.inv(cov*1e8))).dot(cl_diff)	# Gaussian likelihood
lnL     = (-no_real / 2.)*np.log(1 + (cl_diff.dot(np.linalg.inv(cov * 1e8))).dot(cl_diff)/(no_real - 1.) )

#likelihood = np.exp(lnL)	#if you need the likelihood instead of log-likelihood





