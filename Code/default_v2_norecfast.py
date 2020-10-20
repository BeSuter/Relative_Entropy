'''
Created 6 May 2017
by Adam Amara
'''
import collections
import sympy as sp
from PyCosmo.config import Parameter


# Cosmology Input Parameters
# = Parameter(val = , symbol = sp.symbols(''), txt = '', unit = '' )
#--------------------

class Cosmology:

    h = Parameter(val=0.7, symbol=sp.symbols('h'),
                  txt='dimensionless Hubble constant', unit='[1]')

    omega_b = Parameter(val=0.06, symbol=sp.symbols('omega_b'),
                        txt='Baryon density parameter', unit='[1]')

    omega_m = Parameter(val=0.3, symbol=sp.symbols('omega_m'),
                        txt='Matter density paramater (dark matter + baryons)', unit='[1]')

    omega_l_in = Parameter(val='flat', symbol=sp.symbols('omega_l'),
                           txt='Dark energy density. If flat then omega_l is 1.- omega_m - omega_r',
                           unit='[1]')

    w0 = Parameter(val=-1.0, symbol=sp.symbols('w_0'),
                   txt='DE equation of state at z=0', unit='[1]')

    wa = Parameter(val=0.0, symbol=sp.symbols('w_a'),
                   txt='DE equation of state evolution such that w(a)=w0+wa(1-a)', unit='[1]')

    n = Parameter(val=1.0, symbol=sp.symbols('n'),
                  txt='Spectral index for scalar modes', unit='[1]')

    tau = Parameter(val=0.09, symbol=sp.symbols('tau'),
                    txt='Optical depth [under development]', unit='??')

    pk_norm_type = Parameter(val='sigma8', symbol=None,
                             txt='''Power spectrum normalisation scheme: deltah for CMB normalisation
                                    or sigma8 for sigma8 normalisation''',
                             unit=None)

    pk_norm = Parameter(val=.8, symbol=None,
                        txt='''Power spectrum normalisation value:
                            either deltah or sigma8 depending on pk_norm_type setting''',
                        unit=None)

    Tcmb = Parameter(val=2.725, symbol=sp.symbols('T_{cmb}'),
                     txt='CMB temperature', unit='[K]')

    Yp = Parameter(val=0.24, symbol=sp.symbols('Yp'),
                   txt='Helium fraction [under development]', unit='[1]')

    Nnu = Parameter(val=3., symbol=sp.symbols('N_nu'),
                    txt='Number of effective massless neutrino species [under development]', unit='[1]')

    F = Parameter(val=1.14, symbol=sp.symbols('F'),
                  txt='?? [under development]', unit='??')

    fDM = Parameter(val=0.0, symbol=sp.symbols('fDM'),
                    txt='??? [under development]', unit='??')


class Numerics:

    pk_type = Parameter(val='EH', symbol=None,
                        txt='''sets is the linear perturbations should be calculated using boltzman
                               solver (boltz) or approximations (EH for Einstein and Hu or BBKS)''',
                        unit=None)

    pk_nonlin_type = Parameter(val='halofit', symbol=None,
                               txt='''sets if the nonlinear matter power spectrum should be calculated
                                    using the halofit fitting function (halofit) or the revised halofit
                                    fitting function (rev_halofit)''',
                               unit=None)

    recomb = Parameter(val=None, symbol=None,
                       txt='''code to compute recombination:recfast++ or cosmics or class [under
                            development; at present this is only used to build tables for Boltzmann
                            calculations]''',
                       unit=None)

    recomb_dir = Parameter(val="not set", symbol=None,
                           txt='COSMICS or CLASS directory for recombination [under development]',
                           unit=None)

    omega_suppress = Parameter(val=False, symbol=None,
                               txt='suppress radiation contribution in omega total as is often done',
                               unit=None)

    suppress_rad = Parameter(val=False, symbol=sp.symbols('suppress_rad'),
                             txt='suppress radiation contribution in omega total as is often done',
                             unit=None)

    cosmo_nudge = Parameter(val=[1., 1., 1.], symbol=None,
                            txt='''nudge factors for H0, omega_gam, and omega_neu to compare with other
                                codes - set to [1.,1.,1.] or leave out to suppress nudge''',
                            unit=None)

    tabulation = Parameter(val=False, symbol=None,
                           txt='''sets if cosmological observables should be computed using the
                                tabulated power spectra (True) or the non tabulated power spectra
                                (False)''',
                           unit=None)


class BoltzmannSolver:

    table_size = Parameter(val=2000, symbol=sp.symbols('table_size'),
                           txt='defines the interpolation size of the quatities cs**2, eta and taudot',
                           unit=None)

    l_max = Parameter(val=50, symbol=sp.symbols('l_{max}'),
                      txt='tbd: angular trancation limit',
                      unit=None)

    lna_0 = Parameter(val=None, symbol=sp.symbols('lna_0'),
                      txt='tbd, if not passed it automatically calculated by the initial conditions',
                      unit=None)

    y_0 = Parameter(val=None, symbol=sp.symbols('y_0'),
                    txt='tbd, if not passed it automatically calculated by the initial conditions ',
                    unit=None)

    initial_conditions = Parameter(val="class", symbol=sp.symbols('initial_c'),
                                   txt='''initial_conditions: (optional) initial conditons to use. Allowed
                                        values are: cosmics: tbd, class: tbd camb:tbd''',
                                   unit=None)

    lna_max = Parameter(val=None, symbol=sp.symbols('lna_{max}'),
                        txt='''t_end as log of redshift a (if none of them is passed, lnamax=0. is
                            assumed)''',
                        unit=None)

    econ_max = Parameter(val=3e-4, symbol=sp.symbols('econ_{max}'),
                         txt='''sets the maximum econ error for the boltzmann solver. If the econ
                                exeeds this limit, the timestep is reduced''',
                         unit=None)

    econ_ratio = Parameter(val=10, symbol=sp.symbols('econ_{ratio}'),
                           txt='''the ratio between the maximum econ and the minimum econ. The solver
                                increases the timestep if econ is smaller than the econ_max to
                                econ_ratio''',
                           unit=None)

    dt_0 = Parameter(val=1.5e-2, symbol=sp.symbols('dt_0'),
                     txt='the initial timestep size. np.sqrt(econ_max) is normally a good guess',
                     unit=None)

    halflife = Parameter(val=0.1, symbol=sp.symbols('halflife'),
                         txt='''the econ is calculated as a running mean:
                                recon = (recon * (halflife - dt) + dt * abs(econ)) / halflife''',
                         unit=None)

    courant = Parameter(val=[10., 1.], symbol=sp.symbols('courant'),
                        txt='scale parameter for the timescale [a * H / k,  eta * a * H]',
                        unit=None)

    equations = Parameter(val="newtonian_lna", symbol=sp.symbols('equations'),
                          txt='equationset to choose',
                          unit=None)

    #note_book = Parameter(val=None, symbol=None, unit=None,
    #                      txt="path to jupyter notebbok with mofied equations, None will use the "
    #                      "default notebook with boltzmann-einstein equations")

    max_trace_changes = Parameter(val=999, symbol=None, unit="[1]",
                                  txt='''limits the number of trace changes, reduces c code generation,
                                        might sacrifice stability, a value 1 is often sufficient.''')

    sec_factor = Parameter(val=10.0, symbol=None, unit="[1]",
                           txt='''always pivote if it swapped numbers differ by sec_factor or more''')

    trace_changes_log_file = Parameter(val="", unit=None, symbol=None,
                                       txt='''if this is a path and not an empty string the created
                                            solver will track trace changes to this file. Usually
                                            only used for internal uses''')

    traces_folder = Parameter(val="", unit=None, symbol=None,
                              txt='''use different folder for writing and reading traces. if not set
                                    default folder within PyCosmo is used''')

    cache_folder = Parameter(val="", unit=None, symbol=None,
                             txt='''use different folder for writing and reading c code and
                                    compiled solver. if not set default folder within PyCosmo is
                                    used''')


class MeadModel:

    baryons = Parameter(val='DMonly', symbol=sp.symbols('baryons'),
                        txt='''sets the parameters in Mead to account for the baryonic feedback - 
                            DMonly stays for no baryonic feedback, REF for SN feedback, AGN for REF plus
                            AGN feedback and DBLIM stays for REF plus corrections in stellar mass function
                            and SN energy''',
                        unit=None)

    multiplicity_fnct = Parameter(val='ST', symbol=sp.symbols('multiplicity_fnct'),
                                  txt='''Multiplicity function f(nu) as it appears in the calculation 
                            of the Halo Mass Function. The available fitting functions are Press&Schechter (PS), 
                            Sheth&Tormen (ST) and Tinker (Ti)''',
                                  unit=None)

    npoints_k = Parameter(val=5000, symbol=None, unit="[1]",
                          txt="k discretization when computing sigma")


class Observables:

    a_size = Parameter(val=600, symbol=sp.symbols(
        'a_size'), txt='a size', unit=None)

    k_size = Parameter(val=1000, symbol=sp.symbols(
        'k_size'), txt='k size', unit=None)


class LinearPerturbationApprox:
    ainit_growth=None
    rtol_growth=None
    atol_growth=None
    h0_growth=None
