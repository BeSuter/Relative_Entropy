from __future__ import print_function, division, absolute_import
# This file is part of PyCosmo, a multipurpose cosmology calculation tool in Python.
#
# Copyright (C) 2013-2020 ETH Zurich, Institute for Particle and Astrophysics and SIS
# ID.
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.


# Copyright (C) 2018 Uwe Schmitt <uwe.schmitt@id.ethz.ch>

from types import FunctionType, MethodType

from ._structs import Struct


dimensionless = '[1]'
kelvin = '[K]'
not_determined = '??'

# list all required units here, also if requite unit is None:
units = dict(
        Cosmology=dict(h=dimensionless,
                       omega_b=dimensionless,
                       omega_m=dimensionless,
                       omega_l_in=dimensionless,
                       w0=dimensionless,
                       wa=dimensionless,
                       n=dimensionless,
                       tau=not_determined,
                       pk_norm_type=None,
                       pk_norm=None,
                       Tcmb=kelvin,
                       Yp=dimensionless,
                       Nnu=dimensionless,
                       F=not_determined,
                       fDM=not_determined,
                       ),
        Numerics=dict(pk_type=None,
                      pk_nonlin_type=None,
                      pk_nonlin_nu=None,
                      recomb=None,
                      recomb_dir=None,
                      omega_suppress=None,
                      suppress_rad=None,
                      cosmo_nudge=None,
                      tabulation=None,
                      ),
        BoltzmannSolver=dict(table_size=None,
                             l_max=None,
                             lna_0=None,
                             y_0=None,
                             initial_conditions=None,
                             lna_max=None,
                             econ_max=None,
                             econ_ratio=None,
                             dt_0=None,
                             halflife=None,
                             courant=None,
                             equations=None,
                             max_trace_changes=dimensionless,
                             sec_factor=dimensionless,
                             trace_changes_log_file=None,
                             traces_folder=None,
                             cache_folder=None,
                             ),
        MeadModel=dict(A_mead=None,
                       eta0_mead_equation=None,
                       eta0_mead_equation_version=None,
                       eta0_mead=None,
                       multiplicity_fnct=None,
                       npoints_k=dimensionless,
                       ),
        Observables=dict(a_size=None,
                         k_size=None,
                         ),
        LinearPerturbationApprox=dict(ainit_growth=None,
                                      rtol_growth=None,
                                      atol_growth=None,
                                      h0_growth=None,
                                      )
     )


_input_options = sorted(k for subd in units.values() for k in subd.keys())


def check_number_collection(expected_len):
    def checker(data):
        return (isinstance(data, (list, tuple))
                and len(data) == expected_len
                and all(isinstance(n, (int, float)) for n in data)
                )
    return checker


# if no restriction is listed: value is supposed to be float or int

restrictions = dict(omega_l_in=('flat', float),
                    pk_norm_type=('sigma8', 'deltah'),
                    pk_type=('EH', 'boltz', 'BBKS', 'BBKS_CCL'),
                    pk_nonlin_type=('halofit', 'rev_halofit', 'mead',),
                    recomb=('recfast++', 'cosmics', 'class', None),
                    recomb_dir=(str, None),
                    omega_suppress=bool,
                    suppress_rad=bool,
                    cosmo_nudge=check_number_collection(3),
                    tabulation=bool,
                    table_size=int,
                    l_max=int,
                    lna_0=(None, float),
                    y_0=(None, float),
                    initial_conditions=('cosmics', 'class', 'camb'),
                    lna_max=(None, float),
                    courant=check_number_collection(2),
                    equations=('newtonian_lna',),
                    max_trace_changes=int,
                    trace_changes_log_file=str,
                    traces_folder=str,
                    cache_folder=str,
                    baryons=('DMonly', 'REF', 'AGN', 'DBLIM'),
                    multiplicity_fnct=('PS', 'ST', 'Ti'),
                    a_size=int,
                    k_size=int,
                    npoints_k=int,
                    __file__=str,
                    rtol_growth=(None, float),
                    atol_growth=(None, float),
                    )


def check_consistencies(params):
    if params['recomb_dir'] is None and params['recomb'] is not None:
        raise InvalidConfigError(
            'recomb_dir = None is only allowed if recomb is also None')


class InvalidConfigError(Exception):
    pass


def parse_and_check_v2_config(config, file_name):
    try:
        return _parse_and_check_v2_config(config, file_name)
    except InvalidConfigError as e:
        raise InvalidConfigError("error when parsing config file {}: '{}'. Either your config "
                                 " file is invalid or you have to adapt {}"
                                 .format(file_name, e, __file__))


def _parse_and_check_v2_config(config, file_name):
    # global import instead would cause circular imports.
    from .config import Parameter
    sections = units.keys()
    missing = [section for section in sections if section not in config]
    if missing:
        raise InvalidConfigError(
            'section(s) {} missing'.format(', '.join(missing)))

    mismatches = []
    values = {
                '__file__': config.get('__file__')
              }
    for section in sections:
        for name, parameter in config[section].__dict__.items():
            if not isinstance(parameter, Parameter):
                continue
            values[name] = parameter.val
            if name not in units[section]:
                raise InvalidConfigError('unknown setting "{}"'.format(name))
            tobe = units[section][name]
            if parameter.unit != tobe:
                mismatches.append((name, parameter.unit, tobe))
    if mismatches:
        msg = ', '.join("{} has unit '{}', must be '{}'".format(name, is_, tobe)
                        for (name, is_, tobe) in mismatches)
        raise InvalidConfigError(msg)

    missing = set(_input_options) - set(values.keys())
    if missing:
        msg = "settings for {} are missing".format(', '.join(sorted(missing)))
        raise InvalidConfigError(msg)

    unknown = set(values.keys()) - set(_input_options)
    unknown.discard('__file__')
    if unknown:
        msg = "settings for {} are not supported".format(', '.join(sorted(unknown)))
        raise InvalidConfigError(msg)

    failed = []
    for name, value in values.items():
        if name not in restrictions:
            if not isinstance(value, (float, int)):
                failed.append("{} must be a number".format(name))
            continue

        tobe = restrictions[name]

        if isinstance(tobe, (FunctionType, MethodType)):
            ok = tobe(value)
            if not ok:
                failed.append(
                    '{} failed check for given value(s)'.format(name))
            continue

        if not isinstance(tobe, (list, tuple)):
            tobe = (tobe,)

        for ti in tobe:
            if ti is None and value is None:
                break
            elif ti in (str, float, int, bool):
                if isinstance(value, ti):
                    break
            elif isinstance(ti, str):
                if value == ti:
                    break
        else:
            failed.append('{} failed check, tobe in {}'.format(name, tobe))

    if failed:
        raise InvalidConfigError(', '.join(failed))

    check_consistencies(values)

    return Struct(**values)
