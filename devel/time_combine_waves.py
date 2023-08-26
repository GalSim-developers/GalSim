# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import timeit
import galsim
import numpy as np

from galsim.utilities import Profile

def old_combine_wave_list(*args):
    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a list or tuple")

    if len(args) == 0:
        return np.array([], dtype=float), 0.0, np.inf

    if len(args) == 1:
        obj = args[0]
        return obj.wave_list, getattr(obj, 'blue_limit', 0.0), getattr(obj, 'red_limit', np.inf)
    
    blue_limit = np.max([getattr(obj, 'blue_limit', 0.0) for obj in args])
    red_limit = np.min([getattr(obj, 'red_limit', np.inf) for obj in args])
    if blue_limit > red_limit:
        raise GalSimError("Empty wave_list intersection.")

    waves = [np.asarray(obj.wave_list) for obj in args]
    waves = [w[(blue_limit <= w) & (w <= red_limit)] for w in waves]
    wave_list = np.union1d(waves[0], waves[1])
    for w in waves[2:]:
        wave_list = np.union1d(wave_list, w)
    # Make sure both limits are included in final list
    if len(wave_list) > 0 and (wave_list[0] != blue_limit or wave_list[-1] != red_limit):
        wave_list = np.union1d([blue_limit, red_limit], wave_list)
    return wave_list, blue_limit, red_limit

# This edit was suggested by Jim Chiang to not merge things if they are all equal.
# (Slightly improved to use np.array_equal, rather than all(waves[0] == w).)
# It helps a lot when the inputs are equal, but not quite as much as the new C++ code.
def jims_combine_wave_list(*args):
    if len(args) == 1:
        if isinstance(args[0], (list, tuple)):
            args = args[0]
        else:
            raise TypeError("Single input argument must be a list or tuple")

    if len(args) == 0:
        return np.array([], dtype=float), 0.0, np.inf

    if len(args) == 1:
        obj = args[0]
        return obj.wave_list, getattr(obj, 'blue_limit', 0.0), getattr(obj, 'red_limit', np.inf)
    
    blue_limit = np.max([getattr(obj, 'blue_limit', 0.0) for obj in args])
    red_limit = np.min([getattr(obj, 'red_limit', np.inf) for obj in args])
    if blue_limit > red_limit:
        raise GalSimError("Empty wave_list intersection.")

    waves = [np.asarray(obj.wave_list) for obj in args]
    waves = [w[(blue_limit <= w) & (w <= red_limit)] for w in waves]
    if (len(waves[0]) == len(waves[1])
        and all(np.array_equal(waves[0], w) for w in waves[1:])):
        wave_list = waves[0]
    else:
        wave_list = np.union1d(waves[0], waves[1])
        for w in waves[2:]:
            wave_list = np.union1d(wave_list, w)
    # Make sure both limits are included in final list
    if len(wave_list) > 0 and (wave_list[0] != blue_limit or wave_list[-1] != red_limit):
        wave_list = np.union1d([blue_limit, red_limit], wave_list)
    return wave_list, blue_limit, red_limit


sed_list = [ galsim.SED(name, wave_type='ang', flux_type='flambda') for name in
             ['CWW_E_ext.sed', 'CWW_Im_ext.sed', 'CWW_Sbc_ext.sed', 'CWW_Scd_ext.sed'] ]

ref_wave, ref_bl, ref_rl = old_combine_wave_list(sed_list)
wave_list, blue_limit, red_limit = galsim.utilities.combine_wave_list(sed_list)
np.testing.assert_array_equal(wave_list, ref_wave)
assert blue_limit == ref_bl
assert red_limit == ref_rl

n = 10000
t1 = min(timeit.repeat(lambda: old_combine_wave_list(sed_list), number=n))
t2 = min(timeit.repeat(lambda: jims_combine_wave_list(sed_list), number=n))
t3 = min(timeit.repeat(lambda: galsim.utilities.combine_wave_list(sed_list), number=n))

print(f'Time for {n} iterations of combine_wave_list')
print('old time = ',t1)
print('jims time = ',t2)
print('new time = ',t3)

# Check when all wave_lists are equal.
sed_list = [ galsim.SED(name, wave_type='ang', flux_type='flambda') for name in
             ['CWW_E_ext.sed', 'CWW_E_ext.sed', 'CWW_E_ext.sed', 'CWW_E_ext.sed'] ]

ref_wave, ref_bl, ref_rl = old_combine_wave_list(sed_list)
jims_wave, jims_bl, jims_rl = jims_combine_wave_list(sed_list)
wave_list, blue_limit, red_limit = galsim.utilities.combine_wave_list(sed_list)

np.testing.assert_array_equal(wave_list, ref_wave)
assert blue_limit == ref_bl
assert red_limit == ref_rl

t1 = min(timeit.repeat(lambda: old_combine_wave_list(sed_list), number=n))
t2 = min(timeit.repeat(lambda: jims_combine_wave_list(sed_list), number=n))
t3 = min(timeit.repeat(lambda: galsim.utilities.combine_wave_list(sed_list), number=n))

print(f'Time for {n} iterations of combine_wave_list with identical wave_lists')
print('old time = ',t1)
print('jims time = ',t2)
print('new time = ',t3)
