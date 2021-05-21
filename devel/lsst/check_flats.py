# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

from __future__ import print_function
import fitsio
import numpy as np


def cov(a, b):
    return np.mean( (a - np.mean(a)) * (b - np.mean(b)) )

var = cov01 = cov10 = cov11a = cov11b = cov02 = cov20 = cov03 = cov30 = cov0x = covx0 = n = 0

for i in range(10):
    flat0 = fitsio.read('flat{:02d}.fits'.format(i))
    print('mean ',i,' = ',flat0.mean())
    for j in range(i+1,10):
        flat1 = fitsio.read('flat{:02d}.fits'.format(j))

        diff = flat1 - flat0
        var += diff.var()
        cov01 += cov(diff[1:,:], diff[:-1,:])
        cov10 += cov(diff[:,1:], diff[:,:-1])
        cov11a += cov(diff[1:,1:], diff[:-1,:-1])
        cov11b += cov(diff[1:,:-1], diff[:-1,1:])
        cov02 += cov(diff[2:,:], diff[:-2,:])
        cov20 += cov(diff[:,2:], diff[:,:-2])
        cov03 += cov(diff[3:,:], diff[:-3,:])
        cov30 += cov(diff[:,3:], diff[:,:-3])
        cov0x += cov(diff[10:,:], diff[:-10,:])
        covx0 += cov(diff[:,10:], diff[:,:-10])
        n += 1

var /= n
cov01 /= n
cov10 /= n
cov11a /= n
cov11b /= n
cov02 /= n
cov20 /= n
cov03 /= n
cov30 /= n
cov0x /= n
covx0 /= n

print('var(diff)/2 = ',var/2)
print('cov01 = ',cov01)
print('cov10 = ',cov10)
print('cov11a = ',cov11a)
print('cov11b = ',cov11b)
print('cov02 = ',cov02)
print('cov20 = ',cov20)
print('cov03 = ',cov03)
print('cov30 = ',cov30)
print('cov0x = ',cov0x)
print('covx0 = ',covx0)
