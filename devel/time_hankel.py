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


# Compare our Hankel transforms to the hankl package
# https://hankl.readthedocs.io/en/latest/examples.html

import galsim
import hankl
import numpy as np
import matplotlib.pyplot as plt
import time

def f(r, mu=0.0):
    return r**(mu+1.0) * np.exp(-r**2.0 / 2.0)

def g(k, mu=0.0):
    return k**(mu+1.0) * np.exp(- k**2.0 / 2.0)

def f1(r, mu=0.0):
    return r**(mu) * np.exp(-r**2.0 / 2.0)

r = np.logspace(-5, 5, 2**10)

t0 = time.time()
k, G = hankl.FFTLog(r, f(r, mu=0.0), q=0.0, mu=0.0)
t1 = time.time()
G2 = k * galsim.integ.hankel(f1, k, nu=0)
t2 = time.time()
print('Times: ',t1-t0, t2-t1)

plt.figure(figsize=(10,6))

ax1 = plt.subplot(121)
plt.loglog(r, f(r))
plt.title('$f(r) = r \; exp(-r^{2}/2)$')
plt.xlabel('$r$')
plt.ylim(10**(-6), 1)
plt.xlim(10**(-5), 10)

ax1.yaxis.tick_left()
ax1.yaxis.set_label_position("left")

ax2 = plt.subplot(122, sharey=ax1)
plt.loglog(k, g(k), label='Analytical')
plt.loglog(k, G, ls='--', label='hankl - FFTLog')
plt.loglog(k, G2, ls='--', label='galsim.integ.hankel')
plt.title('$g(k) = k \; exp(-k^{2}/2)$')
plt.xlabel('$k$')
plt.ylim(10**(-6), 1)
plt.xlim(10**(-5), 10)
plt.legend()

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.tight_layout()

#plt.show()
plt.savefig('hankl.png')
