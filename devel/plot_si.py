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

import galsim
import scipy.special
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2, constrained_layout=True)

x = np.arange(0.01,100,0.01)

si_galsim = [galsim.bessel.si(xx) for xx in x]
si_scipy = scipy.special.sici(x)[0]

ax[0,0].plot(x, si_galsim, label="GalSim si")
ax[0,0].plot(x, si_scipy, label="SciPy sici")

ax[0,0].legend(loc='lower right')

ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('Si(x)')

ax[1,0].plot(x, si_galsim - si_scipy)
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('Si_galsim(x) - Si_scipy(x)')
ax[1,0].set_ylim(-1.e-15, 1.e-15)

ci_galsim = [galsim.bessel.ci(xx) for xx in x]
ci_scipy = scipy.special.sici(x)[1]

ax[0,1].plot(x, ci_galsim, label="GalSim ci")
ax[0,1].plot(x, ci_scipy, label="SciPy sici")

ax[0,1].legend(loc='lower right')

ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('Ci(x)')

ax[1,1].plot(x, ci_galsim - ci_scipy)
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('Ci_galsim(x) - Ci_scipy(x)')
ax[1,1].set_ylim(-1.e-15, 1.e-15)

fig.savefig('cisi.png')
