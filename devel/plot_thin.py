# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import galsim
import numpy as np
import matplotlib.pyplot as plt
import glob

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('thin.pdf')

for file_name in glob.glob('*.dat') + glob.glob('*.sed'):
    plt.clf()
    x,f = np.loadtxt(file_name, unpack=True)
    plt.plot(x,f, color='black', label='raw')
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4)
    plt.plot(x1,f1, color='blue', label='rel_err = 1.e-4')
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3)
    plt.plot(x2,f2, color='green', label='rel_err = 1.e-3')
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-2)
    plt.plot(x3,f3, color='red', label='rel_err = 1.e-2')
    plt.legend(loc='upper right')
    print "{0} raw size = {1}".format(file_name,len(x))
    print "    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3))

    pp.savefig()

pp.close()
