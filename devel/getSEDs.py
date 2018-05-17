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
"""@file getSEDs.py
Grab example SEDs from the web, clip them and then thin.
Note that the outputs of this script, which are the files GALSIM_DIR/share/SEDs/CWW_*.sed, are
already included in the repository.  This script just lets users know where these files came from
and how they were downloaded and altered.
"""
from __future__ import print_function
try:
    from urllib2 import urlopen
    from StringIO import StringIO
except:
    from urllib.request import urlopen
    from io import BytesIO as StringIO
import os
import galsim
import tarfile
import numpy as np

MAX_WAVE = 22050 # Angstroms

urlfile = 'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'
data = StringIO(urlopen(urlfile).read())
t = tarfile.open(fileobj=data, mode='r:gz')

sednames = ['./ZPHOT/templates/'+sedname for sedname in ['CWW_E_ext.sed',
                                                         'CWW_Im_ext.sed',
                                                         'CWW_Sbc_ext.sed',
                                                         'CWW_Scd_ext.sed']]

for sedname in sednames:
    file_ = t.extractfile(sedname)
    base = os.path.basename(sedname)
    x,f = np.loadtxt(file_, unpack=True)
    w = x <= MAX_WAVE
    x = x[w]
    f = f[w]
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5, fast_search=False)
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4, fast_search=False)
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3, fast_search=False)
    print("{0} raw size = {1}".format(base,len(x)))
    print("    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3)))

    # First write out the 1e-5 version.
    with open(base, 'w') as out:
        out.write(
"""#  {0} SED of Coleman, Wu, and Weedman (1980)
#  Extended below 1400 A and beyond 10000 A by
#  Bolzonella, Miralles, and Pello (2000) using evolutionary models
#  of Bruzual and Charlot (1993)
#
#  Obtained from ZPHOT code at
#  'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'
#
#  Truncated to wavelengths less than {1} Angstroms, and thinned by
#  galsim.utilities.thin_tabulated_values to a relative error of 1.e-5
#  with fast_search=False.  See devel/modules/getSEDs.py for details.
#
#  Angstroms     Flux/A
#
""".format(base.split('_')[1], MAX_WAVE))
        for i in range(len(x1)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x1[i], f1[i]))

    # Then write out the more thinned version.  Have to make a new filename.
    first_part, extension = base.split(".")
    out_base = first_part+"_more."+extension
    with open(out_base, 'w') as out:
        out.write(
"""#  {0} SED of Coleman, Wu, and Weedman (1980)
#  Extended below 1400 A and beyond 10000 A by
#  Bolzonella, Miralles, and Pello (2000) using evolutionary models
#  of Bruzual and Charlot (1993)
#
#  Obtained from ZPHOT code at
#  'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'
#
#  Truncated to wavelengths less than {1} Angstroms, and thinned by
#  galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
#  with fast_search=False.  See devel/modules/getSEDs.py for details.
#
#  Angstroms     Flux/A
#
""".format(base.split('_')[1], MAX_WAVE))
        for i in range(len(x3)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x3[i], f3[i]))
