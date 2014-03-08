# Grab example SEDs from the web, clip them at 12000 Angstroms, and then thin with rel_err = 1.e-5

import os
import galsim
import urllib2
from StringIO import StringIO
import tarfile

import numpy as np

urlfile = 'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'
data = StringIO(urllib2.urlopen(urlfile).read())
t = tarfile.open(fileobj=data, mode='r:gz')

sednames = ['./ZPHOT/templates/'+sedname for sedname in ['CWW_E_ext.sed',
                                                         'CWW_Im_ext.sed',
                                                         'CWW_Sbc_ext.sed',
                                                         'CWW_Scd_ext.sed']]

for sedname in sednames:
    file_ = t.extractfile(sedname)
    base = os.path.basename(sedname)
    x,f = np.loadtxt(file_, unpack=True)
    w = x<12000 # Angstroms
    x=x[w]
    f=f[w]
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5)
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4)
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3)
    print "{0} raw size = {1}".format(base,len(x))
    print "    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3))

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
#  Truncated to wavelengths less than 12000 Angstroms, and thinned by
#  galsim.utilities.thin_tabulated_values to a relative error of 1.e-5
#
#  Angstroms     Flux/A
#
""".format(base.split('_')[1]))
        for i in range(len(x1)):
            out.write(" {0:>10.2f}    {1:>10.5f}\n".format(x1[i], f1[i]))
