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
    x1,f1 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-5)
    plt.plot(x1,f1, color='blue', label='rel_err = 1.e-5')
    x2,f2 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-4)
    plt.plot(x2,f2, color='green', label='rel_err = 1.e-4')
    x3,f3 = galsim.utilities.thin_tabulated_values(x,f,rel_err=1.e-3)
    plt.plot(x3,f3, color='red', label='rel_err = 1.e-3')
    plt.legend(loc='upper right')
    print "{0} raw size = {1}".format(file_name,len(x))
    print "    thinned sizes = {0}, {1}, {2}".format(len(x1),len(x2),len(x3))

    pp.savefig()

pp.close()
