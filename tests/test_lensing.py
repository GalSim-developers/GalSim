import numpy as np
import os

try:
    import galsim
except ImportError:
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..")))
    import galsim

import galsim.lensing

refdir = os.path.join(".", "lensing_reference_data") # Directory containing the reference

def funcname():
    import inspect
    return inspect.stack()[1][3]

def test_nfwhalo():
    # reference data comes from Matthias Bartelmann's libastro code
    # cluster properties: M=1e15, conc=4, redshift=1
    # sources at redshift=2
    # columns:
    # distance [arcsec], deflection [arcsec], shear, reduced shear, convergence
    # distance go from 1 .. 599 arcsec
    ref = np.loadtxt(refdir + '/nfw_lens.dat')

    import time
    t1 = time.time()
    # set up the same halo
    halo = galsim.lensing.NFWHalo(mass=1e15, conc=4, z=1, pos_x=0, pos_y=0)
    pos_x = np.arange(1,600)
    pos_y = np.zeros_like(pos_x)
    z_s = 2
    kappa = halo.getConvergence(pos_x, pos_y, z_s)
    gamma1, gamma2 = halo.getShear(pos_x, pos_y, z_s, reduced=False)
    g1, g2 = halo.getShear(pos_x, pos_y, z_s, reduced=True)

    # check internal correctness:
    # g1 = gamma1/(1-kappa), and g2 = 0
    np.testing.assert_array_equal(g1, gamma1/(1-kappa),  err_msg="Computation of reduced shear g incorrect.")
    np.testing.assert_array_equal(g2, np.zeros_like(g2),  err_msg="Computation of reduced shear g2 incorrect.")

    # comparison to reference:
    # tangential shear in x-direction is purely negative in g1
    np.testing.assert_allclose(-ref[:,2], gamma1,  rtol=1e-4, err_msg="Computation of shear deviates from reference.")
    np.testing.assert_allclose(-ref[:,3], g1,  rtol=1e-4, err_msg="Computation of reduced shear deviates from reference.")
    np.testing.assert_allclose(ref[:,4], kappa,  rtol=1e-4, err_msg="Computation of convergence deviates from reference.")
    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)
    
if __name__ == "__main__":
    test_nfwhalo()
