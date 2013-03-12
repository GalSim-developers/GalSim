import galsim
import pylab
import numpy as np
import time



def scale_free_spectrum(k):
    if isinstance(k, np.ndarray):
        p = np.zeros(k.shape)
        w = (k!=0)
        p[w]=k[w]**-2
        return p
    else:
        if k!=0.:
            return k**-2
        else:
            return 0.


def show_shear(E,B,title):
    # Use deterministic seed.
    gd=galsim.GaussianDeviate(lseed=1234567)

    # Make a PowerSpectrum object with the selected E and B power spectra
    print "Initializing PowerSpectrum"
    my_ps = galsim.PowerSpectrum(E,B)
    # Get the shears and kappa from the PS
    print "Drawing shears and convergences on a grid"
    g1, g2, kappa = my_ps.buildGriddedShears(grid_spacing = 0.1, ngrid = 160,
                                             rng = gd, get_kappa=True)

    print "Some postprocessing as needed for plotting"
    # Need a sign flip!
    g2 = -g2

    g = (g1**2 + g2**2) ** 0.5
    theta = 0.5*np.arctan2(g2, g1)
    gx = g * np.cos(theta)
    gy = g * np.sin(theta)

    # Now start plotting things
    print "Plotting!"
    pylab.figure()
    pylab.pcolor(kappa)
    pylab.colorbar()
    pylab.quiver(gx, gy, scale=50, headwidth=0, pivot='middle')
    pylab.title(title)
    pylab.figure()
    ke, kb = galsim.lensing.invert_kappa_from_shear(g1, g2)
    pylab.pcolor(ke); pylab.colorbar()
    pylab.title(title+' kappa_E')
    pylab.quiver(gx, gy, scale=50, headwidth=0, pivot='middle')
    pylab.figure()
    pylab.pcolor(kb); pylab.colorbar()
    pylab.title(title+' kappa_B')

show_shear(scale_free_spectrum, None, "E-mode")
show_shear(None, scale_free_spectrum, "B-mode")


pylab.show()
