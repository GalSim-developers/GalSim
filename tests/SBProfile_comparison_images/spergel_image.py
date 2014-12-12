import os
import numpy as np
from scipy.special import kv, gamma
from scipy.optimize import brentq
import astropy.io.fits as fits

def f(nu):
    return lambda u: (0.5*u)**nu * kv(nu, u) / gamma(nu + 1)

def HLR(nu):
    """Compute HLR for nu, r0=1."""
    return brentq(lambda r: (1.0+nu)*f(nu+1.0)(r) - 0.25, 0.01, 2.0)

def r0(nu):
    """Compute r0 for nu, HLR=1."""
    return 1./HLR(nu)

def radial_profile(nu, r0):
    cnu = HLR(nu)
    return lambda r: (cnu**2 / r0**2 * f(nu)(cnu * r/r0))

def rebin(a, shape):
    """Bin down image a to have final size given by shape."""
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

#now make some big images and rebin.
stamp_size = 31
dx = 0.2
xs = np.arange(-(stamp_size-1)/2, (stamp_size-1)/2+0.00001) * dx
ys = xs
xs, ys = np.meshgrid(xs, ys)
hlr = 1.0

hdulist = fits.HDUList()

for nu in [-0.9, 0.00, 0.85]:
    img = radial_profile(nu, hlr*r0(nu))(np.sqrt(xs**2 + ys**2))
    print nu, HLR(nu)
    output_file = "spergel_nu{:.2f}.fits".format(nu)

    hdulist = fits.HDUList()
    hdu = fits.PrimaryHDU(img)
    hdulist.append(hdu)
    if os.path.isfile(output_file):
        os.remove(output_file)
    hdulist.writeto(output_file)
