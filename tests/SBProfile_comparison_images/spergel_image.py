import os
import numpy as np
from scipy.special import kv, gamma
from scipy.optimize import brentq
import astropy.io.fits as fits

def f(nu):
    return lambda u: (0.5*u)**nu * kv(nu, u) / gamma(nu + 1)

def c(nu):
    """Compute c_nu"""
    return brentq(lambda r: (1.0+nu)*f(nu+1.0)(r) - 0.25, 0.01, 2.0)

def radial_profile(nu, r0):
    cnu = c(nu)
    return lambda r: (cnu**2 / r0**2 / (2.*np.pi) * f(nu)(cnu * r/r0))

#make some images
stamp_size = 31
dx = 0.2
xs = np.arange(-(stamp_size-1)/2, (stamp_size-1)/2+0.00001) * dx
ys = xs
xs, ys = np.meshgrid(xs, ys)
hlr = 1.0

hdulist = fits.HDUList()

for nu in [-0.85, -0.5, 0.0, 0.85]:
    img = radial_profile(nu, hlr)(np.sqrt(xs**2 + ys**2))
    # deal with the center:
    if nu > 0:
        img[(stamp_size-1)/2, (stamp_size-1)/2] = c(nu)**2 / hlr**2 / (2.*np.pi) / (2.*nu)
    else:
        img[(stamp_size-1)/2, (stamp_size-1)/2] = np.inf

    print nu, c(nu)
    output_file = "spergel_nu{:.2f}.fits".format(nu)

    hdulist = fits.HDUList()
    hdu = fits.PrimaryHDU(img)
    hdulist.append(hdu)
    if os.path.isfile(output_file):
        os.remove(output_file)
    hdulist.writeto(output_file)
