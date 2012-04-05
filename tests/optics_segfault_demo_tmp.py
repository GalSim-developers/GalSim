import numpy as np
import galsim

lod = 0.2   # lamda / D = .2, defining whatever physical scale I'm pretending to adopt for this test

# Then copy paste code from base.Optics()...
ALIAS_THRESHOLD = 0.005

maxk = 2. * np.pi / lod
dx = np.pi / maxk  # Nyquist sample rate
stepk = min(ALIAS_THRESHOLD * .5 * np.pi**3 / lod, np.pi / 5. / lod)

npix = np.ceil(2. * maxk / stepk).astype(int)
optimage = galsim.optics.psf_image(array_shape=(npix, npix), kmax=dx*maxk)
l5 = galsim.Lanczos(5, True, 1.e-4)
interpolant2d = galsim.InterpolantXY(l5)

sbpixel_test = galsim.SBPixel(optimage, interpolant2d, dx=dx)
image_test = sbpixel_test.draw()#.write('airy_sb_test.fits')


#OK, so image_test works, now uncomment this and see if a segfault/bus error happens

opt_test = galsim.Optics(lod, defocus=0.5, astig1=0.2, coma2=-0.1)
image_opt_test = opt_test.draw()




