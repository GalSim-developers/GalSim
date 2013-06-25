try:
    import galsim
except ImportError:
    import os
    import sys
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path, "..", "..", "..")))
    import galsim

sersic = galsim.Sersic(n=3.1, half_light_radius=0.6)

im = galsim.ImageD(512, 512)

sersic.draw(im, dx=0.03)

interpolated = galsim.InterpolatedImage(im, dx=0.03, k_interpolant=galsim.Lanczos(3))
interpolated_convolved = galsim.Convolve([interpolated, galsim.Gaussian(1.e-8)])

outimage = galsim.ImageD(512, 512)
for i in range(7):
    interpolated_convolved.draw(outimage, dx=0.03)
