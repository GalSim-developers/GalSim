import galsim

IMSIZE = 512
G1TEST = 0
G2TEST = -0.1
G1INT = -0.517999883751
G2INT = 0.2271233659
PIXTEST = 0.03

sersic = galsim.Sersic(n=1.3535728, half_light_radius=.4197004)
sersic.applyShear(g1=G1INT, g2=G2INT)
#cosmos_psf = galsim.Airy(lam_over_diam=0.07)
#pixel = galsim.Pixel(PIXTEST)
#convolved_sersic = galsim.Convolve([sersic, cosmos_psf, pixel])
sersic_sheared = sersic.createSheared(g1=G1TEST, g2=G2TEST)

image0 = galsim.ImageD(IMSIZE, IMSIZE)
image_sheared = galsim.ImageD(IMSIZE, IMSIZE)
#convolved_sersic.draw(image0, dx=0.03)
sersic.draw(image0, dx=PIXTEST)
sersic_sheared.draw(image_sheared, dx=PIXTEST)

interpolated_sersic = galsim.InterpolatedImage(
    image0, dx=PIXTEST, x_interpolant=galsim.Quintic(), k_interpolant=galsim.Quintic())
image_test = galsim.ImageD(IMSIZE, IMSIZE)
interpolated_sersic.applyShear(g1=G1TEST, g2=G2TEST)
interpolated_sersic = galsim.Convolve([interpolated_sersic, galsim.Gaussian(1.e-8)])
interpolated_sersic.draw(image_test, dx=PIXTEST)

results_sheared = galsim.hsm.FindAdaptiveMom(image_sheared)
results_test = galsim.hsm.FindAdaptiveMom(image_test)
results_original = galsim.hsm.FindAdaptiveMom(image0)

print G1INT, G2INT
print results_original.observed_shape.g1, results_original.observed_shape.g2
print galsim.Shear(g1=G1INT,g2=G2INT)+galsim.Shear(g1=G1TEST, g2=G2TEST)
print results_sheared.observed_shape.g1, results_sheared.observed_shape.g2
print results_test.observed_shape.g1, results_test.observed_shape.g2
#image_sheared.write('sersic-sheared.fits')
#image_test.write('interpolated-sheared.fits')
#(image_test-image_sheared).write('diff.fits')
