import galsim
import time

nlist = [0.3 + 0.1*k for k in range(60)]

im = galsim.Image(32,32, scale=0.28)
psf = galsim.Moffat(fwhm = 0.9, beta = 3)

for iter in range(2):
    tstart = time.time()
    for n in nlist:
        t0 = time.time()
        gal = galsim.Sersic(half_light_radius = 1.4, n=n)
        final = galsim.Convolve(psf,gal)
        im = final.drawImage(image=im)
        t1 = time.time()
        print 'n = %f, time = %f'%(n,t1-t0)
    tend = time.time()
    print 'Total time = %f'%(tend-tstart)

gsparams = galsim.GSParams(xvalue_accuracy=1.e-2, kvalue_accuracy=1.e-2,
                           maxk_threshold=1.e-2, alias_threshold=1.e-2)

for iter in range(2):
    tstart = time.time()
    for n in nlist:
        t0 = time.time()
        gal = galsim.Sersic(half_light_radius = 1.4, n=n, gsparams=gsparams)
        final = galsim.Convolve(psf,gal)
        im = final.drawImage(image=im)
        t1 = time.time()
        print 'n = %f, time = %f'%(n,t1-t0)
    tend = time.time()
    print 'Total time with loose params = %f'%(tend-tstart)
