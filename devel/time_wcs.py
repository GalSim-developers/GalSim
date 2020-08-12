import time
import timeit
import numpy as np
import galsim
import astropy.io.fits as fits

fns = ["sipsample.fits", "tpv.fits", "tanpv.fits"]

rng = np.random.default_rng()

size = 1_000_000

for fn in fns:
    header = fits.getheader(f"../tests/fits_files/{fn}")
    x = rng.uniform(0, header['NAXIS1'], size=size)
    y = rng.uniform(0, header['NAXIS2'], size=size)

    wcs = galsim.GSFitsWCS(header=header)
    ra, dec = wcs.xyToradec(x, y, units='rad')

    print()
    print(fn)
    print(f"PV?  {'PV1_1' in header}")
    print(f"SIP?  {'A_ORDER' in header}")

    t = min(timeit.repeat(lambda: wcs.xyToradec(x,y,units='rad'), number=3))
    print(f"xyToradec {t:.3f}")
    #t0 = time.time()
    #ra, dec = wcs.xyToradec(x, y, units='rad')
    #t1 = time.time()
    #print(f"xyToradec {t1-t0:.3f}")

    t = min(timeit.repeat(lambda: wcs.radecToxy(ra,dec,units='rad'), number=3))
    print(f"radecToxy {t:.3f}")
    #t0 = time.time()
    #x1, y1 = wcs.radecToxy(ra, dec, units='rad')
    #t1 = time.time()
    #print(f"radecToxy {t1-t0:.3f}")
