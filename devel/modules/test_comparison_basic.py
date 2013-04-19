import sys
import logging
import galsim

# Make the galaxy and PSF objects elliptical Sersic and Moffat, storing all param vals here
# in top level scope

galn = 3.3
galhlr = 0.9

psfbeta = 3.40
psffwhm = 0.85

g1gal = -0.23
g2gal = -0.17
g1psf = +0.03
g2psf = +0.01

# Set a pixel scale (e.g. in arcsec), and image size
dx = 0.27
imsize = 48

# Random seed
rseed = 1234567

# Value of wmult parameter
wmult = 4.

def test_comparison_object():

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("test_comparison_object")

    logger.info("Running basic tests of comparison scripts using objects")

    # Build a trial galaxy
    gal = galsim.Sersic(galn, half_light_radius=galhlr)
    gal.applyShear(g1=g1gal, g2=g2gal)
    # And an example PSF
    psf = galsim.Moffat(beta=psfbeta, fwhm=psffwhm)
    psf.applyShear(g1=g1psf, g2=g2psf)

    # Try a single core run
    res1 = galsim.utilities.compare_dft_vs_photon_object(
        gal, psf_object=psf, rng=galsim.BaseDeviate(rseed), size=imsize, pixel_scale=dx,
        abs_tol_ellip=3.e-4, abs_tol_size=1.e-3, n_photons_per_trial=1e5)

    return

def test_comparison_config():

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("test_comparison_object")

    logger.info("Running basic tests of comparison scripts using config")

    # Set up a config dict to replicate the GSObject spec above
    config = {}

    config['gal'] = {
        "type" : "Sersic",
        "n" : galn,
        "half_light_radius" : galhlr,
        "ellip" : {
            "type" : "G1G2",
            "g1" : g1gal,
            "g2" : g2gal
        }
    }

    config['psf'] = {
        "type" : "Moffat",
        "beta" : psfbeta,
        "fwhm" : psffwhm, 
        "ellip" : {
            "type" : "G1G2",
            "g1" : g1psf,
            "g2" : g2psf
        }
    }

    config['image'] = {
        'size' : imsize,
        'pixel_scale' : dx,
        'random_seed' : rseed,
        #'wmult' : wmult # Note wmult not currently settable via config, but it ought to be I think
    }

    # Try a single core run not setting many kwargs
    #res1 = galsim.utilities.compare_dft_vs_photon_config(
        #config, abs_tol_ellip=3.e-4, abs_tol_size=1.e-3, nproc=1,
        #n_photons_per_trial=1e5, wmult=wmult, logger=logger)

    # Try a dual core run setting a few kwargs by hand to check correct overriding
    res2 = galsim.utilities.compare_dft_vs_photon_config(
        config, random_seed=rseed, size=imsize, pixel_scale=dx, abs_tol_ellip=3.e-4,
        abs_tol_size=1.e-3, n_photons_per_trial=1e5, wmult=wmult, nproc=2, logger=logger)

    return


if __name__ == "__main__":
    #test_comparison_object()
    test_comparison_config()
