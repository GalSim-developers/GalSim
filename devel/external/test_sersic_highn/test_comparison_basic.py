import sys
import logging
import galsim

"""A simple Python test script to demonstrate use of the galsim.utilities.compare_dft_vs_photon_*
functions.

This script generates a model galaxy and PSF, and then compares the rendering of this object by both
photon shooting and DFT methods, by calling the GSObject `drawShoot()` and `draw()` methods
respectively.

There are two functions that do this in galsim.utilities:

    i)  galsim.utilities.compare_dft_vs_photon_object

    ii) galsim.utilities.compare_dft_vs_photon_config

i) allows the object and optional convolving PSF to be specified directly as GSObject instances.
However, as these are not picklable, these tests can only run in single core mode.

ii) provides multi-core processing functionality, but requires that the object and optional
convolving PSF are specified via a `config` dictionary (see, e.g., examples/demo8.py).

The two methods don't provide identical results, because the `object` version uses only one random
generator sequence to generate all the photons, whereas the `config` version uses a number of
differently seeded random number generators, one for each image.  One purpose of this script was
a quick sanity check of their overall consistency, as well as being a demonstration of these testing
utility functions.
"""


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
rseed = 111333555

# Value of wmult parameter
wmult = 4.

# Value of test tolerance parameters
tol_ellip = 3.e-5
tol_size = 1.e-4
n_photons_test= (int(1e6), int(3.e6), int(1.e7))

def test_comparison_object(np):

    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    logger = logging.getLogger("test_comparison_object")

    logger.info("Running basic tests of comparison scripts using objects")

    # Build a trial galaxy
    gal = galsim.Sersic(galn, half_light_radius=galhlr)
    gal.applyShear(g1=g1gal, g2=g2gal)
    # And an example PSF
    psf = galsim.Moffat(beta=psfbeta, fwhm=psffwhm)
    psf.applyShear(g1=g1psf, g2=g2psf)

    # Try a single core run
    print "Starting tests using config file with N_PHOTONS = "+str(np)
    res1 = galsim.utilities.compare_dft_vs_photon_object(
        gal, psf_object=psf, rng=galsim.BaseDeviate(rseed), size=imsize, pixel_scale=dx,
        abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np)
    print res1
    return

def test_comparison_config(np):

    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    logger = logging.getLogger("test_comparison_config")

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
        'wmult' : wmult
    }

    # Use an automatically-determined N core run setting
    print "Starting tests using config file with N_PHOTONS = "+str(np)
    res8 = galsim.utilities.compare_dft_vs_photon_config(
        config, n_photons_per_trial=np, nproc=-1, logger=logger, abs_tol_ellip=tol_ellip,
        abs_tol_size=tol_size)
    print res8
    return


if __name__ == "__main__":

    for n_photons in n_photons_test:
        # First run the config version, then the (slower, single core) object version: see docstring
        # in module header for more info.
        test_comparison_config(n_photons)
        test_comparison_object(n_photons)
