import sys
import logging
import galsim

def test_comparison_basic():

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger("test_comparison_basic")

    logger.info("Running basic tests of comparison scripts")

    # Set a pixel scale (in arcsec), and image size
    dx = 0.27
    imsize = 48

    # Build a trial galaxy
    gal = galsim.Sersic(3.3, half_light_radius=0.9)
    # And an example PSF
    psf = galsim.Moffat(beta=3.4, fwhm=0.85)

    # Try a single core run
    res1 = galsim.utilities.compare_object_dft_vs_photon(
        gal, psf_object=psf, imsize=imsize, dx=dx, abs_tol_ellip=1.e-4, abs_tol_size=3.e-4,
        n_photons_per_trial=1e6, random_seed=12345, ncores=1)

    # Then a multi core run
    res4 = galsim.utilities.compare_object_dft_vs_photon(
        gal, psf_object=psf, imsize=imsize, dx=dx, abs_tol_ellip=1.e-4, abs_tol_size=3.e-4,
        n_photons_per_trial=1e6, random_seed=12345, ncores=4)

    return

if __name__ == "__main__":
    test_comparison_basic()
