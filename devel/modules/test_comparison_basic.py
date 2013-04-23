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
rseed = 12345678

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
    psf_final = galsim.Convolve([psf, galsim.Pixel(dx)])

    # Try a single core run
    res1 = galsim.utilities.compare_dft_vs_photon_object(
        gal, psf_object=psf_final, rng=galsim.BaseDeviate(rseed), size=imsize, pixel_scale=dx,
        abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np)
    print "Object results with N_PHOTONS = "+str(np)
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

    config['pix'] = {
        "type" : "Pixel" ,
        "xw" : dx ,
    } 

    config['image'] = {
        'size' : imsize,
        'pixel_scale' : dx,
        'random_seed' : rseed,
        #'wmult' : wmult # Note wmult not currently settable via config, but it ought to be I think
    }

    import copy

    print "Config results with N_PHOTONS = "+str(np)
    # Try a single core run not setting many kwargs
    #res1 = galsim.utilities.compare_dft_vs_photon_config(
    #    copy.deepcopy(config), random_seed=rseed, size=imsize, pixel_scale=dx,
    #    abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np, wmult=wmult,
    #    nproc=1, logger=logger)
    #print res1

    # Try a dual core run setting
    #res2 = galsim.utilities.compare_dft_vs_photon_config(
    #    copy.deepcopy(config), random_seed=rseed, size=imsize, pixel_scale=dx,
    #    abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np, wmult=wmult,
    #    nproc=2, logger=logger)
    #print res2

    # Try a four core run setting
    #res4 = galsim.utilities.compare_dft_vs_photon_config(
    #    copy.deepcopy(config), random_seed=rseed, size=imsize, pixel_scale=dx,
    #    abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np,
    #    wmult=wmult, nproc=4, logger=logger)
    #print res4

    # Try an eight core run setting
    res8 = galsim.utilities.compare_dft_vs_photon_config(
        config, random_seed=rseed, size=imsize, pixel_scale=dx,
        abs_tol_ellip=tol_ellip, abs_tol_size=tol_size, n_photons_per_trial=np,
        wmult=wmult, nproc=8, logger=logger)

    print res8
    return


if __name__ == "__main__":

    for n_photons in n_photons_test:
        test_comparison_config(n_photons)
        test_comparison_object(n_photons)
