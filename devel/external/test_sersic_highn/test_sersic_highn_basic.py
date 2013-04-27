import cPickle
import logging
import numpy as np
import galsim
import galaxy_sample

PIXEL_SCALE = 0.03
IMAGE_SIZE = 96

RANDOM_SEED = 912424534

# Absolute tolerances on ellipticity and size estimates
TOL_ELLIP = 3.e-5
TOL_SIZE = 3.e-4 # Note this is in pixels by default, so for 0.03 arcsec/pixel this is still small

# Range of sersic n indices to check
SERSIC_N_TEST = [3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.]

WMULT = 1. # This might have an impact
NPHOTONS = 1.e7

# Output filename
OUTFILE = "sersic_highn_basic_output.pkl"

# Params for a very simple, Airy PSF
PSF_LAM_OVER_DIAM = 0.09 # ~ COSMOS width, oversampled at 0.03 arcsec

# If using config, settings
USE_CONFIG = True
if USE_CONFIG:
    config = {}
    config['image'] = {
        "size" : IMAGE_SIZE , "pixel_scale" : PIXEL_SCALE , # Note RANDOM_SEED generated later 
        "wmult" : WMULT, "n_photons" : NPHOTONS}

# Logging level
LOGLEVEL = logging.WARN

if __name__ == "__main__":

    import cPickle
    import logging
    import numpy as np
    import galsim
    import galaxy_sample

    logging.basicConfig(level=LOGLEVEL) 
    logger = logging.getLogger("sersic_highn_basic")
    # Get galaxy sample
    n_cosmos, hlr_cosmos, gabs_cosmos = galaxy_sample.get_galaxy_sample()
    nobs = len(hlr_cosmos[0:3]) # Number of objects
    ntest = len(SERSIC_N_TEST)
    g1obs_draw = np.empty((nobs, ntest)) # Arrays for storing results
    g2obs_draw = np.empty((nobs, ntest))
    sigma_draw = np.empty((nobs, ntest))
    delta_g1obs = np.empty((nobs, ntest))
    delta_g2obs = np.empty((nobs, ntest))
    delta_sigma = np.empty((nobs, ntest))
    err_g1obs = np.empty((nobs, ntest))
    err_g2obs = np.empty((nobs, ntest))
    err_sigma = np.empty((nobs, ntest))
    # Setup a UniformDeviate
    ud = galsim.UniformDeviate(RANDOM_SEED)
    # Start looping through the sample objects and collect the results
    for i, hlr, gabs in zip(range(nobs), hlr_cosmos[0:3], gabs_cosmos[0:3]):
        print "Testing galaxy #"+str(i+1)+"/"+str(nobs)+\
              " with (hlr, |g|) = "+str(hlr)+", "+str(gabs)
        random_theta = 2. * np.pi * ud()
        g1 = gabs * np.cos(2. * random_theta)
        g2 = gabs * np.sin(2. * random_theta)
        for j, sersic_n in zip(range(ntest), SERSIC_N_TEST):
            print "Exploring Sersic n = "+str(sersic_n)
            if USE_CONFIG:
                # Increment the random seed so that each test gets a unique one
                config['image']['random_seed'] = RANDOM_SEED + i * nobs * ntest + j * ntest + 1
                config['gal'] = {
                    "type" : "Sersic" , "n" : sersic_n , "half_light_radius" : hlr ,
                    "ellip" : {
                        "type" : "G1G2" , "g1" : g1 , "g2" : g2
                    }
                }
                config['psf'] = {"type" : "Airy" , "lam_over_diam" : PSF_LAM_OVER_DIAM }
                results = galsim.utilities.compare_dft_vs_photon_config(
                    config, abs_tol_ellip=TOL_ELLIP, abs_tol_size=TOL_SIZE, logger=logger)
            else:
                galaxy = galsim.Sersic(sersic_n, half_light_radius=hlr)
                galaxy.applyShear(g1=g1, g2=g2)
                psf = galsim.Airy(lam_over_diam=PSF_LAM_OVER_DIAM)
                results = galsim.utilities.compare_dft_vs_photon_object(
                    galaxy, psf_object=psf, rng=ud, pixel_scale=PIXEL_SCALE, size=IMAGE_SIZE,
                    abs_tol_ellip=TOL_ELLIP, abs_tol_size=TOL_SIZE, n_photons_per_trial=NPHOTONS,
                    wmult=WMULT)
            g1obs_draw[i, j] = results.g1obs_draw
            g2obs_draw[i, j] = results.g2obs_draw
            sigma_draw[i, j] = results.sigma_draw
            delta_g1obs[i, j] = results.delta_g1obs
            delta_g2obs[i, j] = results.delta_g2obs
            delta_sigma[i, j] = results.delta_sigma
            err_g1obs[i, j] = results.err_g1obs
            err_g2obs[i, j] = results.err_g2obs
            err_sigma[i, j] = results.err_sigma
    for_saving = (
        g1obs_draw, g2obs_draw, sigma_draw, delta_g1obs, delta_g2obs, delta_sigma, err_g1obs,
        err_g2obs, err_sigma)
    fout = open(OUTFILE, 'wb')
    cPickle.dump(for_saving, fout)
    fout.close()
