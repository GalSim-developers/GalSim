# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

"""Script for testing the generation of Gaussian profiles by both DFT and photon shooting, for
comparison of the size and ellipticity in the resulting images.
"""

# Basic params of problem that will be consistent across tests:
PIXEL_SCALE = 0.03
IMAGE_SIZE = 512
# Number of objects from the COSMOS subsample of 300 to test
NOBS = 30#0

# Absolute tolerances on ellipticity and size estimates
TOL_ELLIP = 3.e-5
TOL_SIZE = 3.e-4 # Note this is in pixels by default, so for 0.03 arcsec/pixel this is still small

NPHOTONS = 3.e7

# Params for a very simple, Airy PSF
PSF_LAM_OVER_DIAM = 0.09 # ~ COSMOS width, oversampled at 0.03 arcsec

# MAX_FFT_SIZE (needed for high-n objects)
MAX_FFT_SIZE=65536

FAIL_VALUE=-666.

# Logging level
import logging
LOGLEVEL = logging.WARN

# Define a most basic config dictionary in this scope so that other modules can add to it
config_basic = {}
config_basic['image'] = {
    "size" : IMAGE_SIZE , "pixel_scale" : PIXEL_SCALE , # Note RANDOM_SEED generated later 
    "wmult" : 1., "n_photons" : NPHOTONS, "gsparams" : {"maximum_fft_size" : MAX_FFT_SIZE} }


# Then define a function that runs tests but adds extra gsparams if required, supplied as kwargs
def run_tests(random_seed, outfile, config=None, gsparams=None, wmult=None, logger=None,
              fail_value=-666.):
    """Run a full set of tests, writing pickled tuple output to outfile.
    """
    import cPickle
    import numpy as np
    import galsim
    import galaxy_sample
    
    if config is None:
        use_config = False
        if gsparams is None:
            import warnings
            warnings.warn("No gsparams provided to run_tests?")
        if wmult is None:
            raise ValueError("wmult must be set if config=None.")
    else:
        use_config = True
        if gsparams is not None:
            import warnings
            warnings.warn(
                "gsparams is provided as a kwarg but the config['image']['gsparams'] will take "+
                "precedence.")
        if wmult is not None:
            import warnings
            warnings.warn(
                "wmult is provided as a kwarg but the config['image']['wmult'] will take "+
                "precedence.")
    # Get galaxy sample
    n_cosmos, hlr_cosmos, gabs_cosmos = galaxy_sample.get()
    # Only take the first NOBS objects
    n_cosmos = n_cosmos[0: NOBS]
    hlr_cosmos = hlr_cosmos[0: NOBS]
    gabs_cosmos = gabs_cosmos[0: NOBS]
    # Setup a UniformDeviate
    ud = galsim.UniformDeviate(random_seed)
    # Open the output file and write a header:
    fout = open(outfile, 'wb')
    fout.write(
        '#  g1obs_draw g2obs_draw sigma_draw delta_g1obs delta_g2obs delta_sigma '+
        'err_g1obs err_g2obs err_sigma\n')
    # Start looping through the sample objects and collect the results
    for i, hlr, gabs in zip(range(NOBS), hlr_cosmos, gabs_cosmos):
        print "Testing galaxy #"+str(i+1)+"/"+str(NOBS)+\
              " with (hlr, |g|) = "+str(hlr)+", "+str(gabs)
        random_theta = 2. * np.pi * ud()
        g1 = gabs * np.cos(2. * random_theta)
        g2 = gabs * np.sin(2. * random_theta)
        if use_config:
            # Increment the random seed so that each test gets a unique one
            config['image']['random_seed'] = random_seed + i * NOBS + 1
            config['gal'] = {
                "type" : "Gaussian" , "half_light_radius" : hlr ,
                "ellip" : {
                    "type" : "G1G2" , "g1" : g1 , "g2" : g2
                }
            }
            config['psf'] = {"type" : "Airy" , "lam_over_diam" : PSF_LAM_OVER_DIAM }
            try:
                results = galsim.utilities.compare_dft_vs_photon_config(
                    config, abs_tol_ellip=TOL_ELLIP, abs_tol_size=TOL_SIZE, logger=logger)
                test_ran = True
            except RuntimeError as err:
                test_ran = False
                pass
            # Uncomment lines below to ouput a check image
            #import copy
            #checkimage = galsim.config.BuildImage(copy.deepcopy(config))[0] #im = first element
            #checkimage.write('junk_'+str(i + 1)+'_'+str(j + 1)+'.fits')
        else:
            test_gsparams = galsim.GSParams(maximum_fft_size=MAX_FFT_SIZE)
            galaxy = galsim.Gaussian(half_light_radius=hlr, gsparams=test_gsparams)
            galaxy.applyShear(g1=g1, g2=g2)
            psf = galsim.Airy(lam_over_diam=PSF_LAM_OVER_DIAM, gsparams=test_gsparams)
            try:
                results = galsim.utilities.compare_dft_vs_photon_object(
                    galaxy, psf_object=psf, rng=ud, pixel_scale=PIXEL_SCALE, size=IMAGE_SIZE,
                    abs_tol_ellip=TOL_ELLIP, abs_tol_size=TOL_SIZE,
                    n_photons_per_trial=NPHOTONS, wmult=wmult)
                test_ran = True
            except RuntimeError, err:
                test_ran = False
                pass

        if not test_ran:
            import warnings
            warnings.warn(
                'RuntimeError encountered for galaxy '+str(i + 1)+'/'+str(NOBS)+': '+str(err))
            fout.write(
                '%e %e %e %e %e %e %e %e %e %e %e %e\n' % (
                    fail_value, fail_value, fail_value, fail_value, fail_value, fail_value,
                    fail_value, fail_value, fail_value, fail_value, fail_value, fail_value,
                )
            )
            fout.flush()
        else:
            fout.write(
                '%e %e %e %e %e %e %e %e %e %e %e %e\n' % (
                    results.g1obs_draw, results.g2obs_draw, results.sigma_draw,
                    results.delta_g1obs, results.delta_g2obs, results.delta_sigma,
                    results.err_g1obs, results.err_g2obs, results.err_sigma, hlr, g1, g2
                )
            )
            fout.flush()
    fout.close()
    return


if __name__ == "__main__":
    import os

    # Use the basic config
    config = config_basic

    # Output filename
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    outfile = os.path.join("outputs", "gaussian_basic_output_N"+str(NOBS)+".asc")

    # Setup the logging
    logging.basicConfig(level=LOGLEVEL) 
    logger = logging.getLogger("gaussian_basic")

    random_seed = 912424534

    run_tests(random_seed, outfile, config=config, logger=logger, fail_value=FAIL_VALUE)
