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
"""@file comparison_utilities.py
Module containing general utilities for comparing rendered image output from the GalSim software.
"""

import numpy as np
import galsim

class ComparisonShapeData(object):
    """A class to contain the outputs of a comparison between photon shooting and DFT rendering of
    GSObjects, as measured by the HSM module's FindAdaptiveMom or (in future) EstimateShear.

    Currently this class contains the following attributes (see also the ShapeData
    documentation for a more detailed description of, e.g., `observed_shape`, `moments_sigma`)
    describing the results of the comparison:

    - g1obs_draw: `observed_shape.g1` from adaptive moments on a GSObject image rendered using
      .draw()

    - g2obs_draw: `observed_shape.g2` from adaptive moments on a GSObject image rendered using
      .draw()

    - g1hsm_draw: `corrected_shape.g1` from adaptive moments on a GSObject image rendered using
      .draw()

    - g2hsm_draw: `corrected_shape.g2` from adaptive moments on a GSObject image rendered using
      .draw()

    - sigma_draw: `moments_sigma` from adaptive moments on a GSObject image rendered using .draw()

    - sighs_draw: `moments_sigma` from HSM PSF correction on a GSObject image rendered using .draw()

    - delta_g1obs: estimated mean difference between i) `observed_shape.g1` from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1obs_draw`.
      Defined `delta_g1obs = g1obs_draw - g1obs_shoot`.

    - delta_g2obs: estimated mean difference between i) `observed_shape.g2` from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1obs_draw`.
      Defined `delta_g2obs = g2obs_draw - g2obs_shoot`.

    - delta_g1hsm: estimated mean difference between i) `observed_shape.g1` from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1hsm_draw`.
      Defined `delta_g1hsm = g1hsm_draw - g1hsm_shoot`.

    - delta_g2hsm: estimated mean difference between i) `observed_shape.g2` from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1hsm_draw`.
      Defined `delta_g2hsm = g2hsm_draw - g2hsm_shoot`.

    - delta_sigma: estimated mean difference between i) `moments_sigma` from images of the same
      GSObject rendered with .drawShoot(), and ii) `sigma_draw`.
      Defined `delta_sigma = sigma_draw - sigma_shoot`.

    - delta_sighs: estimated mean difference between i) `moments_sigma` from images of the same
      GSObject rendered with .drawShoot(), and ii) `sigma_draw`. 
      Defined `delta_sigma = sigma_draw - sigma_shoot`. Moments calculated using HSM PSF corr.

    - err_g1obs: standard error in `delta_g1obs` estimated from the test sample.

    - err_g2obs: standard error in `delta_g2obs` estimated from the test sample.

    - err_g1hsm: standard error in `delta_g1hsm` estimated from the test sample.

    - err_g2hsm: standard error in `delta_g2hsm` estimated from the test sample.

    - err_sigma: standard error in `delta_sigma` estimated from the test sample.

    - err_sighs: standard error in `delta_sigma` estimated from the test sample using HSM PSF corr.

    The ComparisonShapeData instance also stores much of the meta-information about the tests:

    - gsobject: the galsim.GSObject for which this test was performed (prior to PSF convolution if
      a PSF was also supplied).

    - psf_object: the optional additional PSF supplied by the user for tests of convolved objects,
      will be `None` if not used.

    - size: the size of the images tested - all test images are currently square.

    - pixel_scale: the pixel scale in the images tested.

    - wmult: the `wmult` parameter used in .draw() (see the GSObject .draw() method docs for more
      details).

    - n_iterations: number of iterations of `n_trials` trials required to get delta quantities to
      the above accuracy.

    - n_trials_per_iter: number of trial images used to estimate or successively re-estimate the
      standard error on the delta quantities above for each iteration.

    - n_photons_per_trial: number of photons shot in drawShoot() for each trial.

    - time: the time taken to perform the test.

    Note this is really only a simple storage container for the results above.  All of the
    non trivial calculation is completed before a ComparisonShapeData instance is initialized,
    typically in the function compare_object_dft_vs_photon().

    - gsobject: optional GSObject for which this test was performed (prior to PSF convolution
      if a PSF was also supplied).

    - psf_object: the optional additional PSF supplied by the user for tests of convolved objects,
      will be `None` if not used.

    - config: optional config object describing the GSObject and PSF if the config comparison script
      was used rather than the (single core only) direct object script.

    Either `config`, or `gsobject`, or `gsobject` and `psf_object`, must be set when a
    ComparisonShapeData instance is created, or an Exception is raised.
    """
    def __init__(self, g1obs_draw, g2obs_draw, g1hsm_draw, g2hsm_draw, sigma_draw, sighs_draw , 
                g1obs_shoot, g2obs_shoot, g1hsm_shoot, g2hsm_shoot, sigma_shoot, sighs_shoot ,
                err_g1obs, err_g2obs, err_g1hsm, err_g2hsm, err_sigma, err_sighs ,size, pixel_scale, 
                wmult, n_iterations, n_trials_per_iter, n_photons_per_trial, time, 
                gsobject=None, psf_object=None, config=None):
       """In general use you should not need to instantiate a ComparisonShapeData instance,
       as this is done within the compare_dft_vs_photon_config() or other such functions. 
       """

       self.g1hsm_draw = g1hsm_draw
       self.g2hsm_draw = g2hsm_draw
       self.g1obs_draw = g1obs_draw
       self.g2obs_draw = g2obs_draw
       self.sigma_draw = sigma_draw
       self.sighs_draw = sighs_draw
       
       self.delta_g1hsm = g1hsm_draw - g1hsm_shoot
       self.delta_g2hsm = g2hsm_draw - g2hsm_shoot
       self.delta_g1obs = g1obs_draw - g1obs_shoot
       self.delta_g2obs = g2obs_draw - g2obs_shoot
       self.delta_sigma = sigma_draw - sigma_shoot
       self.delta_sighs = sighs_draw - sighs_shoot

       self.err_g1hsm = err_g1hsm
       self.err_g2hsm = err_g2hsm
       self.err_g1obs = err_g1obs
       self.err_g2obs = err_g2obs
       self.err_sigma = err_sigma
       self.err_sighs = err_sighs

       if gsobject is not None:
           if config is not None:
               raise ValueError("Specifying both a config and gsobject input kwarg is ambiguous")
       elif config is None:
           raise ValueError(
               "Either config, or gsobject (with an optional psf_object) must be given as input.")
       self.config = config
       self.gsobject = gsobject
       self.psf_object = psf_object

       self.size = size
       self.pixel_scale = pixel_scale
       self.wmult = wmult
       self.n_iterations = n_iterations
       self.n_trials_per_iter = n_trials_per_iter
       self.n_photons_per_trial = n_photons_per_trial
       self.time = time

    def __str__(self):
        retval = "g1obs_draw  = "+str(self.g1obs_draw)+"\n"+\
                 "delta_g1obs = "+str(self.delta_g1obs)+" +/- "+str(self.err_g1obs)+"\n"+\
                 "\n"+\
                 "g2obs_draw  = "+str(self.g2obs_draw)+"\n"+\
                 "delta_g2obs = "+str(self.delta_g2obs)+" +/- "+str(self.err_g2obs)+"\n"+\
                 "\n"+\
                 "g1hsm_draw  = "+str(self.g1hsm_draw)+"\n"+\
                 "delta_g1hsm = "+str(self.delta_g1hsm)+" +/- "+str(self.err_g1hsm)+"\n"+\
                 "\n"+\
                 "g2hsm_draw  = "+str(self.g2hsm_draw)+"\n"+\
                 "delta_g2hsm = "+str(self.delta_g2hsm)+" +/- "+str(self.err_g2hsm)+"\n"+\
                 "\n"+\
                 "sigma_draw  = "+str(self.sigma_draw)+"\n"+\
                 "delta_sigma = "+str(self.delta_sigma)+" +/- "+str(self.err_sigma)+"\n"+\
                 "\n"+\
                 "sigma_draw_hsm  = "+str(self.sighs_draw)+"\n"+\
                 "delta_sigma_hsm = "+str(self.delta_sighs)+" +/- "+str(self.err_sighs)+"\n"+\
                 "\n"+\
                 "image size = "+str(self.size)+"\n"+\
                 "pixel scale = "+str(self.pixel_scale)+"\n"+\
                 "wmult = "+str(self.wmult)+"\n"+\
                 "\n"+\
                 "total time taken = "+str(self.time)+" s\n"+\
                 "total number of iterations = "+str(self.n_iterations)+"\n"+\
                 "number of trials per iteration = "+str(self.n_trials_per_iter)+"\n"+\
                 "number of photons per trial = "+str(self.n_photons_per_trial)+"\n"
        return retval

    # Reuse the __str__ method for __repr__
    __repr__ = __str__


def compare_dft_vs_photon_object(gsobject, psf_object=None, rng=None, pixel_scale=1., size=512,
                                 wmult=4., abs_tol_ellip=1.e-5, abs_tol_size=1.e-5,
                                 n_trials_per_iter=32, n_photons_per_trial=1e7, moments=True,
                                 hsm=False):
    """Take an input object (with optional PSF) and render it in two ways comparing results at high
    precision.

    Using both photon shooting (via drawShoot()) and Discrete Fourier Transform (via draw()) to
    render images, we compare the numerical values of adaptive moments estimates of size and
    ellipticity to check consistency.

    This function takes actual GSObjects as its input, but because these are not yet picklable this
    means that the internals cannot be parallelized using the Python multiprocessing module.  For
    a parallelized function, that instead uses a config dictionary to specify the test objects, see
    the function compare_dft_vs_photon_config() in this module.

    We generate successive sets of `n_trials_per_iter` photon-shot images, using 
    `n_photons_per_trial` photons in each image, until the standard error on the mean absolute size
    and ellipticity drop below `abs_tol_size` and `abs_tol_ellip`.  We then output a
    ComparisonShapeData object which stores the results.

    Note that `n_photons_per_trial` should be large (>~ 1e6) to ensure that any biases detected
    between the photon shooting and DFT-drawn images are due to numerical differences rather than
    biases on adaptive moments due to noise itself, a generic feature in this work.  This can be
    verified with a convergence test.

    @param gsobject         The GSObject for which this test is to be performed (prior
                            to PSF convolution if a PSF is also supplied via `psf_object`).
                            Note that this function will automatically handle integration over
                            a Pixel of width `pixel_scale`, so a Pixel should not be included in
                            the supplied `gsobject` (unless you really mean to include it, which
                            will be very rare in normal usage).
    @param psf_object       Optional additional PSF for tests of convolved objects, also a
                            GSObject.  Note that this function will automatically handle
                            integration over a Pixel of width `pixel_scale`, so this should not
                            be included in the supplied `psf_object`.  [default: None]
    @param rng              A BaseDeviate or derived deviate class instance to provide
                            the pseudo random numbers for the photon shooting.  [default: None]
    @param pixel_scale      The pixel scale to use in the test images. [default: 1]
    @param size             The size of the images in the rendering tests - all test images
                            are currently square. [default: 512]
    @param wmult            The `wmult` parameter used in .draw() (see the GSObject .draw()
                            method docs via `help(galsim.GSObject.draw)` for more details).
                            [default: 4]
    @param abs_tol_ellip    The test will keep iterating, adding ever greater numbers of
                            trials, until estimates of the 1-sigma standard error on mean 
                            ellipticity moments from photon-shot images are smaller than this
                            param value. [default: 1.e-5]
    @param abs_tol_size     The test will keep iterating, adding ever greater numbers of
                            trials, until estimates of the 1-sigma standard error on mean 
                            size moments from photon-shot images are smaller than this param
                            value. [default: 1.e-5]
    @param n_trials_per_iter  Number of trial images used to estimate (or successively
                            re-estimate) the standard error on the delta quantities above for
                            each iteration of the tests. [default: 32]
    @param n_photons_per_trial  Number of photons shot in drawShoot() for each trial.  This should
                            be large enough that any noise bias (a.k.a. noise rectification
                            bias) on moments estimates is small. [default: 1e7]
    @param moments          Set True to compare rendered images using AdaptiveMoments
                            estimates of simple observed estimates. [default: True]
    @param hsm              Should the rendered images be compared using HSM shear estimates?
                            (i.e. including a PSF correction for shears) [not implemented]
    """
    import sys
    import logging
    import time     

    # Some sanity checks on inputs
    if hsm is True:
        if psf_object is None:
            raise ValueError('An input psf_object is required for HSM shear estimate testing.')
        else:
            # Raise an apologetic exception about the HSM not yet being implemented!
            raise NotImplementedError('Sorry, HSM tests not yet implemented!')

    if rng is None:
        rng = galsim.BaseDeviate()

    # Then define some convenience functions for handling lists and multiple trial operations
    def _mean(array_like):
        return np.mean(np.asarray(array_like))

    def _stderr(array_like):
        return np.std(np.asarray(array_like)) / np.sqrt(len(array_like))

    def _shoot_trials_single(gsobject, ntrials, dx, imsize, rng, n_photons):
        """Convenience function to run `ntrials` and collect the results, uses only a single core.

        Uses a Python for loop but this is very unlikely to be a rate determining factor provided
        n_photons is suitably large (>1e6).
        """
        g1obslist = []
        g2obslist = []
        sigmalist = []
        im = galsim.ImageF(imsize, imsize)
        for i in xrange(ntrials):
            gsobject.drawShoot(im, dx=dx, n_photons=n_photons, rng=rng)
            res = im.FindAdaptiveMom()
            g1obslist.append(res.observed_shape.g1)
            g2obslist.append(res.observed_shape.g2)
            sigmalist.append(res.moments_sigma)
            logging.debug('Completed '+str(i + 1)+'/'+str(ntrials)+' trials in this iteration')
            #im.write('check_shoot_trial'+str(i + 1)) CHECK IMAGE
        return g1obslist, g2obslist, sigmalist

    # OK, that's the end of the helper functions-within-helper functions, back to the main unit

    # Start the timer
    t1 = time.time()

    # If a PSF is supplied, do the convolution, otherwise just use the gal_object
    if psf_object is None:
        logging.info('No psf_object supplied, running tests using input gsobject only')
        test_object = gsobject
    else:
        logging.info('Generating test_object by convolving gsobject with input psf_object')
        test_object = galsim.Convolve([gsobject, psf_object])

    # Draw the FFT image, only needs to be done once
    # For the FFT drawn image we need to include the galsim.Pixel, for the photon shooting we don't!
    test_object_pixelized = galsim.Convolve([test_object, galsim.Pixel(pixel_scale)])
    im_draw = galsim.ImageF(size, size)
    test_object_pixelized.draw(im_draw, dx=pixel_scale, wmult=wmult)
    res_draw = im_draw.FindAdaptiveMom()
    sigma_draw = res_draw.moments_sigma
    g1obs_draw = res_draw.observed_shape.g1
    g2obs_draw = res_draw.observed_shape.g2

    # Setup storage lists for the trial shooting results
    sigma_shoot_list = []
    g1obs_shoot_list = []
    g2obs_shoot_list = [] 
    sigmaerr = 666. # Slightly kludgy but will not accidentally fail the first `while` condition
    g1obserr = 666.
    g2obserr = 666.

    # Initialize iteration counter
    itercount = 0

    # Then begin while loop, farming out sets of n_trials_per_iter trials until we get the
    # statistical accuracy we require 
    while (g1obserr > abs_tol_ellip) or (g2obserr > abs_tol_ellip) or (sigmaerr > abs_tol_size):

        # Run the trials using helper function
        g1obs_list_tmp, g2obs_list_tmp, sigma_list_tmp = _shoot_trials_single(
            test_object, n_trials_per_iter, pixel_scale, size, rng, n_photons_per_trial)

        # Collect results and calculate new standard error
        g1obs_shoot_list.extend(g1obs_list_tmp)
        g2obs_shoot_list.extend(g2obs_list_tmp)
        sigma_shoot_list.extend(sigma_list_tmp)
        g1obserr = _stderr(g1obs_shoot_list)
        g2obserr = _stderr(g2obs_shoot_list)
        sigmaerr = _stderr(sigma_shoot_list)
        itercount += 1
        sys.stdout.write(".") # This doesn't add a carriage return at the end of the line, nice!
        logging.debug('Completed '+str(itercount)+' iterations')
        logging.debug(
            '(g1obserr, g2obserr, sigmaerr) = '+str(g1obserr)+', '+str(g2obserr)+', '+str(sigmaerr))

    sys.stdout.write("\n")

    # Take the runtime and collate results into a ComparisonShapeData
    runtime = time.time() - t1
    results = ComparisonShapeData(
        g1obs_draw, g2obs_draw, sigma_draw,
        _mean(g1obs_shoot_list), _mean(g2obs_shoot_list), _mean(sigma_shoot_list),
        g1obserr, g2obserr, sigmaerr, size, pixel_scale, wmult, itercount, n_trials_per_iter,
        n_photons_per_trial, runtime, gsobject=gsobject, psf_object=psf_object)

    logging.info('\n'+str(results))
    return results

def compare_dft_vs_photon_config(config, gal_num=0, random_seed=None, nproc=None, pixel_scale=None,
                                 size=None, wmult=None, abs_tol_ellip=1.e-5, abs_tol_size=1.e-5,
                                 n_trials_per_iter=32, n_max_iter=-1, n_photons_per_trial=1e7,
                                 moments=True, hsm=False, logger=None):
    """Take an input config dictionary and render the object it describes in two ways, comparing
    results at high precision. 

    The config dictionary can contain either: (i) one single object, (ii) a collection of objects, 
    each one of them repeated in a Sequence `n_trials_per_iter` times. The image type should be 
    'Single'. Example config fragment:

        &n_trials_per_iter 32 
        gal :
          type : Sersic    
          half_light_radius : 
            type : Catalog , 
            col : 2,  index : { type: Sequence, repeat: *n_trials_per_iter} }
          n : 
            type : Catalog , 
            col : 1,  
            index : { type: Sequence, repeat: *n_trials_per_iter} 
        ...
        image: { type : Single  ... }

    For both cases there should be no randomly selected parameters in the galaxy and PSF config 
    specification. 

    For an example of defining a config dictionary of the sort suitable for input to this function,
    see examples/demo8.py in the GalSim repository.

    Using both photon shooting (via drawShoot()) and Discrete Fourier Transform (via draw()) to
    render images, we compare the numerical values of adaptive moments estimates of ellipticity
    and size to check consistency.

    We generate successive sets of `n_trials_per_iter` photon-shot images, using 
    `n_photons_per_trial` photons in each image, until the standard error on the mean absolute size
    and ellipticty drop below `abs_tol_size` and `abs_tol_ellip`.  We then output a
    ComparisonShapeData object which stores the results.

    Note that `n_photons_per_trial` should be large (>~ 1e6) to ensure that any biases detected
    between the photon shooting and DFT-drawn images are due to numerical differences rather than
    biases on adaptive moments due to noise itself, a generic feature in this work.  This can be
    verified with a convergence test.

    @param config           GalSim config dictionary describing the GSObject we wish to test
                            (see e.g. examples/demo8.py).
    @param gal_num          Number for the galaxy in the config dictionary, which will be 
                            passed to the config system. It relates to `obj_num` in the config
                            system by obj_num = gal_num * n_trials_per_iter (assuming the
                            config is created correctly as explained in the example above)
                            [default: 0]
    @param random_seed      Integer to be used as the basis of all seeds for the random number
                            generator, overrides any value in config['image']. [default: None]
    @param nproc            Number of cpu processes to run in parallel, overrides any value
                            in config['image']. [default: None]
    @param pixel_scale      The pixel scale to use in the test images, overrides any value in
                            config['image']. [default: None]
    @param size             The size of the images in the rendering tests - all test images
                            are currently square, overrides any value in config['image'].
                            [default: None]
    @param wmult            The `wmult` parameter used in .draw() (see the GSObject .draw()
                            method docs via `help(galsim.GSObject.draw)` for more details),
                            overrides any value in config['image']. [default: None]
    @param abs_tol_ellip    The test will keep iterating, adding ever greater numbers of
                            trials, until estimates of the 1-sigma standard error on mean 
                            ellipticity moments from photon-shot images are smaller than this
                            param value. If `moments=False`, then using the measurements 
                            from HSM. [default: 1.e-5]
    @param abs_tol_size     The test will keep iterating, adding ever greater numbers of
                            trials, until estimates of the 1-sigma standard error on mean 
                            size moments from photon-shot images are smaller than this param
                            value. If `moments=False`, then using the measurements 
                            from HSM. [default: 1.e-5]
    @param n_trials_per_iter  Number of trial images used to estimate (or successively
                            re-estimate) the standard error on the delta quantities above for
                            each iteration of the tests. [default: 32]
    @param n_max_iter       Maximum number of iterations. After reaching it, the current
                            uncertainty on shape measurement is reported, even if
                            `abs_tol_ellip` and `abs_tol_size` was not reached. If a negative
                            number is supplied, then there is no limit on number of 
                            iterations. [default: -1]
    @param n_photons_per_trial  Number of photons shot in drawShoot() for each trial.  This should
                            be large enough that any noise bias (a.k.a. noise rectification
                            bias) on moments estimates is small. [default: 1e7]
    @param moments          Set True to compare rendered images using FindAdaptiveMoment()
                            estimates of simple observed estimates. [default: True]
    @param hsm              Should the rendered images be compared using HSM shear estimates?
                            (i.e. including a PSF correction for shears) [default: False]
    @param logger           Logging Logger instance to record output and pass down to the
                            config layer for debuging / verbose output if desired. [default: None]
    """
    import sys
    import logging
    import time     

    # Some sanity checks on inputs
    if moments is False and hsm is False:
        raise ValueError("At least one of 'moments','hsm' is required to be True")

    # Then check the config inputs, overriding and warning where necessary
    if random_seed is None:
        if 'random_seed' in config['image']:
            pass
        else:
            raise ValueError('Required input random_seed not set via kwarg or in config')
    else:
        if 'random_seed' in config['image']:
            import warnings
            warnings.warn(
                'Overriding random_seed in config with input kwarg value '+str(random_seed))
        config['image']['random_seed'] = random_seed

    if nproc is None:
        if 'nproc' in config['image']:
            pass
        else:
            from multiprocessing import cpu_count
            config['image']['nproc'] = cpu_count()
    else:
        if 'nproc' in config['image']:
            import warnings
            warnings.warn(
                'Overriding nproc in config with input kwarg value '+str(nproc))
        config['image']['nproc'] = nproc

    if pixel_scale is None:
        if 'pixel_scale' in config['image']:
            pass
        else:
            raise ValueError('Required input pixel_scale not set via kwarg or in image config')
    else:
        if 'pixel_scale' in config['image']:
            import warnings
            warnings.warn(
                'Overriding pixel_scale in config with input kwarg value '+str(pixel_scale))
        config['image']['pixel_scale'] = pixel_scale

    if size is None:
        if 'size' in config['image']:
            pass
        else:
            raise ValueError('Required input size not set via kwarg or in image config')
    else:
        if 'size' in config['image']:
            import warnings
            warnings.warn(
                'Overriding size in config with input kwarg value '+str(size))
        config['image']['size'] = size

    if wmult is None:
        if 'wmult' in config['image']:
            pass
        else:
            raise ValueError('Required input wmult not set via kwarg or in image config')
    else:
        if 'wmult' in config['image']:
            import warnings
            warnings.warn(
                'Overriding wmult in config with input kwarg value '+str(wmult))
        config['image']['wmult'] = wmult

    # Then define some convenience functions for handling lists and multiple trial operations
    def _mean(array_like):
        return np.mean(np.asarray(array_like))

    def _stderr(array_like):
        return np.std(np.asarray(array_like)) / np.sqrt(len(array_like))

    # OK, that's the end of the helper functions-within-helper functions, back to the main unit

    # Start the timer
    t1 = time.time()

    # calculate the obj_num in the config system
    obj_num = n_trials_per_iter*gal_num
    
    # Draw the FFT image, only needs to be done once
    # The BuidImage function stores things in the config that aren't picklable.
    # If you want to use config later for multiprocessing, you have to deepcopy it here.
    config1 = galsim.config.CopyConfig(config)

    # choose a shear estimator - I chose KSB, because then corrected_g1 is available
    hsm_shear_est = 'KSB'

    # get the fft image
    im_draw, im_psf, _, _ = galsim.config.BuildImage(
      config1, obj_num=obj_num, make_psf_image=True, logger=logger)

    # get the moments for FFT image
    if moments:
        res_draw = im_draw.FindAdaptiveMom()
        sigma_draw = res_draw.moments_sigma
        g1obs_draw = res_draw.observed_shape.g1
        g2obs_draw = res_draw.observed_shape.g2

    # Get the HSM for FFT image
    if hsm:
        res_draw_hsm= galsim.hsm.EstimateShear(im_draw,im_psf,strict=True,
                                               shear_est=hsm_shear_est)
        g1hsm_draw = res_draw_hsm.corrected_g1
        g2hsm_draw = res_draw_hsm.corrected_g2
        sighs_draw = res_draw_hsm.moments_sigma   # Short for sigma_hsm, to fit it in 5 characters

    
    # Setup storage lists for the trial shooting results
    sighs_shoot_list = []
    sigma_shoot_list = []
    g1obs_shoot_list = []
    g2obs_shoot_list = [] 
    g1hsm_shoot_list = []
    g2hsm_shoot_list = [] 
    sigmaerr = 666. # Slightly kludgy but will not accidentally fail the first `while` condition
    sighserr = 666. # Shorthand for sigma_hsm, to fit it in 5 characters
    g1obserr = 666.
    g2obserr = 666.
    g1hsmerr = 666.
    g2hsmerr = 666.

    # Initialize iteration counter
    itercount = 0

    # Change the draw_method to photon shooting
    # We'll also use a new copy here so that this function is non-destructive of any input
    config2 = galsim.config.CopyConfig(config)
    config2['image']['draw_method'] = 'phot'
    config2['image']['n_photons'] = n_photons_per_trial

    # Then begin while loop, farming out sets of n_trials_per_iter trials until we get the
    # statistical accuracy we require
    start_random_seed = config2['image']['random_seed'] 

    # If using moments, then the criteria will be on observed g1,g2,sigma, else on hsm corrected.
    # Ideally we would use some sort of pointer here, but I am going to update these at the end 
    # of the loop
    if moments:     
        err_g1_use,err_g2_use,err_sig_use = (g1obserr,g2obserr,sigmaerr)
    else:           
        err_g1_use,err_g2_use,err_sig_use = (g1hsmerr,g2hsmerr,sighserr)

    while (err_g1_use>abs_tol_ellip) or (err_g2_use>abs_tol_ellip) or (err_sig_use>abs_tol_size) :
        if n_max_iter > 0 and itercount >= n_max_iter: break

        # Reset the random_seed depending on the iteration number so that these never overlap
        config2['image']['random_seed'] = start_random_seed + itercount * (n_trials_per_iter + 1)

        # Run the trials using galsim.config.BuildImages function
        trial_images = galsim.config.BuildImages( 
            nimages=n_trials_per_iter, obj_num=obj_num,
            config=config2, logger=logger , nproc=config2['image']['nproc'])[0] 

        # Collect results 
        trial_results = []
        trial_results_hsm = []
        for image in trial_images:

            if moments:
                trial_results += [image.FindAdaptiveMom()]

            if hsm:
                trial_results_hsm += [galsim.hsm.EstimateShear(image,im_psf,strict=True,
                                                               shear_est=hsm_shear_est)]

        # Get lists of g1,g2,sigma estimate (this might be quicker using a single list comprehension
        # to get a list of (g1,g2,sigma) tuples, and then unzip with zip(*), but this is clearer)
        if moments:
            g1obs_shoot_list.extend([res.observed_shape.g1 for res in trial_results]) 
            g2obs_shoot_list.extend([res.observed_shape.g2 for res in trial_results]) 
            sigma_shoot_list.extend([res.moments_sigma for res in trial_results])
        if hsm:
            g1hsm_shoot_list.extend([res.corrected_g1 for res in trial_results_hsm]) 
            g2hsm_shoot_list.extend([res.corrected_g2 for res in trial_results_hsm])   
            sighs_shoot_list.extend([res.moments_sigma for res in trial_results_hsm])   

        #Then calculate new standard error
        if moments:
            g1obserr = _stderr(g1obs_shoot_list)
            g2obserr = _stderr(g2obs_shoot_list)
            sigmaerr = _stderr(sigma_shoot_list)  
        if hsm:
            g1hsmerr = _stderr(g1hsm_shoot_list)
            g2hsmerr = _stderr(g2hsm_shoot_list)
            sighserr = _stderr(sighs_shoot_list)

        itercount += 1
        sys.stdout.write(".") # This doesn't add a carriage return at the end of the line, nice!
        if logger:
            logger.debug('Completed '+str(itercount)+' iterations')
            logger.debug(
                '(g1obserr, g2obserr, g1hsmerr, g2hsmerr, sigmaerr, sigmaerr_hsm) = '
                +str(g1obserr)+', '+str(g2obserr)+', '+str(g1hsmerr)+', '+str(g2hsmerr)+', '
                +str(sigmaerr) + ', ' + str(sighserr) )

        # assing the variables governing the termination
        if moments:     
            err_g1_use,err_g2_use,err_sig_use = (g1obserr,g2obserr,sigmaerr)
        else:           
            err_g1_use,err_g2_use,err_sig_use = (g1hsmerr,g2hsmerr,sighserr)

    sys.stdout.write("\n")
         
    # prepare results for the ComparisonShapeData
    NO_HSM_OUTPUT_VALUE = 77
    NO_OBS_OUTPUT_VALUE = 88

    if moments:
        # get statistics
        mean_g1obs = _mean(g1obs_shoot_list) 
        mean_g2obs = _mean(g2obs_shoot_list) 
        mean_sigma = _mean(sigma_shoot_list)
    else:
        # assign the values to a NO_OBS_OUTPUT_VALUE flag
        mean_g1obs = mean_g2obs = NO_OBS_OUTPUT_VALUE
        g1obserr = g2obserr = NO_OBS_OUTPUT_VALUE
        g1obs_draw = g2obs_draw = NO_OBS_OUTPUT_VALUE
        sigma_draw = mean_sigma = sigmaerr = NO_OBS_OUTPUT_VALUE
    if hsm:
        mean_g1hsm = _mean(g1hsm_shoot_list)
        mean_g2hsm = _mean(g2hsm_shoot_list)
        mean_sighs = _mean(sighs_shoot_list)
    else:
        mean_g1hsm = mean_g2hsm = NO_HSM_OUTPUT_VALUE
        g1hsmerr = g2hsmerr = NO_HSM_OUTPUT_VALUE
        g1hsm_draw = g2hsm_draw = NO_HSM_OUTPUT_VALUE
        sighs_draw = mean_sighs = sighserr = NO_HSM_OUTPUT_VALUE


    # Take the runtime and collate results into a ComparisonShapeData
    runtime = time.time() - t1
    results = ComparisonShapeData(
        g1obs_draw, g2obs_draw, g1hsm_draw, g2hsm_draw, sigma_draw, sighs_draw ,
        mean_g1obs, mean_g2obs, mean_g1hsm , mean_g2hsm , mean_sigma , mean_sighs ,
        g1obserr, g2obserr, g1hsmerr, g2hsmerr, sigmaerr, sighserr ,
        config2['image']['size'], config2['image']['pixel_scale'],
        wmult, itercount, n_trials_per_iter, n_photons_per_trial, runtime, config=config2)

    if logger: logging.info('\n'+str(results))
    return results
