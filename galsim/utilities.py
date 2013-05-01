# Copyright 2012, 2013 The GalSim developers:
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
#
# GalSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GalSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GalSim.  If not, see <http://www.gnu.org/licenses/>
#
"""@file utilities.py
Module containing general utilities for the GalSim software.
"""

import numpy as np
import galsim

def roll2d(image, (iroll, jroll)):
    """Perform a 2D roll (circular shift) on a supplied 2D numpy array, conveniently.

    @param image            the numpy array to be circular shifted.
    @param (iroll, jroll)   the roll in the i and j dimensions, respectively.

    @returns the rolled image.
    """
    return np.roll(np.roll(image, jroll, axis=1), iroll, axis=0)

def kxky(array_shape=(256, 256)):
    """Return the tuple kx, ky corresponding to the DFT of a unit integer-sampled array of input
    shape.
    
    Uses the SBProfile conventions for Fourier space, so k varies in approximate range (-pi, pi].
    Uses the most common DFT element ordering conventions (and those of FFTW), so that `(0, 0)`
    array element corresponds to `(kx, ky) = (0, 0)`.

    See also the docstring for np.fftfreq, which uses the same DFT convention, and is called here,
    but misses a factor of pi.
    
    Adopts Numpy array index ordering so that the trailing axis corresponds to kx, rather than the
    leading axis as would be expected in IDL/Fortran.  See docstring for numpy.meshgrid which also
    uses this convention.

    @param array_shape   the Numpy array shape desired for `kx, ky`. 
    """
    # Note: numpy shape is y,x
    k_xaxis = np.fft.fftfreq(array_shape[1]) * 2. * np.pi
    k_yaxis = np.fft.fftfreq(array_shape[0]) * 2. * np.pi
    return np.meshgrid(k_xaxis, k_yaxis)

def g1g2_to_e1e2(g1, g2):
    """Convenience function for going from (g1, g2) -> (e1, e2).

    Here g1 and g2 are reduced shears, and e1 and e2 are distortions - see shear.py for definitions
    of reduced shear and distortion in terms of axis ratios or other ways of specifying ellipses.
    @param g1  First reduced shear component (along pixel axes)
    @param g2  Second reduced shear component (at 45 degrees with respect to image axes)
    @returns The corresponding distortions, e1 and e2.
    """
    # Conversion:
    # e = (a^2-b^2) / (a^2+b^2)
    # g = (a-b) / (a+b)
    # b/a = (1-g)/(1+g)
    # e = (1-(b/a)^2) / (1+(b/a)^2)
    gsq = g1*g1 + g2*g2
    if gsq > 0.:
        g = np.sqrt(gsq)
        boa = (1-g) / (1+g)
        e = (1 - boa*boa) / (1 + boa*boa)
        e1 = g1 * (e/g)
        e2 = g2 * (e/g)
        return e1, e2
    elif gsq == 0.:
        return 0., 0.
    else:
        raise ValueError("Input |g|^2 < 0, cannot convert.")

class AttributeDict(object):
    """Dictionary class that allows for easy initialization and refs to key values via attributes.

    NOTE: Modified a little from Jim's bot.git AttributeDict class so that tab completion now works
    in ipython since attributes are actually added to __dict__.
    
    HOWEVER this means the __dict__ attribute has been redefined to be a collections.defaultdict()
    so that Jim's previous default attribute behaviour is also replicated.
    """
    def __init__(self):
        import collections
        object.__setattr__(self, "__dict__", collections.defaultdict(AttributeDict))

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def merge(self, other):
        self.__dict__.update(other.__dict__)

    def _write(self, output, prefix=""):
        for k, v in self.__dict__.iteritems():
            if isinstance(v, AttributeDict):
                v._write(output, prefix="{0}{1}.".format(prefix, k))
            else:
                output.append("{0}{1} = {2}".format(prefix, k, repr(v)))

    def __nonzero__(self):
        return not not self.__dict__

    def __repr__(self):
        output = []
        self._write(output, "")
        return "\n".join(output)

    __str__ = __repr__

    def __len__(self):
        return len(self.__dict__)

def rand_arr(shape, deviate):
    """Function to make a 2d array of random deviates (of any sort).

    @param shape A list of length 2, indicating the desired 2d array dimensions
    @param deviate Any GalSim deviate (see random.py) such as UniformDeviate, GaussianDeviate,
    etc. to be used to generate random numbers
    @returns A Numpy array of the desired dimensions with random numbers generated using the
    supplied deviate.
    """
    if len(shape) is not 2:
        raise ValueError("Can only make a 2d array from this function!")
    # note reversed indices due to Numpy vs. Image array indexing conventions!
    tmp_img = galsim.ImageD(shape[1], shape[0])
    galsim.DeviateNoise(deviate).applyTo(tmp_img.view())
    return tmp_img.array

def convert_interpolant_to_2d(interpolant):
    """Convert a given interpolant to an Interpolant2d if it is given as a string or 1-d.
    """
    if interpolant == None:
        return None  # caller is responsible for setting a default if desired.
    elif isinstance(interpolant, galsim.Interpolant2d):
        return interpolant
    elif isinstance(interpolant, galsim.Interpolant):
        return galsim.InterpolantXY(interpolant)
    else:
        # Will raise an appropriate exception if this is invalid.
        return galsim.Interpolant2d(interpolant)


class ComparisonShapeData(object):
    """A class to contain the outputs of a comparison between photon shooting and DFT rendering of
    GSObjects, as measured by the HSM module's FindAdaptiveMom or (in future) EstimateShearHSM.

    Currently this class contains the following attributes (see also the HSMShapeData
    documentation for a more detailed description of, e.g., observed_shape, moments_sigma)
    describing the results of the comparison:

    - g1obs_draw: observed_shape.g1 from adaptive moments on a GSObject image rendered using .draw()

    - g2obs_draw: observed_shape.g2 from adaptive moments on a GSObject image rendered using .draw()

    - sigma_draw: moments_sigma from adaptive moments on a GSObject image rendered using .draw()

    - delta_g1obs: estimated mean difference between i) observed_shape.g1 from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1obs_draw`.
      Defined `delta_g1obs = g1obs_draw - g1obs_shoot`.

    - delta_g2obs: estimated mean difference between i) observed_shape.g2 from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1obs_draw`.
      Defined `delta_g2obs = g2obs_draw - g2obs_shoot`.

    - delta_sigma: estimated mean difference between i) moments_sigma from images of the same
      GSObject rendered with .drawShoot(), and ii) `sigma_draw`.
      Defined `delta_sigma = sigma_draw - sigma_shoot`.

    - err_g1obs: standard error in `delta_g1obs` estimated from the test sample.

    - err_g2obs: standard error in `delta_g2obs` estimated from the test sample.

    - err_sigma: standard error in `delta_sigma` estimated from the test sample.

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
    typically in the function compare_object_dft_vs_photon.

    - gsobject: optional galsim.GSObject for which this test was performed (prior to PSF convolution
      if a PSF was also supplied).

    - psf_object: the optional additional PSF supplied by the user for tests of convolved objects,
      will be `None` if not used.

    - config: optional config object describing the gsobject and PSF if the config comparison script
      was used rather than the (single core only) direct object script.

    Either config, or gsobject, or gsobject and psf_object, must be set when a ComparisonShapeData
    instance is created or an Exception is raised.
    """
    def __init__(self, g1obs_draw, g2obs_draw, sigma_draw, g1obs_shoot, g2obs_shoot, sigma_shoot,
                 err_g1obs, err_g2obs, err_sigma, size, pixel_scale, wmult, n_iterations,
                 n_trials_per_iter, n_photons_per_trial, time, gsobject=None, psf_object=None,
                 config=None):
        """In general use you should not need to instantiate a ComparisonShapeData instance,
        as this is done within the `compare_dft_vs_photon_config`/`object` functions. 
        """

        self.g1obs_draw = g1obs_draw
        self.g2obs_draw = g2obs_draw
        self.sigma_draw = sigma_draw
        
        self.delta_g1obs = g1obs_draw - g1obs_shoot
        self.delta_g2obs = g2obs_draw - g2obs_shoot
        self.delta_sigma = sigma_draw - sigma_shoot

        self.err_g1obs = err_g1obs
        self.err_g2obs = err_g2obs
        self.err_sigma = err_sigma

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
                 "sigma_draw  = "+str(self.sigma_draw)+"\n"+\
                 "delta_sigma = "+str(self.delta_sigma)+" +/- "+str(self.err_sigma)+"\n"+\
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

    Using both photon shooting (via drawShoot) and Discrete Fourier Transform (via draw) to render
    images, we compare the numerical values of adaptive moments estimates of size and ellipticity to
    check consistency.

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

    @param gsobject               the galsim.GSObject for which this test is to be performed (prior
                                  to PSF convolution if a PSF is also supplied via `psf_object`).
                                  Note that this function will automatically handle integration over
                                  a galsim.Pixel of width `pixel_scale`, so a galsim.Pixel should 
                                  not be included in the supplied `gsobject` (unless you really mean
                                  to include it, which will be very rare in normal usage).

    @param psf_object             optional additional PSF for tests of convolved objects, also a
                                  galsim.GSObject.  Note that this function will automatically 
                                  handle integration over a galsim.Pixel of width `pixel_scale`,
                                  so this should not be included in the supplied `psf_object`.

    @param rng                    galsim.BaseDeviate or derived deviate class instance to provide
                                  the pseudo random numbers for the photon shooting.  If `None` on 
                                  input (default) a galsim.BaseDeviate is internally initialized.

    @param pixel_scale            the pixel scale to use in the test images.

    @param size                   the size of the images in the rendering tests - all test images
                                  are currently square.

    @param wmult                  the `wmult` parameter used in .draw() (see the GSObject .draw()
                                  method docs via `help(galsim.GSObject.draw)` for more details).

    @param abs_tol_ellip          the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  ellipticity moments from photon-shot images are smaller than this
                                  param value.

    @param abs_tol_size           the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  size moments from photon-shot images are smaller than this param
                                  value.

    @param n_trials_per_iter      number of trial images used to estimate (or successively
                                  re-estimate) the standard error on the delta quantities above for
                                  each iteration of the tests. Default = 32.

    @param n_photons_per_trial    number of photons shot in drawShoot() for each trial.  This should
                                  be large enough that any noise bias (a.k.a. noise rectification
                                  bias) on moments estimates is small. Default ~1e7 should be
                                  sufficient.

    @param moments                set True to compare rendered images using AdaptiveMoments
                                  estimates of simple observed estimates (default=`True`).

    @param hsm                    set True to compare rendered images using HSM shear estimates
                                  (i.e. including a PSF correction for shears; default=`False` as
                                  this feature is not yet implemented!)
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
        """Convenience function to run ntrials and collect the results, uses only a single core.

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
                                 n_trials_per_iter=32, n_photons_per_trial=1e7, moments=True,
                                 hsm=False, logger=None):
    """Take an input config dictionary and render the object it describes in two ways, comparing
    results at high precision. 

    The config dictionary can contain either: (i) one single object, (ii) a collection of objects, 
    each one of them repeated in a Sequence n_trials_per_iter times. The image type should be 
    'Single'. Example config fragment:

        &n_trials_per_iter 32 
        gal :
          type : Sersic    
          half_light_radius : 
            type : InputCatalog , 
            col : 2,  index : { type: Sequence, repeat: *n_trials_per_iter} }
          n : 
            type : InputCatalog , 
            col : 1,  
            index : { type: Sequence, repeat: *n_trials_per_iter} 
        ...
        image: { type : Single  ... }

    For both cases there should be no randomly selected parameters in the galaxy and PSF config 
    specification. 

    For an example of defining a config dictionary of the sort suitable for input to this function,
    see examples/demo8.py in the GalSim repository.

    Using both photon shooting (via drawShoot) and Discrete Fourier Transform (via shoot) to render
    images, we compare the numerical values of adaptive moments estimates of ellipticity and size 
    to check consistency.

    We generate successive sets of `n_trials_per_iter` photon-shot images, using 
    `n_photons_per_trial` photons in each image, until the standard error on the mean absolute size
    and ellipticty drop below `abs_tol_size` and `abs_tol_ellip`.  We then output a
    ComparisonShapeData object which stores the results.

    Note that `n_photons_per_trial` should be large (>~ 1e6) to ensure that any biases detected
    between the photon shooting and DFT-drawn images are due to numerical differences rather than
    biases on adaptive moments due to noise itself, a generic feature in this work.  This can be
    verified with a convergence test.

    @param config                 GalSim config dictionary describing the GSObject we wish to test
                                  (see e.g. examples/demo8.py).

    @param gal_num                number for the galaxy in the config dictionary, which will be 
                                  passed to the config system. It related to obj_num in the config
                                  system by obj_num = gal_num * n_trials_per_iter (assuming the
                                  config is created correctly as explained in the example above)

    @param random_seed            integer to be used as the basis of all seeds for the random number
                                  generator, overrides any value in config['image'].

    @param nproc                  number of cpu processes to run in parallel, overrides any value
                                  in config['image'].

    @param pixel_scale            the pixel scale to use in the test images, overrides any value in
                                  config['image'].

    @param size                   the size of the images in the rendering tests - all test images
                                  are currently square, overrides any value in config['image'].

    @param wmult                  the `wmult` parameter used in .draw() (see the GSObject .draw()
                                  method docs via `help(galsim.GSObject.draw)` for more details),
                                  overrides any value in config['image'].

    @param abs_tol_ellip          the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  ellipticity moments from photon-shot images are smaller than this
                                  param value.

    @param abs_tol_size           the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  size moments from photon-shot images are smaller than this param
                                  value.

    @param n_trials_per_iter      number of trial images used to estimate (or successively
                                  re-estimate) the standard error on the delta quantities above for
                                  each iteration of the tests. Default = 32.

    @param n_photons_per_trial    number of photons shot in drawShoot() for each trial.  This should
                                  be large enough that any noise bias (a.k.a. noise rectification
                                  bias) on moments estimates is small. Default ~1e7 should be
                                  sufficient.

    @param moments                set True to compare rendered images using AdaptiveMoments
                                  estimates of simple observed estimates (default=`True`).

    @param hsm                    set True to compare rendered images using HSM shear estimates
                                  (i.e. including a PSF correction for shears; default=`False` as
                                  this feature is not yet implemented!)

    @param logger                 logging Logger instance to record output and pass down to the
                                  config layer for debuging / verbose output if desired.
    """
    import sys
    import logging
    import time     

    # Some sanity checks on inputs
    if hsm is True:
        # Raise an apologetic exception about the HSM not yet being implemented!
        raise NotImplementedError('Sorry, HSM tests not yet implemented!')

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
    import copy
    config1 = copy.deepcopy(config)
    im_draw = galsim.config.BuildImage(config1, obj_num = obj_num, logger=logger)[0]
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

    # Change the draw_method to photon shooting
    # We'll also use a new copy here so that this function is non-destructive of any input
    config2 = copy.deepcopy(config)
    config2['image']['draw_method'] = 'phot'
    config2['image']['n_photons'] = n_photons_per_trial

    # Then begin while loop, farming out sets of n_trials_per_iter trials until we get the
    # statistical accuracy we require
    start_random_seed = config2['image']['random_seed'] 
    start_random_seed = config2['image']['random_seed'] 
    while (g1obserr > abs_tol_ellip) or (g2obserr > abs_tol_ellip) or (sigmaerr > abs_tol_size):

        # Reset the random_seed depending on the iteration number so that these never overlap
        config2['image']['random_seed'] = start_random_seed + itercount * (n_trials_per_iter + 1)

        # Run the trials using galsim.config.BuildImages function
        trial_images = galsim.config.BuildImages( nimages = n_trials_per_iter, obj_num = obj_num , 
          config = config2, logger=logger, nproc=config2['image']['nproc'])[0] 

        # Collect results 
        trial_results = [image.FindAdaptiveMom() for image in trial_images]

        # Get lists of g1,g2,sigma estimate (this might be quicker using a single list comprehension
        # to get a list of (g1,g2,sigma) tuples, and then unzip with zip(*), but this is clearer)
        g1obs_shoot_list.extend([res.observed_shape.g1 for res in trial_results]) 
        g2obs_shoot_list.extend([res.observed_shape.g2 for res in trial_results]) 
        sigma_shoot_list.extend([res.moments_sigma for res in trial_results])

        #Then calculate new standard error
        g1obserr = _stderr(g1obs_shoot_list)
        g2obserr = _stderr(g2obs_shoot_list)
        sigmaerr = _stderr(sigma_shoot_list)
        itercount += 1
        sys.stdout.write(".") # This doesn't add a carriage return at the end of the line, nice!
        if logger:
            logger.debug('Completed '+str(itercount)+' iterations')
            logger.debug(
                '(g1obserr, g2obserr, sigmaerr) = '+str(g1obserr)+', '+str(g2obserr)+', '+
            str(sigmaerr))

    sys.stdout.write("\n")


    # Take the runtime and collate results into a ComparisonShapeData
    runtime = time.time() - t1
    results = ComparisonShapeData(
        g1obs_draw, g2obs_draw, sigma_draw,
        _mean(g1obs_shoot_list), _mean(g2obs_shoot_list), _mean(sigma_shoot_list),
        g1obserr, g2obserr, sigmaerr, config2['image']['size'], config2['image']['pixel_scale'],
        wmult, itercount, n_trials_per_iter, n_photons_per_trial, runtime, config=config2)

    if logger: logging.info('\n'+str(results))
    return results
