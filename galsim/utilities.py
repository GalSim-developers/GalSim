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
        try:
            return galsim.Interpolant2d(interpolant)
        except:
            raise RuntimeError('Specified interpolant is not valid!')

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

    - delta_g2obs: estimated mean difference between i) observed_shape.g2 from images of the same
      GSObject rendered with .drawShoot(), and ii) `g1obs_draw`.

    - delta_sigma: estimated mean difference between i) moments_sigma from images of the same
      GSObject rendered with .drawShoot(), and ii) `sigma_draw`.

    - err_g1obs: standard error in `delta_g1obs` estimated from the test sample.

    - err_g2obs: standard error in `delta_g2obs` estimated from the test sample.

    - err_sigma: standard error in `delta_sigma` estimated from the test sample.

    The ComparisonShapeData instance also stores much of the meta-information about the tests:

    - gsobject: the galsim.GSObject for which this test was performed (prior to PSF convolution if
      a PSF was also supplied).

    - psf_object: the optional additional PSF supplied by the user for tests of convolved objects,
      will be `None` if not used.

    - imsize: the size of the  images tested - all test images are currently square.

    - dx: the pixel scale in the images tested.

    - wmult: the `wmult` parameter used in .draw() (see the GSObject .draw() method docs for more
      details).

    - n_iterations: number of iterations of `n_trials` trials required to get delta quantities to
      the above accuracy.

    - n_trials_per_iter: number of trial images of used to estimate or successively re-estimate the
      standard error on the delta quantities above for each iteration.

    - n_photons_per_trial: number of photons shot in drawShoot() for each trial.

    - time: the time taken to perform the test.

    Note this is really only a simple storage container for the results above.  All of the
    non trivial calculation is completed before a ComparisonShapeData instance is initialized,
    typically in the function compare_object_dft_vs_photon.
    """
    def __init__(self, g1obs_draw, g2obs_draw, sigma_draw, g1obs_shoot, g2obs_shoot, sigma_shoot,
                 err_g1obs, err_g2obs, err_sigma, gsobject, psf_object, imsize, dx, wmult,
                 n_iterations, n_trials_per_iter, n_photons_per_trial, time):

        self.g1obs_draw = g1obs_draw
        self.g2obs_draw = g2obs_draw
        self.sigma_draw = sigma_draw
        
        self.delta_g1obs = g1obs_draw - g1obs_shoot
        self.delta_g2obs = g2obs_draw - g2obs_shoot
        self.delta_sigma = sigma_draw - sigma_shoot

        self.err_g1obs = err_g1obs
        self.err_g2obs = err_g2obs
        self.err_sigma = err_sigma

        self.gsobject = gsobject
        self.psf_object = psf_object
        self.imsize = imsize
        self.dx = dx
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
                 "time taken = "+str(self.time)+" s" 
        return retval

    # Reuse the __str__ method for __repr__
    __repr__ = __str__


def compare_dft_vs_photon_object(gsobject, psf_object=None, rng=None, dx=1., imsize=512, wmult=4.,
                                 abs_tol_ellip=1.e-5, abs_tol_size=1.e-5, n_trials_per_iter=32,
                                 n_photons_per_trial=1e7, moments=True, hsm=False):
    """Take an input object and render it in two ways comparing results at high precision.

    Using both photon shooting (via drawShoot) and Discrete Fourier Transform (via shoot) to render
    images, we compare  the numerical values of adaptive moments and optionally HSM shear estimates,
    or both, to check consistency.

    This function takes actual GSObjects as its input, but because these are not yet picklable this
    means that the internals cannot be parallelized using the Python multiprocessing module.  For
    a parallelized function, that instead uses a config dictionary to specify the test objects, see
    the function compare_dft_vs_photon_config() in this module.

    We generate successive sets of `n_trials_per_iter` photon-shot images, using 
    `n_photons_per_trial` photons in each image, until the standard error on the mean absolute size
    and ellipticty drop below `abs_tol_size` and `abs_tol_ellip`.  We then output a
    ComparisonShapeData object.

    @param gsobject               the galsim.GSObject for which this test is to be performed (prior
                                  to PSF convolution if a PSF is also supplied via `psf_object`).

    @param psf_object             optional additional PSF for tests of convolved objects, also a
                                  galsim.GSObject.

    @param rng                    galsim.BaseDeviate or derived deviate class instance to provide
                                  the pseudo random numbers for the photon shooting.

    @param dx                     the pixel scale to use in the test images.

    @param imsize                 the size of the images in the rendering tests - all test images
                                  are currently square.

    @param wmult                  the `wmult` parameter used in .draw() (see the GSObject .draw()
                                  method docs via `help(galsim.GSObject.draw)` for more details).

    @abs_tol_ellip                the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  ellipticity moments from photon-shot images are smaller than this
                                  param value.

    @abs_tol_size                 the test will keep iterating, adding ever greater numbers of
                                  trials, until estimates of the 1-sigma standard error on mean 
                                  size moments from photon-shot images are smaller than this param
                                  value.

    @n_trials_per_iter            number of trial images used to estimate (or successively
                                  re-estimate) the standard error on the delta quantities above for
                                  each iteration of the tests. Default = 32.

    @n_photons_per_trial          number of photons shot in drawShoot() for each trial.  This should
                                  be large enough that any noise bias (a.k.a. noise rectification
                                  bias) on moments estimates is small. Default ~1e7 should be
                                  sufficient.

    @param moments                set True to compare rendered images using AdaptiveMoments
                                  estimates of simple observed estimates (default=`True`).

    @param hsm                    set True to compare rendered images using HSM shear estimates
                                  (i.e. including a PSF correction for shears; default=`False`, not
                                  yet implemented).
    """
    import logging
    import time     

    # Some sanity checks on inputs
    if hsm is True:
        if psf_object is None:
            raise ValueError('An input psf_object is required for HSM shear estimate testing.')
        else:
            # Raise an apologetic exception about the HSM not yet being implemented!
            raise NotImplementedError('Sorry, HSM tests not yet implemented!')

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

    # Draw the shoot image, only needs to be done once
    im_draw = galsim.ImageF(imsize, imsize)
    test_object.draw(im_draw, dx=dx, wmult=wmult)
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
            test_object, n_trials_per_iter, dx, imsize, rng, n_photons_per_trial)

        # Collect results and calculate new standard error
        g1obs_shoot_list.extend(g1obs_list_tmp)
        g2obs_shoot_list.extend(g2obs_list_tmp)
        sigma_shoot_list.extend(sigma_list_tmp)
        g1obserr = _stderr(g1obs_shoot_list)
        g2obserr = _stderr(g2obs_shoot_list)
        sigmaerr = _stderr(sigma_shoot_list)
        itercount += 1
        logging.debug('Completed '+str(itercount)+' iterations')
        logging.debug(
            '(g1obserr, g2obserr, sigmaerr) = '+str(g1obserr)+', '+str(g2obserr)+', '+str(sigmaerr))

    # Take the runtime and collate results into a ComparisonShapeData
    runtime = time.time() - t1
    results = ComparisonShapeData(
        g1obs_draw, g2obs_draw, sigma_draw,
        _mean(g1obs_shoot_list), _mean(g2obs_shoot_list), _mean(sigma_shoot_list),
        g1obserr, g2obserr, sigmaerr, gsobject, psf_object, imsize, dx, wmult,
        itercount, n_trials_per_iter, n_photons_per_trial, runtime)

    logging.info(str(results))
    return results

#@random_seed                  If set, an integer to be used as the basis for the random number
#                              generator; if `ncores` > 1 (see below) then `random_seed` *must*
#                              be set and will be used to generated multiple seeds (TODO: TRY TO
#                              DO THIS BETTER!??).  If `None`, an internally determined seed will
#                              be used.
#
#@ncores                       Number of cores to use, switches the parallel processing behaviour
#                                  `ncores`=1 (default) - Use only single core, serial processing
#                                  `ncores`>1           - Use ncores independent processes for
#                                                         shooting photons through each set of
#                                                         `ntrials_per_iter` processes: works
#                                                         best if `ntrials_per_iter % ncores`= 0.
#                                  `ncores`=`None`      - Let the Python multiprocessing module
#                                                         determine the number of cores to use,
#                                                         processing in parallel.
