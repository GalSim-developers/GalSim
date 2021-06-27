# Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

import numpy as np

from ..errors import GalSimBoundsError, GalSimConfigError
from ..position import PositionD
from ..bounds import BoundsD
from ..shapelet import Shapelet
from ..gsparams import GSParams
from ..config import InputLoader, RegisterInputType, RegisterObjectType
from ..config import GetAllParams, GetInputObj, SkipThisObject

class DES_Shapelet(object):
    """Class that handles DES files describing interpolated polar shapelet decompositions.
    These are stored as ``*_fitpsf.fits`` files.  They are not used in DES anymore, so this
    class is at best of historical interest

    The shapelet PSFs measure a shapelet decomposition of each star and interpolate the shapelet
    coefficients over the image positions.

    Unlike PSFEx, these PSF models are built directly in world coordinates.  The shapelets know
    about the WCS, so they are able to fit the shapelet model directly in terms of arcsec.
    Thus, the getPSF function always returns a profile in world coordinates.

    Typical usage:

        >>> des_shapelet = galsim.des.DES_Shapelet(fitpsf_file_name)
        >>> image_pos = galsim.PositionD(image_x, image_y)    # position in pixels on the image
        >>>                                                   # NOT in arcsec on the sky!
        >>> psf = des_shapelet.getPSF(image_pos)   # profile is in world coordinates

    Note that the returned psf here already includes the pixel.  This is what is sometimes
    called an "effective PSF".  Thus, you should not convolve by the pixel profile again
    (nor integrate over the pixel).  This would effectively include the pixel twice!

    This class will only interpolate within the defining bounds.  It won't extrapolate
    beyond the bounding box of where the stars defined the interpolation.
    If you try to use it with an invalid position, it will throw an IndexError.
    You can check whether a position is valid with

        >>> if des_shapelet.bounds.includes(pos):
        >>>     psf = des_shapelet.getPSF(pos)
        >>> else:
        >>>     [...skip this object...]


    Parameters:
        file_name:  The name of the file to be read in.
        dir:        Optionally a directory name can be provided if the file names do not
                    already include it. [default: None]
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, dir=None):
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        self.file_name = file_name
        self.read_fits()

    def read_fits(self):
        """Read in a DES_Shapelet stored in FITS file.
        """
        from .._pyfits import pyfits
        with pyfits.open(self.file_name) as fits:
            cat = fits[1].data
        # These fields each only contain one element, hence the [0]'s.
        self.psf_order = cat.field('psf_order')[0]
        self.psf_size = (self.psf_order+1) * (self.psf_order+2) // 2
        self.sigma = cat.field('sigma')[0]
        self.fit_order = cat.field('fit_order')[0]
        self.fit_size = (self.fit_order+1) * (self.fit_order+2) // 2
        self.npca = cat.field('npca')[0]

        self.bounds = BoundsD(
            float(cat.field('xmin')[0]), float(cat.field('xmax')[0]),
            float(cat.field('ymin')[0]), float(cat.field('ymax')[0]))

        self.ave_psf = cat.field('ave_psf')[0]
        assert self.ave_psf.shape == (self.psf_size,)

        # Note: older pyfits versions don't get the shape right.
        # For newer pyfits versions the reshape command should be a no op.
        self.rot_matrix = cat.field('rot_matrix')[0].reshape((self.psf_size,self.npca)).T
        assert self.rot_matrix.shape == (self.npca, self.psf_size)

        self.interp_matrix = cat.field('interp_matrix')[0].reshape((self.npca,self.fit_size)).T
        assert self.interp_matrix.shape == (self.fit_size, self.npca)

    def getBounds(self):
        return self.bounds

    def getSigma(self):
        return self.sigma

    def getOrder(self):
        return self.psf_order

    def getPSF(self, image_pos, gsparams=None):
        """Returns the PSF at position image_pos

        Parameters:
            image_pos:  The position in pixel units for which to build the PSF.
            gsparams:   An optional `GSParams` instance to pass to the constructed GSObject.
                        [default: None]

        Returns:
            the PSF as a galsim.Shapelet instance
        """
        psf = Shapelet(self.sigma, self.psf_order, self.getB(image_pos), gsparams=gsparams)

        # The fitpsf files were built with respect to (u,v) = (ra,dec).  The GalSim convention is
        # to use sky coordinates with u = -ra.  So we need to flip the profile across the v axis
        # to take u -> -u.
        psf = psf.transform(-1,0,0,1)

        return psf

    def getB(self, pos):
        """Get the B vector as a numpy array at position pos
        """
        if not self.bounds.includes(pos):
            raise GalSimBoundsError("position in DES_Shapelet.getPSF is out of bounds",
                                    pos, self.bounds)

        Px = self._definePxy(pos.x,self.bounds.xmin,self.bounds.xmax)
        Py = self._definePxy(pos.y,self.bounds.ymin,self.bounds.ymax)
        order = self.fit_order
        P = np.array([ Px[n-q] * Py[q] for n in range(order+1) for q in range(n+1) ])
        assert len(P) == self.fit_size

        # Note: This is equivalent to:
        #
        #     P = numpy.empty(self.fit_size)
        #     k = 0
        #     for n in range(self.fit_order+1):
        #         for q in range(n+1):
        #             P[k] = Px[n-q] * Py[q]
        #             k = k+1

        b1 = np.dot(P,self.interp_matrix)
        b = np.dot(b1,self.rot_matrix)
        assert len(b) == self.psf_size
        b += self.ave_psf
        return b

    def _definePxy(self, x, min, max):
        x1 = (2.*x-min-max)/(max-min)
        temp = np.empty(self.fit_order+1)
        temp[0] = 1
        if self.fit_order > 0:  # pragma: no branch (always true for file we have for testing.)
            temp[1] = x1
        for i in range(2,self.fit_order+1):
            temp[i] = ((2.*i-1.)*x1*temp[i-1] - (i-1.)*temp[i-2]) / float(i)
        return temp

# First we need to add the class itself as a valid input_type.
RegisterInputType('des_shapelet', InputLoader(DES_Shapelet))

# Also make a builder to create the PSF object for a given position.
# The builders require 4 args.
# config is a dictionary that includes 'type' plus other items you might want to allow or require.
# base is the top level config dictionary where some global variables are stored.
# ignore is a list of key words that might be in the config dictionary that you should ignore.
def BuildDES_Shapelet(config, base, ignore, gsparams, logger):
    """Build a GSObject representing the shapelet model at the correct location in the image in a
    config-processing context.

    This is used as object type ``DES_Shapelet`` in a config file.

    It requires the use of the ``des_shapelet`` input field.
    """
    des_shapelet = GetInputObj('des_shapelet', config, base, 'DES_Shapelet')

    opt = { 'flux' : float , 'num' : int, 'image_pos' : PositionD }
    params, safe = GetAllParams(config, base, opt=opt, ignore=ignore)

    if 'image_pos' in params:
        image_pos = params['image_pos']
    elif 'image_pos' in base:
        image_pos = base['image_pos']
    else:
        raise GalSimConfigError("DES_Shapelet requested, but no image_pos defined in base.")

    # Convert gsparams from a dict to an actual GSParams object
    if gsparams: gsparams = GSParams(**gsparams)
    else: gsparams = None

    if des_shapelet.getBounds().includes(image_pos):
        #psf = des_shapelet.getPSF(image_pos, gsparams)
        # Because of serialization issues, the above call doesn't work.  So we need to
        # repeat the internals of getPSF here.
        b = des_shapelet.getB(image_pos)
        sigma = des_shapelet.getSigma()
        order = des_shapelet.getOrder()
        psf = Shapelet(sigma, order, b, gsparams=gsparams).transform(-1,0,0,1)
    else:
        message = 'Position '+str(image_pos)+' not in interpolation bounds: '
        message += str(des_shapelet.getBounds())
        raise SkipThisObject(message)

    if 'flux' in params:
        psf = psf.withFlux(params['flux'])

    # The second item here is "safe", a boolean that declares whether the returned value is
    # safe to save and use again for later objects.  In this case, we wouldn't want to do
    # that, since they will be at different positions, so the interpolated PSF will be different.
    return psf, False

# Register this builder with the config framework:
RegisterObjectType('DES_Shapelet', BuildDES_Shapelet, input_type='des_shapelet')

