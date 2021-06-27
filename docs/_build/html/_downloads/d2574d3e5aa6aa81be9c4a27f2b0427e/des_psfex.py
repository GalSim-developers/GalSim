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

import os
import numpy as np

from ..errors import GalSimIncompatibleValuesError, GalSimConfigError
from ..fits import FitsHeader
from ..wcs import readFromFitsHeader, PixelScale
from ..position import PositionD
from ..image import Image
from ..interpolatedimage import InterpolatedImage
from ..interpolant import Lanczos
from ..gsparams import GSParams
from ..config import InputLoader, RegisterInputType, RegisterObjectType
from ..config import GetAllParams, GetInputObj


class DES_PSFEx(object):
    r"""Class that handles DES files describing interpolated principal component images
    of the PSF.  These are usually stored as \*_psfcat.psf files.

    PSFEx is software written by Emmanuel Bertin.  If you want more detail about PSFEx, please
    check out the web site:

    http://www.astromatic.net/software/psfex

    It builds PSF objects from images of stars in a given exposure, finds a reasonable basis
    set to describe those images, and then fits the coefficient of these bases as a function
    of the ``(x,y)`` position on the image.

    Note that while the interpolation is done in image coordinates, GalSim usually deals with
    object profiles in world coordinates.  However, PSFEx does not consider the WCS of the
    image when building its bases.  The bases are built in image coordinates.  So there are
    two options to get GalSim to handle this difference.

    1. Ignore the WCS of the original image.  In this case, the \*.psf files have all the
       information you need:

           >>> des_psfex = galsim.des.DES_PSFEx(fitpsf_file_name)
           >>> image_pos = galsim.PositionD(image_x, image_y)    # position in pixels on the image
           >>>                                                   # NOT in arcsec on the sky!
           >>> psf = des_psfex.getPSF(image_pos)      # profile is in image coordinates

       The psf profile that is returned will be in image coordinates.  Therefore, it should be
       drawn onto an image with no wcs.  (Or equivalently, one with ``scale = 1``.)  If you want
       to use this to convolve a galaxy profile, you would want to either project the galaxy
       (typically constructed in world coordinates) to the correct image coordinates or project
       the PSF up into world coordinates.

    2. Build the PSF in world coordinates directly.  The DES_PSFEx constructor can take an
       extra argument, either ``image_file_name`` or ``wcs``, to tell GalSim what WCS to use for
       the coversion between image and world coordinates.  The former option is the name of
       the file from which to read the WCS, which will often be more convenient, but you can
       also just pass in a WCS object directly.

           >>> des_psfex = galsim.des.DES_PSFEx(fitpsf_file_name, image_file_name)
           >>> image_pos = galsim.PositionD(image_x, image_y)    # position in pixels on the image
           >>>                                                   # NOT in arcsec on the sky!
           >>> psf = des_psfex.getPSF(image_pos)      # profile is in world coordinates

       This time the psf profile that is returned will already be in world coordinates as
       GalSim normally expects, so you can use it in the normal ways.  If you want to draw it
       (or a convolved object) onto an image with the original WCS at that location, you can use
       ``des_psfex.getLocalWCS(image_pos)`` for the local wcs at the location of the PSF.

    Note that the returned psf here already includes the pixel.  This is what is sometimes
    called an "effective PSF".  Thus, you should not convolve by the pixel profile again
    (nor integrate over the pixel).  This would effectively include the pixel twice!

    In GalSim, you should always pass ``method='no_pixel'`` when drawing images of objects
    convolved with PSFs produced with this class.  Other drawing methods, such as photon shooting
    (``method='phot'``) or an FFT (``method='fft'``), will result in convolving the pixel twice.

    Parameters:
       file_name:       The file name to be read in, or a pyfits HDU in which case it is
                        used directly instead of being opened.
       image_file_name: The name of the fits file of the original image (needed for the
                        WCS information in the header).  If unavailable, you may omit this
                        (or use None), but then the returned profiles will be in image
                        coordinates, not world coordinates. [default: None]
       wcs:             Optional way to provide the WCS if you already have it loaded from
                        the image file. [default: None]
       dir:              Optionally a directory name can be provided if the ``file_name``
                        does not already include it.  (The image file is assumed to be in
                        the same directory.) [default: None].
    """
    # For config, image_file_name is required, since that always works in world coordinates.
    _req_params = { 'file_name' : str }
    _opt_params = { 'dir' : str, 'image_file_name' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, image_file_name=None, wcs=None, dir=None):

        if dir:
            if not isinstance(file_name, str):
                raise TypeError("file_name must be a string")
            file_name = os.path.join(dir,file_name)
            if image_file_name is not None:
                image_file_name = os.path.join(dir,image_file_name)
        self.file_name = file_name
        if image_file_name:
            if wcs is not None:
                raise GalSimIncompatibleValuesError(
                    "Cannot provide both image_file_name and wcs",
                    image_file_name=image_file_name, wcs=wcs)
            header = FitsHeader(file_name=image_file_name)
            wcs, origin = readFromFitsHeader(header)
            self.wcs = wcs
        elif wcs:
            self.wcs = wcs
        else:
            self.wcs = None
        self.read()

    def read(self):
        from .._pyfits import pyfits
        if isinstance(self.file_name, str):
            hdu_list = pyfits.open(self.file_name)
            hdu = hdu_list[1]
        else:
            hdu = self.file_name
            hdu_list = None
        # Number of parameters used for the interpolation.  We require this to be 2.
        pol_naxis = hdu.header['POLNAXIS']

        # These are the names of the two axes.  Should be X_IMAGE, Y_IMAGE.
        # If they aren't, then the way we use the interpolation will be wrong.
        # Well, really they can also be XWIN_IMAGE, etc.  So just check that it
        # starts with X and ends with IMAGE.
        pol_name1 = hdu.header['POLNAME1']
        pol_name2 = hdu.header['POLNAME2']

        # Zero points and scale.  Interpolation is in terms of (x-x0)/xscale, (y-y0)/yscale
        pol_zero1 = hdu.header['POLZERO1']
        pol_zero2 = hdu.header['POLZERO2']
        pol_scal1 = hdu.header['POLSCAL1']
        pol_scal2 = hdu.header['POLSCAL2']

        # This defines the number of "context groups".
        # Here is Emmanuel's explanation:
        #
        #     POLNGRP is the number of "context groups". A group represents a set of variables
        #     (SExtractor measurements or FITS header parameters if preceded with ":") which share
        #     the same maximum polynomial degree. For instance if x and y are in group 1, and the
        #     degree of that group is 2, and z is in group 2 with degree 1, the polynomial will
        #     consist of:
        #         1, x, x^2, y, y.x, y^2, z, z.x^2, z.y, z.y.x, z.y^2
        #     (see eq 14 in https://www.astromatic.net/pubsvn/software/psfex/trunk/doc/psfex.pdf )
        #     By default, POLNGRP is 1 and the group contains X_IMAGE and Y_IMAGE measurements
        #     from SExtractor.
        #
        # For now, we require this to be 1, since I didn't have any files with POLNGRP != 1 to
        # test on.
        pol_ngrp = hdu.header['POLNGRP']

        # Which group each item is in.  We require group 1.
        pol_group1 = hdu.header['POLGRP1']
        pol_group2 = hdu.header['POLGRP2']

        # The degree of the polynomial.  E.g. POLDEG1 = 2 means the values will be:
        #     1, x, x^2, y, xy, y^2
        # If we start allowing POLNGRP > 1, there is a separate degree for each group.
        pol_deg = hdu.header['POLDEG1']

        # The number of axes in the basis object.  We require this to be 3.
        psf_naxis = hdu.header['PSFNAXIS']

        # The first two axes are the image size of the PSF postage stamp.
        psf_axis1 = hdu.header['PSFAXIS1']
        psf_axis2 = hdu.header['PSFAXIS2']

        # The third axis is the direction of the polynomial interpolation.  So it should
        # be equal to (d+1)(d+2)/2.
        psf_axis3 = hdu.header['PSFAXIS3']

        # This is the PSF "sample size".  Again, from Emmanuel:
        #
        #     PSF_SAMP is the sampling step of the PSF. PSF_SAMP=0.5 means that the PSF model has
        #     two samples per original image pixel (superresolution, so in automatic mode it is a
        #     sign that the original images were undersampled)
        #
        # In other words, it can be thought of as a unit conversion:
        #     "image pixels" / "psfex pixels"
        # So 1 image pixel = (1/psf_samp) psfex pixels.
        psf_samp = hdu.header['PSF_SAMP']

        # The basis object is a data cube (assuming PSFNAXIS==3)
        basis = hdu.data.field('PSF_MASK')[0]

        # Make sure to close the hdu before we might raise exceptions.
        if hdu_list:
            hdu_list.close()

        # Check for valid values of all these things.
        # Not sure which of these are actually required in PSFEx files, but this implementation
        # assumes these things are true, so if this fails, we probably need to rework some aspect
        # of this code.
        try:
            assert pol_naxis == 2
            assert pol_name1.startswith('X') and pol_name1.endswith('IMAGE')
            assert pol_name2.startswith('Y') and pol_name2.endswith('IMAGE')
            assert pol_ngrp == 1
            assert pol_group1 == 1
            assert pol_group2 == 1
            assert psf_naxis == 3
            assert psf_axis3 == ((pol_deg+1)*(pol_deg+2))//2
            assert basis.shape[0] == psf_axis3
            assert basis.shape[1] == psf_axis2
            assert basis.shape[2] == psf_axis1
        except AssertionError as e:
            raise OSError("PSFEx file %s is not as expected.\n%r"%(self.file_name, e))

        # Save some of these values for use in building the interpolated images
        self.basis = basis
        self.fit_order = pol_deg
        self.fit_size = psf_axis3
        self.x_zero = pol_zero1
        self.y_zero = pol_zero2
        self.x_scale = pol_scal1
        self.y_scale = pol_scal2
        self.sample_scale = psf_samp

    def getSampleScale(self):
        return self.sample_scale

    def getLocalWCS(self, image_pos):
        """If the original image was provided to the constructor, this will return the local
        WCS at a given location in that original image.  If not, this will return None.

        Parameter:
           image_pos (Position):    The position in pixels in the image.

        Returns:
           A LocalWCS or None
        """
        if self.wcs:
            return self.wcs.local(image_pos)
        else:
            return None

    def getPSF(self, image_pos, gsparams=None):
        """Returns the PSF at position ``image_pos``.

        Parameters:
           image_pos:   The position in image coordinates at which to build the PSF.
           gsparams:    A `GSParams` instance to pass to the constructed `GSObject`.
                        [defualt: None]

        Returns:
           the PSF as a `GSObject`
        """
        # Build an image version of the numpy array
        im = Image(self.getPSFArray(image_pos))

        # Build the PSF profile in the image coordinate system.
        psf = InterpolatedImage(im, scale=self.sample_scale, flux=1,
                                x_interpolant=Lanczos(3), gsparams=gsparams)

        # This brings if from image coordinates to world coordinates.
        if self.wcs:
            psf = self.wcs.toWorld(psf, image_pos=image_pos)

        return psf

    def getPSFArray(self, image_pos):
        """Returns the PSF image as a numpy array at position image_pos in image coordinates.
        """
        xto = self._define_xto( (image_pos.x - self.x_zero) / self.x_scale )
        yto = self._define_xto( (image_pos.y - self.y_zero) / self.y_scale )
        order = self.fit_order
        P = np.array([ xto[nx] * yto[ny] for ny in range(order+1) for nx in range(order+1-ny) ])
        assert len(P) == self.fit_size
        ar = np.tensordot(P,self.basis,(0,0)).astype(np.float32)
        # Note: This is equivalent to:
        #   ar = self.basis[0].astype(numpy.float32)
        #   for n in range(1,self.fit_order+1):
        #       for ny in range(n+1):
        #           nx = n-ny
        #           k = nx+ny*(self.fit_order+1)-ny*(ny-1)/2
        #           ar += xto[nx] * yto[ny] * self.basis[k]
        # which is pretty much Peter's version of this code.
        return ar

    def _define_xto(self, x):
        xto = np.empty(self.fit_order+1)
        xto[0] = 1
        for i in range(1,self.fit_order+1):
            xto[i] = x*xto[i-1]
        return xto


class PSFExLoader(InputLoader):
    # Allow the user to not provide the image file.  In this case, we'll grab the wcs from the
    # config dict.
    def getKwargs(self, config, base, logger):
        req = { 'file_name' : str }
        opt = { 'dir' : str, 'image_file_name' : str }
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

        if 'image_file_name' not in kwargs:
            if 'wcs' in base:
                kwargs['wcs'] = base['wcs']
            else:
                # Then we aren't doing normal config processing, so just use pixel scale = 1.
                kwargs['wcs'] = PixelScale(1.)

        return kwargs, safe

# First we need to add the class itself as a valid input_type.
RegisterInputType('des_psfex', PSFExLoader(DES_PSFEx))

# Also make a builder to create the PSF object for a given position.
# The builders require 4 args.
# config is a dictionary that includes 'type' plus other items you might want to allow or require.
# base is the top level config dictionary where some global variables are stored.
# ignore is a list of key words that might be in the config dictionary that you should ignore.
def BuildDES_PSFEx(config, base, ignore, gsparams, logger):
    """Build a GSObject representing the PSFex model at the correct location in the image in a
    config-processing context.

    This is used as object type ``DES_PSFEx`` in a config file.

    It requires the use of the ``des_psfex`` input field.
    """
    des_psfex = GetInputObj('des_psfex', config, base, 'DES_PSFEx')

    opt = { 'flux' : float , 'num' : int, 'image_pos' : PositionD }
    params, safe = GetAllParams(config, base, opt=opt, ignore=ignore)

    if 'image_pos' in params:
        image_pos = params['image_pos']
    elif 'image_pos' in base:
        image_pos = base['image_pos']
    else:
        raise GalSimConfigError("DES_PSFEx requested, but no image_pos defined in base.")

    # Convert gsparams from a dict to an actual GSParams object
    if gsparams: gsparams = GSParams(**gsparams)
    else: gsparams = None

    #psf = des_psfex.getPSF(image_pos, gsparams=gsparams)
    # Because of serialization issues, the above call doesn't work.  So we need to
    # repeat the internals of getPSF here.
    # Also, this is why we have getSampleScale and getLocalWCS.  The multiprocessing.managers
    # stuff only makes available methods of classes that are proxied, not all the attributes.
    # So this is the only way to access these attributes.
    im = Image(des_psfex.getPSFArray(image_pos))
    psf =InterpolatedImage(im, scale=des_psfex.getSampleScale(), flux=1,
                           x_interpolant=Lanczos(3), gsparams=gsparams)
    psf = des_psfex.getLocalWCS(image_pos).toWorld(psf)

    if 'flux' in params:
        psf = psf.withFlux(params['flux'])

    # The second item here is "safe", a boolean that declares whether the returned value is
    # safe to save and use again for later objects.  In this case, we wouldn't want to do
    # that, since they will be at different positions, so the interpolated PSF will be different.
    return psf, False

# Register this builder with the config framework:
RegisterObjectType('DES_PSFEx', BuildDES_PSFEx, input_type='des_psfex')
