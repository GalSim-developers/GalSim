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
"""@file des_psfex.py

Part of the DES module.  This file implements one way that DES measures the PSF.

The DES_PSFEx class handles interpolated PCA images, which are generally stored in *_psfcat.psf 
files.

See documentation here:

    https://www.astromatic.net/pubsvn/software/psfex/trunk/doc/psfex.pdf
"""

import galsim

class DES_PSFEx(object):
    """Class that handles DES files describing interpolated principal component images
    of the PSF.  These are usually stored as *_psfcat.psf files.

    Typical usage:
        
        des_psfex = galsim.des.DES_PSFEx(fitpsf_file_name)
        
        ...

        pos = galsim.PositionD(image_x, image_y)  # position in pixels on the image
                                                  # NOT in arcsec on the sky!
        psf = des_psfex.getPSF(pos, pixel_scale=0.27)


    @param file_name  The file name to be read in.
    @param dir        Optionally a directory name can be provided if the file_name does not 
                      already include it.
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
        self.read()

    def read(self):
        import pyfits
        hdu = pyfits.open(self.file_name)[1]
        # Number of parameters used for the interpolation.  We require this to be 2.
        pol_naxis = hdu.header['POLNAXIS']
        if pol_naxis != 2:
            raise IOError("PSFEx: Expected POLNAXIS == 2, got %d"%pol_naxis)

        # These are the names of the two axes.  Should be X_IMAGE, Y_IMAGE.
        # If they aren't, then the way we use the interpolation will be wrong.
        pol_name1 = hdu.header['POLNAME1']
        if pol_name1 != 'X_IMAGE':
            raise IOError("PSFEx: Expected POLNAME1 == X_IMAGE, got %s"%pol_name1)
        pol_name2 = hdu.header['POLNAME2']
        if pol_name2 != 'Y_IMAGE':
            raise IOError("PSFEx: Expected POLNAME2 == Y_IMAGE, got %s"%pol_name2)

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
        if pol_ngrp != 1:
            raise IOError("PSFEx: Current implementation requires POLNGRP == 1, got %d"%pol_ngrp)

        # Which group each item is in.  We require group 1.
        pol_group1 = hdu.header['POLGRP1']
        if pol_group1 != 1:
            raise IOError("PSFEx: Expected POLGRP1 == 1, got %s"%pol_group1)
        pol_group2 = hdu.header['POLGRP2']
        if pol_group2 != 1:
            raise IOError("PSFEx: Expected POLGRP2 == 1, got %s"%pol_group2)

        # The degree of the polynomial.  E.g. POLDEG1 = 2 means the values will be:
        #     1, x, x^2, y, xy, y^2
        # If we start allowing POLNGRP > 1, there is a separate degree for each group.
        pol_deg = hdu.header['POLDEG1']

        # The number of axes in the basis object.  We require this to be 3.
        psf_naxis = hdu.header['PSFNAXIS']
        if psf_naxis != 3:
            raise IOError("PSFEx: Expected PSFNAXIS == 3, got %d"%psfnaxis)

        # The first two axes are the image size of the PSF postage stamp.
        psf_axis1 = hdu.header['PSFAXIS1']
        psf_axis2 = hdu.header['PSFAXIS2']

        # The third axis is the direction of the polynomial interpolation.  So it should
        # be equal to (d+1)(d+2)/2.
        psf_axis3 = hdu.header['PSFAXIS3']
        if psf_axis3 != ((pol_deg+1)*(pol_deg+2))/2:
            raise IOError("PSFEx: POLDEG and PSFAXIS3 disagree")

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
        # Note: older pyfits versions don't get the shape right.
        # For newer pyfits versions the reshape command should be a no op.
        basis = hdu.data.field('PSF_MASK')[0].reshape(psf_axis3,psf_axis2,psf_axis1)
        # Make sure this turned out right.
        if basis.shape[0] != psf_axis3:
            raise IOError("PSFEx: PSFAXIS3 disagrees with actual basis size")
        if basis.shape[1] != psf_axis2:
            raise IOError("PSFEx: PSFAXIS2 disagrees with actual basis size")
        if basis.shape[2] != psf_axis1:
            raise IOError("PSFEx: PSFAXIS1 disagrees with actual basis size")

        # Save some of these values for use in building the interpolated images
        self.basis = basis
        self.fit_order = pol_deg
        self.fit_size = psf_axis3
        self.x_zero = pol_zero1
        self.y_zero = pol_zero2
        self.x_scale = pol_scal1
        self.y_scale = pol_scal2
        self.sample_scale = psf_samp


    def getPSF(self, pos, pixel_scale, gsparams=None):
        """Returns the PSF at position pos

        The PSFEx class does everything in pixel units, so it has no concept of the pixel_scale.
        For Galsim, we do everything in physical units (i.e. arcsec typically), so the returned 
        psf needs to account for the pixel_scale.

        @param pos          The position in pixel units for which to build the PSF.
        @param pixel_scale  The pixel scale in arcsec/pixel.
        @param gsparams     (Optional) A GSParams instance to pass to the constructed GSObject.

        @returns an InterpolatedImage instance.
        """
        import numpy
        xto = self._define_xto( (pos.x - self.x_zero) / self.x_scale )
        yto = self._define_xto( (pos.y - self.y_zero) / self.y_scale )
        order = self.fit_order
        P = numpy.array([ xto[nx] * yto[ny] for ny in range(order+1) for nx in range(order+1-ny) ])
        assert len(P) == self.fit_size
        ar = numpy.tensordot(P,self.basis,(0,0)).astype(numpy.float32)

        # Note: This is equivalent to:
        #   ar = self.basis[0].astype(numpy.float32)
        #   for n in range(1,self.fit_order+1):
        #       for ny in range(n+1):
        #           nx = n-ny
        #           k = nx+ny*(self.fit_order+1)-ny*(ny-1)/2
        #           ar += xto[nx] * yto[ny] * self.basis[k]
        # which is pretty much Peter's version of this code.

        im = galsim.ImageViewF(array=ar)
        # We need the scale in arcsec/psfex_pixel, which is 
        #    (arcsec / image_pixel) * (image_pixel / psfex_pixel)
        #    = pixel_scale * sample_scale
        im.scale = pixel_scale * self.sample_scale
        return galsim.InterpolatedImage(im, flux=1, x_interpolant=galsim.Lanczos(3),
                                        gsparams=gsparams)

    def _define_xto(self, x):
        import numpy
        xto = numpy.empty(self.fit_order+1)
        xto[0] = 1
        for i in range(1,self.fit_order+1):
            xto[i] = x*xto[i-1]
        return xto

# Now add this class to the config framework.
import galsim.config

# First we need to add the class itself as a valid input_type.
galsim.config.process.valid_input_types['des_psfex'] = ('galsim.des.DES_PSFEx', [], False)

# Also make a builder to create the PSF object for a given position.
# The builders require 4 args.
# config is a dictionary that includes 'type' plus other items you might want to allow or require.
# key is the key name one level up in the config structure.  Probably 'psf' in this case.
# base is the top level config dictionary where some global variables are stored.
# ignore is a list of key words that might be in the config dictionary that you should ignore.
def BuildDES_PSFEx(config, key, base, ignore, gsparams):
    """@brief Build a RealGalaxy type GSObject from user input.
    """
    opt = { 'flux' : float }
    kwargs, safe = galsim.config.GetAllParams(config, key, base, opt=opt, ignore=ignore)

    if 'des_psfex' not in base:
        raise ValueError("No DES_PSFEx instance available for building type = DES_PSFEx")
    des_psfex = base['des_psfex']

    if 'image_pos' not in base:
        raise ValueError("DES_PSFEx requested, but no image_pos defined in base.")
    image_pos = base['image_pos']

    if 'pixel_scale' not in base:
        raise ValueError("DES_PSFEx requested, but no pixel_scale defined in base.")
    pixel_scale = base['pixel_scale']

    # Convert gsparams from a dict to an actual GSParams object
    if gsparams: gsparams = galsim.GSParams(**gsparams)
    else: gsparams = None

    psf = des_psfex.getPSF(image_pos, pixel_scale, gsparams=gsparams)

    if 'flux' in kwargs:
        psf.setFlux(kwargs['flux'])

    # The second item here is "safe", a boolean that declares whether the returned value is 
    # safe to save and use again for later objects.  In this case, we wouldn't want to do 
    # that, since they will be at different positions, so the interpolated PSF will be different.
    return psf, False


# Register this builder with the config framework:
galsim.config.gsobject.valid_gsobject_types['DES_PSFEx'] = 'galsim.des.BuildDES_PSFEx'

