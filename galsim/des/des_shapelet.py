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
"""@file des_shapelet.py

Part of the DES module.  This file implements one way that DES measures the PSF.

The DES_Shapelet class handles interpolated shapelet decompositions, which are generally
stored in *_fitpsf.fits files.
"""

import galsim

class DES_Shapelet(object):
    """Class that handles DES files describing interpolated shapelet decompositions.
    These are usually stored as *_fitpsf.fits files, although there is also an ASCII
    version stored as *_fitpsf.dat.

    Typical usage:
        
        des_shapelet = galsim.des.DES_Shapelet(fitpsf_file_name)
        
        ...

        pos = galsim.Position(image_x, image_y)  # position in pixels on the image
                                                 # NOT in arcsec on the sky!
        psf = des_shapelet.getPSF(pos)

    Note that the DES_Shapelet profile is measured with respect to sky coordinates, not 
    pixel coordinates.  So if you want the drawn image to look like the original, it should be
    drawn with the same WCS as found in the original image.  However, GalSim doesn't yet have
    the ability to handle such WCS functions.  This is Issue #364.  Until then, an approximate
    workaround is to use pixel_scale=0.262, and apply a rotation of -90 degrees before drawing.

    This class will only interpolate within the defining bounds.  It won't extrapolate
    beyond the bounding box of where the stars defined the interpolation.
    If you try to use it with an invalid position, it will throw an IndexError.
    You can check whether a position is valid with

        if des_shapelet.bounds.includes(pos):
            psf = des_shapelet.getPSF(pos)
        else:
            [...skip this object...]


    @param file_name  The file name to be read in.
    @param dir        Optionally a directory name can be provided if the file_name does not 
                      already include it.
    @param file_type  Either 'ASCII' or 'FITS' or None.  If None, infer from the file name ending
                      (default = None).
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'file_type' : str , 'dir' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, dir=None, file_type=None):
        if dir:
            import os
            file_name = os.path.join(dir,file_name)
        self.file_name = file_name

        if not file_type:
            if self.file_name.lower().endswith('.fits'):
                file_type = 'FITS'
            else:
                file_type = 'ASCII'
        file_type = file_type.upper()
        if file_type not in ['FITS', 'ASCII']:
            raise ValueError("file_type must be either FITS or ASCII if specified.")

        if file_type == 'FITS':
            self.read_fits()
        else:
            self.read_ascii()

    def read_ascii(self):
        """Read in a DES_Shapelet stored using the the ASCII-file version.
        """
        import numpy
        fin = open(self.file_name, 'r')
        lines = fin.readlines()
        temp = lines[0].split()
        self.psf_order = int(temp[0])
        self.psf_size = (self.psf_order+1) * (self.psf_order+2) / 2
        self.sigma = float(temp[1])
        self.fit_order = int(temp[2])
        self.fit_size = (self.fit_order+1) * (self.fit_order+2) / 2
        self.npca = int(temp[3])

        temp = lines[1].split()
        self.bounds = galsim.BoundsD(
            float(temp[0]), float(temp[1]),
            float(temp[2]), float(temp[3]))

        temp = lines[2].split()
        assert int(temp[0]) == self.psf_size
        self.ave_psf = numpy.array(temp[2:self.psf_size+2]).astype(float)
        assert self.ave_psf.shape == (self.psf_size,)

        temp = lines[3].split()
        assert int(temp[0]) == self.npca
        assert int(temp[1]) == self.psf_size
        self.rot_matrix = numpy.array(
            [ lines[4+k].split()[1:self.psf_size+1] for k in range(self.npca) ]
            ).astype(float)
        assert self.rot_matrix.shape == (self.npca, self.psf_size)

        temp = lines[5+self.npca].split()
        assert int(temp[0]) == self.fit_size
        assert int(temp[1]) == self.npca
        self.interp_matrix = numpy.array(
            [ lines[6+self.npca+k].split()[1:self.npca+1] for k in range(self.fit_size) ]
            ).astype(float)
        assert self.interp_matrix.shape == (self.fit_size, self.npca)

    def read_fits(self):
        """Read in a DES_Shapelet stored using the the FITS-file version.
        """
        import pyfits
        cat = pyfits.getdata(self.file_name,1)
        # These fields each only contain one element, hence the [0]'s.
        self.psf_order = cat.field('psf_order')[0]
        self.psf_size = (self.psf_order+1) * (self.psf_order+2) / 2
        self.sigma = cat.field('sigma')[0]
        self.fit_order = cat.field('fit_order')[0]
        self.fit_size = (self.fit_order+1) * (self.fit_order+2) / 2
        self.npca = cat.field('npca')[0]

        self.bounds = galsim.BoundsD(
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

    def getPSF(self, pos):
        """Returns the PSF at position pos

        @param pos   The position in pixel units for which to build the PSF.

        @returns a Shapelet instance.
        """
        if not self.bounds.includes(pos):
            raise IndexError("position in DES_Shapelet.getPSF is out of bounds")

        import numpy
        Px = self._definePxy(pos.x,self.bounds.xmin,self.bounds.xmax)
        Py = self._definePxy(pos.y,self.bounds.ymin,self.bounds.ymax)
        order = self.fit_order
        P = numpy.array([ Px[n-q] * Py[q] for n in range(order+1) for q in range(n+1) ])
        assert len(P) == self.fit_size

        # Note: This is equivalent to:
        #
        #     P = numpy.empty(self.fit_size)
        #     k = 0
        #     for n in range(self.fit_order+1):
        #         for q in range(n+1):
        #             P[k] = Px[n-q] * Py[q]
        #             k = k+1

        b1 = numpy.dot(P,self.interp_matrix)
        b = numpy.dot(b1,self.rot_matrix)
        assert len(b) == self.psf_size
        b += self.ave_psf
        ret = galsim.Shapelet(self.sigma, self.psf_order, b)
        return ret

    def _definePxy(self, x, min, max):
        import numpy
        x1 = (2.*x-min-max)/(max-min)
        temp = numpy.empty(self.fit_order+1)
        temp[0] = 1
        if self.fit_order > 0:
            temp[1] = x1
        for i in range(2,self.fit_order+1):
            temp[i] = ((2.*i-1.)*x1*temp[i-1] - (i-1.)*temp[i-2]) / float(i)
        return temp

# Now add this class to the config framework.
import galsim.config

# First we need to add the class itself as a valid input_type.
galsim.config.process.valid_input_types['des_shapelet'] = ('galsim.des.DES_Shapelet', [], False)

# Also make a builder to create the PSF object for a given position.
# The builders require 4 args.
# config is a dictionary that includes 'type' plus other items you might want to allow or require.
# key is the key name one level up in the config structure.  Probably 'psf' in this case.
# base is the top level config dictionary where some global variables are stored.
# ignore is a list of key words that might be in the config dictionary that you should ignore.
def BuildDES_Shapelet(config, key, base, ignore):
    """@brief Build a RealGalaxy type GSObject from user input.
    """
    opt = { 'flux' : float }
    kwargs, safe = galsim.config.GetAllParams(config, key, base, opt=opt, ignore=ignore)

    if 'des_shapelet' not in base:
        raise ValueError("No DES_Shapelet instance available for building type = DES_Shapelet")
    des_shapelet = base['des_shapelet']

    if 'chip_pos' not in base:
        raise ValueError("DES_Shapelet requested, but no chip_pos defined in base.")
    chip_pos = base['chip_pos']

    if des_shapelet.bounds.includes(chip_pos):
        psf = des_shapelet.getPSF(chip_pos)
    else:
        message = 'Position '+str(chip_pos)+' not in interpolation bounds: '
        message += str(des_shapelet.bounds)
        raise galsim.config.gsobject.SkipThisObject(message)

    if 'flux' in kwargs:
        psf.setFlux(kwargs['flux'])

    # The second item here is "safe", a boolean that declares whether the returned value is 
    # safe to save and use again for later objects.  In this case, we wouldn't want to do 
    # that, since they will be at different positions, so the interpolated PSF will be different.
    return psf, False

# Register this builder with the config framework:
galsim.config.gsobject.valid_gsobject_types['DES_Shapelet'] = 'galsim.des.BuildDES_Shapelet'

