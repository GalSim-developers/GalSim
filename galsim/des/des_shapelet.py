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
"""@file des_psf.py

Part of the DES module.  This file implements two ways that DES measures the PSF.

The DES_Shapelet class handles interpolated shapelet decompositions, which are generally
stored in *_fitpsf.fits files.

The DES_PsfEx class handles interpolated PCA images, which are generally stored in 
*_psfcat.psf files.
"""

import galsim

class DES_Shapelet(object):
    """Class that handles DES files describing interpolated shapelet decompositions.

    
    @param file_name  The file name to be read in.
    @param file_type  Either 'ASCII' or 'FITS' or None.  If None, infer from the file name ending
                      (default = None).
    """
    _req_params = { 'file_name' : str }
    _opt_params = { 'file_type' : str }
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, file_type=None):
        import os
        self.file_name = file_name.strip()

        if not file_type:
            if self.file_name.lower().endswith('.fits'):
                file_type = 'FITS'
            else:
                file_type = 'ASCII'
        if file_type.upper() not in ['FITS', 'ASCII']:
            raise ValueError("file_type must be either FITS or ASCII if specified.")

        try:
            if file_type.upper() == 'FITS':
                import pyfits
                cat = pyfits.getdata(self.file_name)
                # These fields each only contain one element, hence the [0]'s.
                self.psf_order = cat.field('psf_order')[0]
                self.psf_size = (self.psf_order+1) * (self.psf_order+2) / 2
                self.sigma = cat.field('sigma')[0]
                self.fit_order = cat.field('fit_order')[0]
                self.fit_size = (self.fit_order+1) * (self.fit_order+2) / 2
                self.npca = cat.field('npca')[0]

                self.xmin = cat.field('xmin')[0]
                self.xmax = cat.field('xmax')[0]
                self.ymin = cat.field('ymin')[0]
                self.ymax = cat.field('ymax')[0]

                self.ave_psf = cat.field('ave_psf')[0]
                assert self.ave_psf.shape == (self.psf_size,)

                self.rot_matrix = cat.field('rot_matrix')[0].T
                assert self.rot_matrix.shape == (self.npca, self.psf_size)

                self.interp_matrix = cat.field('interp_matrix')[0].T
                assert self.interp_matrix.shape == (self.fit_size, self.npca)

            else:
                import numpy
                fin = open(self.file_name, 'r')
                lines = fin.readlines()
                temp = lines[0].split()
                self.psf_order = float(temp[0])
                self.psf_size = (self.psf_order+1) * (self.psf_order+2) / 2
                self.sigma = float(temp[1])
                self.fit_order = float(temp[2])
                self.fit_size = (self.fit_order+1) * (self.fit_order+2) / 2
                self.npca = float(temp[3])

                temp = lines[1].split()
                self.xmin = float(temp[0])
                self.xmax = float(temp[1])
                self.ymin = float(temp[2])
                self.ymax = float(temp[3])

                temp = lines[2].split()
                assert int(temp[0]) == self.psf_size
                self.ave_psf = numpy.array(temp[2:self.psf_size+2])
                assert self.ave_psf.shape == (self.psf_size,)

                temp = lines[3].split()
                assert int(temp[0]) == self.npca
                assert int(temp[1]) == self.psf_size
                self.rot_matrix = numpy.array(
                    [ lines[4+k].split()[1:self.psf_size+1] for k in range(self.npca) ] )
                assert self.rot_matrix.shape == (self.npca, self.psf_size)

                temp = lines[5+self.npca].split()
                assert int(temp[0]) == self.fit_size
                assert int(temp[1]) == self.npca
                self.rot_matrix = numpy.array(
                    [ lines[6+self.npca+k].split()[1:self.npca+1] for k in range(self.fit_size) ] )
                assert self.interp_matrix.shape == (self.fit_size, self.npca)

        except Exception, e:
            print e
            raise RuntimeError("Unable to read %s DES_Shapelet file %s."%(
                    file_type,self.file_name))

    def getPSF(self, pos):
        """Returns the PSF at position pos

        This returns a Shapelet instance.
        """
        if (pos.x < self.xmin or pos.x > self.xmax or
            pos.y < self.ymin or pos.y > self.ymax):
            raise ValueError("position in DES_Shapelet.getPSF is out of bounds")

        import numpy
        Px = self.definePxy(pos.x,self.xmin,self.xmax)
        Py = self.definePxy(pos.y,self.ymin,self.ymax)
        P = numpy.zeros(self.fit_size)
        i = 0
        for n in range(self.fit_order+1):
            for q in range(n):
                P[i] = Px[n-q] * Py[q]
                i = i+1

        b1 = numpy.dot(P,self.interp_matrix)
        b = numpy.dot(b1,self.rot_matrix)
        assert len(b) == self.psf_size
        b += self.ave_psf
        return galsim.Shapelet(self.sigma, self.psf_order, b)

    def definePxy(self, x, min, max):
        import numpy
        x1 = (2.*x-min-max)/(max-min)
        temp = numpy.ones(self.fit_order+1)
        if self.fit_order > 0:
            temp[1] = x1
        for i in range(2,self.fit_order):
            temp[i] = ((2.*i-1.)*x1*temp[i-1] - (i-1.)*temp[i-2]) / float(i)
        return temp

