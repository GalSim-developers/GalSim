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
"""@file noise.py
Module which adds the addNoise and addNoiseSNR methods to the galsim.Image classes at the Python
layer.
"""

from . import _galsim

def addNoise(image, noise):
    """Noise addition Image method, adding noise according to a supplied noise model.

    >>> Image.addNoise(noise)

    Noise following supplied model will be added to the image.

    @param  image  The image on which to add the noise.
    @param  noise  Instantiated noise model (currently CCDNoise, UniformDeviate, BinomialDeviate,
                   GaussianDeviate and PoissonDeviate are supported).

    If the supplied noise model object does not have an applyTo() method, then this will raise an
    AttributeError exception.
    """
    im_view = image.view()
    noise.applyTo(im_view)

def addNoiseSNR(self, snr, sky_level=None, rng=_galsim.UniformDeviate()):
    """Adds CCDNoise to an image in a way that achieves the specified signal-to-noise ratio.
    
    Possible ways to call addNoiseSNR:
    
    >>> Image.addNoiseSNR(snr)                     # Add noise so that the image has a 
                                                   # signal-to-noise ratio snr
                                                   
    >>> Image.addNoiseSNR(snr,sky_level=sky_level) # Use the given sky_level and rescale the image
                                                   # flux to get the given SNR
                                                   
    >>> Image.addNoiseSNR(snr,rng=rng)             # Add noise using the same underlying RNG as rng
    
    sky_level should be the sky_level PER PIXEL--if you have a sky_level per square arcsec, for 
    example, the value you pass should be multiplied by (pixel scale/1 arcsec) squared.
    
    If sky_level is passed to addNoiseSNR, the flux of the input image will be rescaled to achieve
    the desired signal-to-noise ratio (useful if adding noise separately to multiple galaxies with 
    the same sky_level).  If it is not, then the sky_level is chosen based on the given flux and 
    SNR in a Great08-like way.  Taking a weighted integral of the flux:
        S = sum W(x,y) I(x,y) / sum W(x,y)
        N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
    and assuming that Var(I(x,y)) is dominated by the sky noise:
        Var(I(x,y)) = sky_level
    We then assume that we are using a matched filter for W, so W(x,y) = I(x,y).  Then a few things 
    cancel and we find that
        S/N = sqrt( sum I(x,y)^2 / sky_level )
    and therefore, for a given I(x,y) and S/N,
        sky_level = sum I(x,y)^2/(S/N)^2.
    """
    import numpy
    if not isinstance(rng,(_galsim.BaseDeviate,int,long)):
        raise TypeError ("Rng %s passed to AddNoiseSNR cannot be used to initialize a "
                         "BaseDeviate!"%rng)
    if sky_level is None:
        sky_level=numpy.sum(self.array**2)/snr/snr
    else:
        sn_meas=numpy.sqrt( numpy.sum(self.array**2)/sky_level )
        flux=snr/sn_meas
        self*=flux
    self+=sky_level
    self.addNoise(_galsim.CCDNoise(rng))
    self-=sky_level

# inject addNoise and addNoiseSNR as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

for Class in _galsim.ImageView.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

del Class # cleanup public namespace
