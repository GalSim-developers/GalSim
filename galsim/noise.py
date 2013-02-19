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
    noise.applyTo(image.view())

def addNoiseSNR(image, noise, snr, preserve_flux=False):
    """Adds CCDNoise to an image in a way that achieves the specified signal-to-noise ratio.
    
    >>> Image.addNoiseSNR(noise, snr, preserve_flux)  
                                                   
    Noise following the suppled model will be added to the image modifying either the flux of the
    object (if `preserve_flux=True`) or the variance of the noise (if `preserve_flux=False`) such
    that the given signal-to-noise ratio, `snr`, is reached.

    If `preserve_flux=False` (the default), the flux of the input image will be rescaled to achieve
    the desired signal-to-noise ratio (useful if adding noise separately to multiple galaxies with
    the same sky_level).
    
    If `preserve_flux=True`, then the variance of the noise model is modified.

    The definition of SNR is equivalent to the one used by Great08.  Taking a weighted integral 
    of the flux:
        S = sum W(x,y) I(x,y) / sum W(x,y)
        N^2 = Var(S) = sum W(x,y)^2 Var(I(x,y)) / (sum W(x,y))^2
    and assuming that Var(I(x,y)) is constant
        Var(I(x,y)) = noise_var
    We then assume that we are using a matched filter for W, so W(x,y) = I(x,y).  Then a few things 
    cancel and we find that
        snr = S/N = sqrt( sum I(x,y)^2 / noise_var )
    and therefore, for a given I(x,y) and snr,
        noise_var = sum I(x,y)^2/snr^2.

    Not that for noise models such as Poisson and CCDNoise, the constant Var(I(x,y)) assumption
    is only approximate, since the flux of the object adds to the Poisson noise in those pixels.
    Thus, the real S/N on the final image will be slightly lower than the target `snr` value, 
    and this effect will be larger for brighter objects.
    """
    import numpy
    noise_var = noise.getVariance()
    if preserve_flux:
        new_noise_var = numpy.sum(image.array**2)/snr/snr
        noise.setVariance(new_noise_var)
        image.addNoise(noise)
        noise.setVariance(noise_var)  # Revert to condition on input.
    else:
        sn_meas = numpy.sqrt( numpy.sum(image.array**2)/noise_var )
        flux = snr/sn_meas
        image *= flux
        image.addNoise(noise)

# inject addNoise and addNoiseSNR as methods of Image classes
for Class in _galsim.Image.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

for Class in _galsim.ImageView.itervalues():
    Class.addNoise = addNoise
    Class.addNoiseSNR = addNoiseSNR

del Class # cleanup public namespace
