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
Module which adds the addNoise method to the galsim.Image classes at the Python layer.
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

# inject addNoise as a method of Image classes
for Class in _galsim.Image.itervalues():
    Class.addNoise = addNoise

for Class in _galsim.ImageView.itervalues():
    Class.addNoise = addNoise

del Class # cleanup public namespace
