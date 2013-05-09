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


# Import things from other files we want to be in the galsim namespace
from ._galsim import *
from base import *
from real import RealGalaxy, RealGalaxyCatalog, simReal
from optics import OpticalPSF
from shapelet import Shapelet
from interpolatedimage import InterpolatedImage
from compound import Add, Convolve, Deconvolve, AutoConvolve, AutoCorrelate

from shear import Shear
from lensing_ps import PowerSpectrum
from nfw_halo import NFWHalo, Cosmology
from catalog import InputCatalog
from table import LookupTable
from random import DistDeviate
from correlatednoise import CorrelatedNoise, getCOSMOSNoise
from fits import FitsHeader

# packages with docs and such, so nothing really to import by name.
from . import position
from . import bounds
from . import angle
from . import noise
from . import image
from . import random

# packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import des
from . import pse
from . import hsm
from . import deprecated
