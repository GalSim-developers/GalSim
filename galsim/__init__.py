# Copyright 2012-2014 The GalSim developers:
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

# Two options for pyfits module:
try:
    import astropy.io.fits as pyfits
    pyfits_version = '4.0'  # astropy.io.fits doesn't define a __version__ attribute,
                            # so mock it up as 4.0.  We might need to revisit this if
                            # we need to start discriminating on different astropy
                            # versions.
except:
    import pyfits
    pyfits_version = pyfits.__version__

# Import things from other files we want to be in the galsim namespace
from ._galsim import *
from image import Image, ImageS, ImageI, ImageF, ImageD
from base import version, GSObject, Gaussian, Moffat, Airy, Kolmogorov, Pixel, Box, Sersic
from base import Exponential, DeVaucouleurs
from chromatic import ChromaticObject, ChromaticAtmosphere, Chromatic, ChromaticSum
from chromatic import ChromaticConvolution, ChromaticDeconvolution, ChromaticAutoConvolution
from chromatic import ChromaticAutoCorrelation
from sed import SED
from bandpass import Bandpass
from real import RealGalaxy, RealGalaxyCatalog, simReal
from optics import OpticalPSF
from shapelet import Shapelet
from interpolatedimage import InterpolatedImage
from compound import Add, Sum, Convolve, Convolution, Deconvolve, Deconvolution, AutoConvolve
from compound import AutoConvolution, AutoCorrelate, AutoCorrelation
from shear import Shear
from wcs import BaseWCS, PixelScale, ShearWCS, JacobianWCS
from wcs import OffsetWCS, OffsetShearWCS, AffineTransform, UVFunction, RaDecFunction
from fitswcs import AstropyWCS, PyAstWCS, WcsToolsWCS, GSFitsWCS, FitsWCS, TanWCS
from lensing_ps import PowerSpectrum
from nfw_halo import NFWHalo, Cosmology
from catalog import Catalog, Dict
from table import LookupTable
from random import DistDeviate
from noise import VariableGaussianNoise
from correlatednoise import CorrelatedNoise, getCOSMOSNoise, UncorrelatedNoise
from fits import FitsHeader
from angle import HMS_Angle, DMS_Angle
from celestial import CelestialCoord

# packages with docs and such, so nothing really to import by name.
from . import position
from . import bounds
from . import random

# packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import bessel
from . import pse
from . import hsm
from . import deprecated
from . import dcr
