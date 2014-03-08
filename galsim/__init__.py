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

# First some basic building blocks that don't usually depend on anything else
from position import PositionI, PositionD
from bounds import BoundsI, BoundsD
from shear import Shear
from angle import Angle, AngleUnit, radians, hours, degrees, arcmin, arcsec, HMS_Angle, DMS_Angle
from catalog import Catalog, Dict
from table import LookupTable

# Image 
from image import Image, ImageS, ImageI, ImageF, ImageD
# These are obsolete, but we currently still suppert them.  They will be deprecated and 
# removed eventually.
from image import ImageView, ImageViewS, ImageViewI, ImageViewF, ImageViewD
from image import ConstImageView, ConstImageViewS, ConstImageViewI, ConstImageViewF, ConstImageViewD

# Noise
from random import BaseDeviate, UniformDeviate, GaussianDeviate, PoissonDeviate, DistDeviate
from random import BinomialDeviate, Chi2Deviate, GammaDeviate, WeibullDeviate
from noise import BaseNoise, GaussianNoise, PoissonNoise, CCDNoise
from noise import DeviateNoise, VariableGaussianNoise
from correlatednoise import CorrelatedNoise, getCOSMOSNoise, UncorrelatedNoise

# GSObject
from base import GSParams, GSObject, Gaussian, Moffat, Airy, Kolmogorov, Pixel, Box
from base import Exponential, Sersic, DeVaucouleurs
from real import RealGalaxy, RealGalaxyCatalog, simReal
from optics import OpticalPSF
from shapelet import Shapelet, ShapeletSize, FitShapelet, LVectorSize
from interpolatedimage import Interpolant, Interpolant2d, InterpolantXY
from interpolatedimage import Nearest, Linear, Cubic, Quintic, Lanczos, SincInterpolant, Delta
from interpolatedimage import InterpolatedImage
from compound import Add, Convolve, Deconvolve, AutoConvolve, AutoCorrelate

# WCS
from fits import FitsHeader
from celestial import CelestialCoord
from wcs import BaseWCS, PixelScale, ShearWCS, JacobianWCS
from wcs import OffsetWCS, OffsetShearWCS, AffineTransform, UVFunction, RaDecFunction
from fitswcs import AstropyWCS, PyAstWCS, WcsToolsWCS, GSFitsWCS, FitsWCS, TanWCS

# Lensing stuff
from lensing_ps import PowerSpectrum
from nfw_halo import NFWHalo, Cosmology

# Packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import pse
from . import hsm
from . import deprecated
