# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""
GalSim: The modular galaxy image simulation toolkit

GalSim is open-source software for simulating images of astronomical objects
(stars, galaxies) in a variety of ways.  The bulk of the calculations are
carried out in C++, and the user interface is in python.  In addition, the code
can operate directly on "config" files, for those users who prefer not to work
in python.  The impetus for the software package was a weak lensing community
data challenge, called GREAT3:

    http://great3challenge.info/

However, the code has numerous additional capabilities beyond those needed for
the challenge, and has been useful for a number of projects that needed to
simulate high-fidelity galaxy images with accurate sizes and shears.

For an overview of GalSim workflow and python tools, please see the file
`doc/GalSim_Quick_Reference.pdf` in the GalSim repository.  A guide to using
the configuration files to generate simulations, a FAQ for installation issues,
a link to Doxygen-generated documentation, and other useful references can be
found on the GalSim wiki,

    https://github.com/GalSim-developers/GalSim/wiki

If you experience any issues with the software, or if you would like to file
a feature request, please do so on the main github site for GalSim,

    https://github.com/GalSim-developers/GalSim

Finally, if you have questions about how to do something in GalSim, you may
ask a question on StackOverflow,

    http://stackoverflow.com/

Use the galsim tag to flag it as a question about GalSim.



Copyright (c) 2012-2014 by the GalSim developers team on GitHub
https://github.com/GalSim-developers

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

This software is made available to you on an ``as is'' basis with no
representations or warranties, express or implied, including but not
limited to any warranty of performance, merchantability, fitness for a
particular purpose, commercial utility, non-infringement or title.
Neither the authors nor the organizations providing the support under
which the work was developed will be liable to you or any third party
with respect to any claim arising from your further development of the
software or any products related to or derived from the software, or for
lost profits, business interruption, or indirect special or consequential
damages of any kind.
"""

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

# Define the current code version
version = '1.2'

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
# These are obsolete, but we currently still support them.  They will be deprecated and 
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
from compound import Add, Sum, Convolve, Convolution, Deconvolve, Deconvolution
from compound import AutoConvolve, AutoConvolution, AutoCorrelate, AutoCorrelation

# Chromatic
from chromatic import ChromaticObject, ChromaticAtmosphere, Chromatic, ChromaticSum
from chromatic import ChromaticConvolution, ChromaticDeconvolution, ChromaticAutoConvolution
from chromatic import ChromaticAutoCorrelation
from sed import SED
from bandpass import Bandpass

# WCS
from fits import FitsHeader
from celestial import CelestialCoord
from wcs import BaseWCS, PixelScale, ShearWCS, JacobianWCS
from wcs import OffsetWCS, OffsetShearWCS, AffineTransform, UVFunction, RaDecFunction
from fitswcs import AstropyWCS, PyAstWCS, WcsToolsWCS, GSFitsWCS, FitsWCS, TanWCS

# Lensing stuff
from lensing_ps import PowerSpectrum
from nfw_halo import NFWHalo, Cosmology

# Detector effects
from . import detectors

# Packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import bessel
from . import pse
from . import hsm
from . import deprecated
from . import dcr
from . import meta_data
from . import cdmodel
