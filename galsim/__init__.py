# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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



Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
import re

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
# We don't use setup.py, so it's not so important to do it this way, but if we ever switch...
# And it does make it a bit easier to get the version number in SCons too.
from ._version import __version__
vi = re.split(r'\.|-',__version__)
__version_info__ = tuple([int(x) for x in vi if x.isdigit()])

# Define the current code version, in addition to the hidden attribute, to be consistent with
# previous GalSim versions that indicated the version number in this way.
version = __version__

# Import things from other files we want to be in the galsim namespace

# First some basic building blocks that don't usually depend on anything else
from .position import Position, PositionI, PositionD
from .bounds import Bounds, BoundsI, BoundsD, _BoundsI, _BoundsD
from .shear import Shear, _Shear
from .angle import Angle, AngleUnit, _Angle, radians, hours, degrees, arcmin, arcsec
from .catalog import Catalog, Dict, OutputCatalog
from .scene import COSMOSCatalog
from .table import LookupTable, LookupTable2D

# Exception and Warning classes
from .errors import GalSimError, GalSimRangeError, GalSimValueError
from .errors import GalSimKeyError, GalSimIndexError, GalSimNotImplementedError
from .errors import GalSimBoundsError, GalSimUndefinedBoundsError, GalSimImmutableError
from .errors import GalSimIncompatibleValuesError, GalSimSEDError, GalSimHSMError
from .errors import GalSimFFTSizeError
from .errors import GalSimConfigError, GalSimConfigValueError
from .errors import GalSimWarning, GalSimDeprecationWarning

# Image
from .image import Image, ImageS, ImageI, ImageF, ImageD, ImageCF, ImageCD, ImageUS, ImageUI, _Image

# PhotonArray
from .photon_array import PhotonArray, WavelengthSampler, FRatioAngles, PhotonDCR

# Noise
from .random import BaseDeviate, UniformDeviate, GaussianDeviate, PoissonDeviate, DistDeviate
from .random import BinomialDeviate, Chi2Deviate, GammaDeviate, WeibullDeviate
from .noise import BaseNoise, GaussianNoise, PoissonNoise, CCDNoise
from .noise import DeviateNoise, VariableGaussianNoise
from .correlatednoise import CorrelatedNoise, getCOSMOSNoise, UncorrelatedNoise, CovarianceSpectrum

# GSObject
from .gsobject import GSObject
from .gsparams import GSParams
from .gaussian import Gaussian
from .moffat import Moffat
from .airy import Airy
from .kolmogorov import Kolmogorov
from .box import Pixel, Box, TopHat
from .exponential import Exponential
from .sersic import Sersic, DeVaucouleurs
from .spergel import Spergel
from .deltafunction import DeltaFunction
from .real import RealGalaxy, RealGalaxyCatalog, ChromaticRealGalaxy
from .phase_psf import Aperture, PhaseScreenList, PhaseScreenPSF, OpticalPSF
from .phase_screens import AtmosphericScreen, Atmosphere, OpticalScreen
from .shapelet import Shapelet
from .inclined import InclinedExponential, InclinedSersic
from .interpolant import Interpolant
from .interpolant import Nearest, Linear, Cubic, Quintic, Lanczos, SincInterpolant, Delta
from .interpolatedimage import InterpolatedImage, _InterpolatedImage
from .interpolatedimage import InterpolatedKImage, _InterpolatedKImage
from .sum import Add, Sum
from .convolve import Convolve, Convolution, Deconvolve, Deconvolution
from .convolve import AutoConvolve, AutoConvolution, AutoCorrelate, AutoCorrelation
from .fouriersqrt import FourierSqrt, FourierSqrtProfile
from .randwalk import RandomWalk
from .transform import Transform, Transformation, _Transform
from .vonkarman import VonKarman
from .second_kick import SecondKick

# Chromatic
from .chromatic import ChromaticObject, ChromaticAtmosphere, ChromaticSum
from .chromatic import ChromaticConvolution, ChromaticDeconvolution, ChromaticAutoConvolution
from .chromatic import ChromaticAutoCorrelation, ChromaticTransformation
from .chromatic import ChromaticFourierSqrtProfile
from .chromatic import ChromaticOpticalPSF, ChromaticAiry, InterpolatedChromaticObject
from .sed import SED
from .bandpass import Bandpass

# WCS
from .fits import FitsHeader
from .celestial import CelestialCoord
from .wcs import BaseWCS, PixelScale, ShearWCS, JacobianWCS
from .wcs import OffsetWCS, OffsetShearWCS, AffineTransform, UVFunction, RaDecFunction
from .fitswcs import AstropyWCS, PyAstWCS, WcsToolsWCS, GSFitsWCS, FitsWCS, TanWCS

# Lensing stuff
from .lensing_ps import PowerSpectrum
from .nfw_halo import NFWHalo, Cosmology

# Detector effects
from .sensor import Sensor, SiliconSensor
from . import detectors  # Everything here is a method of Image, so nothing to import by name.

# Deprecated functionality
from . import deprecated

# Packages we intentionally keep separate.  E.g. requires galsim.fits.read(...)
from . import fits
from . import config
from . import integ
from . import bessel
from . import pse
from . import hsm
from . import dcr
from . import meta_data
from . import cdmodel
from . import utilities
from . import fft
from . import download_cosmos
from . import zernike
