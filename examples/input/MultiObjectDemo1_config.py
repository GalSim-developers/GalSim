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

import logging

# USERS NOTE - THIS IS CURRENTLY IN DEVELOPMENT AND UNFINISHED!
#

# ---------------------------------------------------
# Configuration file for the MultiObjectDemo Script 1
# ---------------------------------------------------


# --- pixel scale ---
#
# Image sampling rate / separation between pixels. Sets the physical units for all lengths in the
# output images.  If including pixel integration, it probably makes sense for 
# config.PSF.Pixel.xw and config.PSF.Pixel.yw (defaults = 1.) to be set to config.dx.
config.dx = 1.


# --- PSF type ---
#
# Specify the PSF type(s) for the simulated galaxy images; placing a None object as the first entry
# tells GalSim to expect this type information to come from the input catalogue.
#
# The currently available PSF_types are:
#
# "Gaussian"         A Gaussian light distribution.
# "DoubleGaussian"   A sum of two Gaussians found to provide a reasonable empirical fit to ground-
#                    based PSFs.
# "Moffat"           A Moffat (1969) parametric PSF model.
# "Airy"             An Airy function describing ideal diffraction limited optics.
# "OpticalPSF"       A PSF describing telescope optics with coma, astigmatism, defocus and spherical#                    aberration.
# "Pixel"            A square boxcar convolution that describes pixelation / pixel flux integration.
#
# Each of these PSF types has it's own set of input parameters, which must be set for each output
# object via input catalogue values, or given a specified distribution via config file.  See the
# GalSim object docstrings / GalSim documentation for details.
#
# More than one PSF can be specified via adding multiple entries to the list below; the resulting
# PSF in the output image will be the *convolution* of all entries in the list.
#
config.PSF.type = ["Moffat", "Pixel"]
#
# Distributions for PSF parameters (including constant values) may be set below.  These will be 
# overwritten by any values for these parameters found in the input catalogues.
#
# e.g.:
#
# config.PSF.Moffat.beta = 3.   # Note that single scalar values are interpreted as constants
# config.PSF.Moffat.e1.distribution = "GaussianDeviate"
# config.PSF.Moffat.e1.mean = 0.
# config.PSF.Moffat.e1.sigma = 0.03
#
# The attribute of config.PSF (i.e. Moffat in the example above) must match one of the type
# strings in the list above, or it will be ignored when parsing the inputs.
#
# Any parameters not set either in configuration files or input catalogues will adopt their
# keyword default values, and GalSim will raise a Warning.
config.PSF.Moffat.flux = 1.
config.PSF.Moffat.beta = 3.
config.PSF.Moffat.g1 = -0.019
config.PSF.Moffat.g2 = -0.007
config.PSF.Moffat.fwhm = 2.85
config.PSF.Moffat.truncation_fwhm = 1.e6   # ~No truncation

config.PSF.Pixel.flux = 1.
config.PSF.Pixel.xw = config.dx
config.PSF.Pixel.yw = config.dx


# --- background ---
#
# A background level to be added to each postage stamp image before noise addition,
# to mimic a sky background.
#
config.sky.value = 1.e4
#
# Set the following parameter = True to subtract the background after noise addition.
#
config.sky.postsubtract = False


# --- galaxy type ---
#
# Specify the galaxy type(s) for the simulated galaxy images; placing a None object as the first
# entry tells GalSim to expect this type information to come from the input catalogue.
#
# The currently available galaxy types are:
#
# "Gaussian"        A Gaussian light distribution.
# "Sersic"          A Sersic profile.
# "Exponential"     Special case of a Sersic profile with index n = 1.
# "DeVaucouleurs"   Special case of a Sersic profile with index n = 4.
#
# Each of these galaxy types has it's own set of input parameters, which must be set for each output
# object via input catalogue values, or given a specified distribution via config file.  See the
# GalSim object docstrings / GalSim documentation for details.
#
# More than one galaxy can be specified via adding multiple entries to the list below; the resulting
# galaxy in the output image will be the *sum* of all entries in the list.
#
config.galaxy.type = ["DeVaucouleurs", "Exponential"]
#
# Distributions for galaxy parameters (including constant values) may be set below.  These will be 
# overwritten by any values for these parameters found in the input catalogues.
#
# e.g.:
#
# config.galaxy.Exponential.re.distribution = "GammaDeviate"
# config.galaxy.Exponential.re.alpha = 3.5
# config.galaxy.Exponential.re.beta = 1.3
#
# Any parameters not set either in configuration files or input catalogues will adopt their
# keyword default values, and GalSim will raise a Warning.
config.galaxy.DeVaucouleurs.dx.distribution = "GaussianDeviate"
config.galaxy.DeVaucouleurs.dy.distribution = "GaussianDeviate"
config.galaxy.DeVaucouleurs.dx.mean = 0.
config.galaxy.DeVaucouleurs.dy.mean = 0.
config.galaxy.DeVaucouleurs.dx.sigma = 1.0  # } Apply a Gaussian centroid offset ~N(0., 1.) to each 
config.galaxy.DeVaucouleurs.dy.sigma = 1.0  # } galaxy
config.galaxy.DeVaucouleurs.half_light_radius = 1.59  # GREAT08 single component value

bulge_disc_flux_ratio = 0.5  # flux(DeVaucouleurs) / flux(Exponential)
desired_SNR = 200.
# TODO: need a better mechanism to determine what flux we need given SNR!
# We make the approximation here (~following Mike) that the noise is essentially the noise in the
# pixels within the half_light_radius.
# The variance per pixel is given by ~the sky background value....
# The relevant number of pixels is pi * (half_flight_radius**2).
# ...So the total noise is sqrt(sky * pi * half_light_radius**2)
import math
dVc_noise = math.sqrt(config.sky.value * math.pi) * config.galaxy.DeVaucouleurs.half_light_radius 
config.galaxy.DeVaucouleurs.flux = bulge_disc_flux_ratio * desired_SNR * dVc_noise

config.galaxy.Exponential.dx = config.DeVaucouleurs.dx # } Make the two galaxy components co-centric
config.galaxy.Exponential.dy = config.DeVaucouleurs.dy # }
config.galaxy.Exponential.half_light_radius = 0.82     # GREAT08 single component value

# Then set the flux as done above for de Vaucouleurs galaxies
exp_noise = math.sqrt(config.sky.value * math.pi) * config.galaxy.Exponential.half_light_radius 
config.galaxy.Exponential.flux = bulge_disc_flux_ratio * desired_SNR * exp_noise




# --- shear ---
#
# Shear to be applied to images.  Can follow a distribution (e.g. according to a power spectrum) or
# be set to constant value.
#
# Distributions for parameters (including constant values) are set below.  These will be 
# overwritten by any values for these parameters found in the input catalogues.
#
config.shear.g1 =  0.013  # } Constant shear
config.shear.g2 = -0.008  # }


# --- noise ---
#
# Specify what type of noise to add to images (parameter values for the noise can either be set
# via config file such as this or via input catalogue values).
#
# The currently available noise models include:
#
# "GaussianDeviate"  Random noise with a Gaussian distribution of specified mean and sigma.
# "PoissonDeviate"   Random noise with a Poisson distribution of specified mean.
# "CCDNoise"         A simple model of Poisson + Gaussian random noise for a detector system with
#                    specified gain and read_noise.
#
# However, any of the random distributions implemented by GalSim may be applied, if desired.
# 
# Distributions for noise model parameters may be set below.  These will be overwritten by any 
# values for these parameters found in the input catalogues.
#
config.noise.distribution = ["CCDNoise"] 
config.noise.gain = 1.
config.noise.read_noise = 5.   #  off the top of Barney's head value!


# --- input cat ---
#
# Specify the type, name and expected contents of input catalogs to be used to set object-by-object
# parameters not specified by parameter distributions defined above.
#
# The currently supported input catalogue formats are:
#
# "ASCII"  Standard ASCII text format: note that the column field names must also be specified in
#          the order they will be presented in the catalogue via the config.input_cat.ascii_fields
#          parameter.
# "FITS"   FITS binary table format: the column field names will be expected to match a standard
#          format of <Object_name>.<parameter_name>, "Sersic.re" (see the table of object/parameter
#          pair field names in **TODO WRITE THIS TABLE!**).  This default behaviour can be modified
#          by supplying a dictionary of alternative field names where these differ in the FITS table
#          as a parameter config.input_cat.fits_field_dict.  If GalSim is unable to find all the 
#          fields it is expecting, it will raise a Warning and use default values for the object in
#          question.
#
config.input_cat.type = "ASCII"
#
# If using ASCII catalogs, give the column names in order in the list below, using names from 
# Table (TODO/XXXX)
#
# To ignore columns in the input cat, place a None object for that list entry.
#
# Note below, so that we can get matching pairs of rotated galaxy ellipticities, we will use
# catalog input containing the correct ellipticities in matched pairs in an object-by-object
# basis
config.input_cat.ascii_fields = ["Exponential.g1", "Exponential.g2",
                                 "DeVaucouleurs.g1", "DeVaucouleurs.g2"]
# *** Do not comment out the two lines below, it's a useful check! ***
if config.input_cat.type == "ASCII" and not hasattr(config.input_cat, "ascii_fields"):
    raise IOError("config.input_cat.ascii_fields must be set if using ASCII catalogues")
#
# Give the filename for the input catalog:
#
config.input_cat.filename = "examples/input/MultiObject1_input.asc"


# --- output ---
#
# Specify the type and filename of the output.  The currently supported output types WILL BE
# (TODO: ACTUALLY DO THIS!)
#
# "PStamps"  An individual postage stamp for each output object image.
# "Image"    A large 2D image containing multiple objects.
# "Cube"     A 3D array containing multiple object images, with the leading dimension = Number of
#            objects.
#
config.output.type = "Image"
config.output.filename = "galsim_MultiObjectDemo1_gal.fits"
#

# Note that here we adopt the same output size specifications as in config/galsim_default, and so
# need not set any additional values.

#
# Do you want a separate PSF output file?  If so specify below and give filename
#
config.output.PSFs.generate = True
config.output.PSFs.filename = "galsim_MultiObjectDemo1_psf.fits"


# Logging level at which to echo the commands we run
config.logging_level = logging.INFO
# Setup the basic logger.
logging.basicConfig(level=config.logging_level)#


