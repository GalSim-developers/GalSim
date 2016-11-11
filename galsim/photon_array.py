# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
"""@file photon_array.py
Implements the PhotonArray class describing a collection of photons incident on a detector.
Also includes classes that modify PhotonArray objects in a number of ways.
"""

import numpy as np
# Most of the functionality comes from the C++ layer
from ._galsim import PhotonArray
import galsim

# Add on more methods in the python layer

PhotonArray.__doc__ = """
The PhotonArray class encapsulates the concept of a collection of photons incident on
a detector.

A PhotonArray object is not typically constructed directly by the user.  Rather, it is
typically constructed as the return value of the `GSObject.shoot` method.
At this point, the photons only have x,y,flux values.  Then there are a number of classes
that perform various modifications to the photons such as giving them wavelenghts or
inclination angles or remove some due to fringing or vignetting.

TODO: fringing, vignetting, and angles are not implemented yet, but we expect them to
be implemented soon, so the above paragraph is a bit aspirational atm.

Attributes
----------

A PhotonArray instance has the following attributes, each of which is a numpy array:

- x,y           the incidence positions at the top of the detector
- flux          the flux of the photons
- dxdz, dydz    the tangent of the inclination angles in each direction
- wavelength    the wavelength of the photons

Unlike most GalSim objects (but like Images), PhotonArrays are mutable.  It is permissible
to write values to the above attributes with code like

    >>> photon_array.x += numpy.random.random(1000) * 0.01
    >>> photon_array.flux *= 20.
    >>> photon_array.wavelength = sed.sampleWavelength(photonarray.size(), bandpass)
    etc.

All of these will update the existing numpy arrays being used by the photon_array instance.

Note about the flux attribute
-----------------------------

Normal photons have flux=1, but we allow for "fat" photons that combine the effect of several
photons at once for efficiency.  Also, some profiles need to use negative flux photons to properly
implement photon shooting (e.g. InterpolateImage, which uses negative flux photons to get the
interpolation correct).  Finally, when we "remove" photons, for better efficiency, we actually
just set the flux to 0 rather than recreate new numpy arrays.

Initialization
--------------

The initialization constructs a PhotonArray to hold N photons, but does not set the values of
anything yet.  The constructor allocates space for the x,y,flux arrays, since those are always
needed.  The other arrays are only allocated on demand if the user accesses these attributes.

@param N            The number of photons to store in this PhotonArray.  This value cannot be
                    changed.
@param x            Optionally, the inital x values. [default: None]
@param y            Optionally, the inital y values. [default: None]
@param flux         Optionally, the inital flux values. [default: None]
@param dxdz         Optionally, the inital dxdz values. [default: None]
@param dydz         Optionally, the inital dydz values. [default: None]
@param wavelength   Optionally, the inital wavelength values. [default: None]
"""

# In python we want the init function to be a bit more functional so we can serialize properly.
# Save the C++-layer constructor as _PhotonArray_empty_init.
_PhotonArray_empty_init = PhotonArray.__init__

# Now make the one we want as the python-layer init function.
def PhotonArray_init(self, N, x=None, y=None, flux=None, dxdz=None, dydz=None, wavelength=None):
    _PhotonArray_empty_init(self, N)
    if x is not None: self.x[:] = x
    if y is not None: self.y[:] = y
    if flux is not None: self.flux[:] = flux
    if dxdz is not None: self.dxdz[:] = dxdz
    if dydz is not None: self.dydz[:] = dydz
    if wavelength is not None: self.wavelength[:] = wavelength
PhotonArray.__init__ = PhotonArray_init

PhotonArray.__getinitargs__ = lambda self: (
        self.size(), self.x, self.y, self.flux,
        self.dxdz if self.hasAllocatedAngles() else None,
        self.dydz if self.hasAllocatedAngles() else None,
        self.wavelength if self.hasAllocatedWavelengths() else None)

def PhotonArray_repr(self):
    s = "galsim.PhotonArray(%d, x=array(%r), y=array(%r), flux=array(%r)"%(
            self.size(), self.x.tolist(), self.y.tolist(), self.flux.tolist())
    if self.hasAllocatedAngles():
        s += ", dxdz=array(%r), dydz=array(%r)"%(self.dxdz.tolist(), self.dydz.tolist())
    if self.hasAllocatedWavelengths():
        s += ", wavelength=array(%r)"%(self.wavelength.tolist())
    s += ")"
    return s

def PhotonArray_str(self):
    return "galsim.PhotonArray(%d)"%self.size()

PhotonArray.__repr__ = PhotonArray_repr
PhotonArray.__str__ = PhotonArray_str
PhotonArray.__hash__ = None

PhotonArray.__eq__ = lambda self, other: (
        isinstance(other, PhotonArray) and
        np.array_equal(self.x,other.x) and
        np.array_equal(self.y,other.y) and
        np.array_equal(self.flux,other.flux) and
        self.hasAllocatedAngles() == other.hasAllocatedAngles() and
        self.hasAllocatedWavelengths() == other.hasAllocatedWavelengths() and
        (np.array_equal(self.dxdz,other.dxdz) if self.hasAllocatedAngles() else True) and
        (np.array_equal(self.dydz,other.dydz) if self.hasAllocatedAngles() else True) and
        (np.array_equal(self.wavelength,other.wavelength)
                if self.hasAllocatedWavelengths() else True) )
PhotonArray.__ne__ = lambda self, other: not self == other

# Make properties for convenient access to the various arrays
def PhotonArray_setx(self, x): self.getXArray()[:] = x
PhotonArray.x = property(PhotonArray.getXArray, PhotonArray_setx)

def PhotonArray_sety(self, y): self.getYArray()[:] = y
PhotonArray.y = property(PhotonArray.getYArray, PhotonArray_sety)

def PhotonArray_setflux(self, flux): self.getFluxArray()[:] = flux
PhotonArray.flux = property(PhotonArray.getFluxArray, PhotonArray_setflux)

def PhotonArray_setdxdz(self, dxdz): self.getDXDZArray()[:] = dxdz
PhotonArray.dxdz = property(PhotonArray.getDXDZArray, PhotonArray_setdxdz)

def PhotonArray_setdydz(self, dydz): self.getDYDZArray()[:] = dydz
PhotonArray.dydz = property(PhotonArray.getDYDZArray, PhotonArray_setdydz)

def PhotonArray_setwavelength(self, wavelength): self.getWavelengthArray()[:] = wavelength
PhotonArray.wavelength = property(PhotonArray.getWavelengthArray, PhotonArray_setwavelength)

def PhotonArray_makeFromImage(cls, image, max_flux=1., rng=None):
    """Turn an existing image into a PhotonArray that would accumulate into this image.

    The flux in each non-zero pixel will be turned into 1 or more photons with random positions
    within the pixel bounds.  The `max_flux` parameter (which defaults to 1) sets an upper
    limit for the absolute value of the flux of any photon.  Pixels with abs values > maxFlux will
    spawn multiple photons.

    TODO: This corresponds to the `Nearest` interpolant.  It would be worth figuring out how
          to implement other (presumably better) interpolation options here.

    @param image        The image to turn into a PhotonArray
    @param max_flux     The maximum flux value to use for any output photon [default: 1]
    @param rng          A BaseDeviate to use for the random number generation [default: None]

    @returns a PhotonArray
    """
    if rng is None:
        ud = galsim.UniformDeviate()
    else:
        ud = galsim.UniformDeviate(rng)
    max_flux = float(max_flux)
    return galsim._galsim.MakePhotonsFromImage(image.image, max_flux, ud)

PhotonArray.makeFromImage = classmethod(PhotonArray_makeFromImage)

class WavelengthSampler(object):
    """This class is a sensor operation that uses sed.sampleWavelength to set the wavelengths
    array of a PhotonArray.

    @param sed          The SED to use for the objects spectral energy distribution.
    @param bandpass     A Bandpass object representing a filter, or None to sample over the full
                        SED wavelength range.
    @param rng          If provided, a random number generator that is any kind of BaseDeviate
                        object. If `rng` is None, one will be automatically created, using the
                        time as a seed. [default: None]
    @param npoints      Number of points DistDeviate should use for its internal interpolation
                        tables. [default: 256]
    """
    def __init__(self, sed, bandpass, rng=None, npoints=256):
        self.sed = sed
        self.bandpass = bandpass
        self.rng = rng
        self.npoints = npoints

    def applyTo(self, photon_array):
        """Assign wavelengths to the photons sampled from the SED * Bandpass."""
        photon_array.wavelength = self.sed.sampleWavelength(
                photon_array.size(), self.bandpass, rng=self.rng, npoints=self.npoints)

class FRatioAngles(object):
    """A surface-layer operator that assigns photon directions based on the f/ratio and
    obscuration.

    Assigns arrival directions at the focal plane for photons, drawing from a uniform
    brightness distribution between the obscuration angle and the angle of the FOV defined
    by the f/ratio of the telescope.  The angles are expressed in terms of slopes dx/dz
    and dy/dz.

    @param fratio           The f-ratio of the telescope (e.g. 1.2 for LSST)
    @param obscuration      Linear dimension of central obscuration as fraction of aperture
                            linear dimension. [0., 1.).  [default: 0.0]
    @param rng              A random number generator to use or None, in which case an rng
                            will be automatically constructed for you. [default: None]
    """
    def __init__(self, fratio, obscuration=0.0, rng=None):

        if fratio < 0:
            raise ValueError("The f-ratio must be positive.")
        if obscuration < 0 or obscuration >= 1:
            raise ValueError("The obscuration fraction must be between 0 and 1.")
        if rng is None:
            ud = galsim.UniformDeviate()
        else:
            ud = galsim.UniformDeviate(rng)

        self.fratio = fratio
        self.obscuration = obscuration
        self.ud = ud


    def applyTo(self, photon_array):
        """Assign directions to the photons in photon_array."""

        dxdz = photon_array.getDXDZArray()
        dydz = photon_array.getDYDZArray()
        n_photons = len(dxdz)

        fov_angle = np.arctan(0.5 / self.fratio)  # radians
        obscuration_angle = self.obscuration * fov_angle

        # Generate azimuthal angles for the photons
        # Set up a loop to fill the array of azimuth angles for now
        # (The array is initialized below but there's no particular need to do this.)
        phi = np.zeros(n_photons)

        for i in np.arange(n_photons):
            phi[i] = self.ud() * 2 * np.pi

        # Generate inclination angles for the photons, which are uniform in sin(theta) between
        # the sine of the obscuration angle and the sine of the FOV radius
        sintheta = np.zeros(n_photons)

        for i in np.arange(n_photons):
            sintheta[i] = np.sin(obscuration_angle) + (np.sin(fov_angle) - \
                          np.sin(obscuration_angle))*self.ud()

        # Assign the directions to the arrays. In this class the convention for the
        # zero of phi does not matter but it might if the obscuration dependent on
        # phi
        costheta = np.sqrt(1. - np.square(sintheta))
        dxdz[:] = costheta * np.sin(phi)
        dydz[:] = costheta * np.cos(phi)


