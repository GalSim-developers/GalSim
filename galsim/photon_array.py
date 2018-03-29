# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
that perform various modifications to the photons such as giving them wavelengths or
inclination angles or removing some due to fringing or vignetting.

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
@param x            Optionally, the initial x values. [default: None]
@param y            Optionally, the initial y values. [default: None]
@param flux         Optionally, the initial flux values. [default: None]
@param dxdz         Optionally, the initial dxdz values. [default: None]
@param dydz         Optionally, the initial dydz values. [default: None]
@param wavelength   Optionally, the initial wavelength values. [default: None]
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
def PhotonArray_setx(self, x): self._getXArray()[:] = x
PhotonArray.x = property(PhotonArray._getXArray, PhotonArray_setx)

def PhotonArray_sety(self, y): self._getYArray()[:] = y
PhotonArray.y = property(PhotonArray._getYArray, PhotonArray_sety)

def PhotonArray_setflux(self, flux): self._getFluxArray()[:] = flux
PhotonArray.flux = property(PhotonArray._getFluxArray, PhotonArray_setflux)

def PhotonArray_setdxdz(self, dxdz): self._getDXDZArray()[:] = dxdz
PhotonArray.dxdz = property(PhotonArray._getDXDZArray, PhotonArray_setdxdz)

def PhotonArray_setdydz(self, dydz): self._getDYDZArray()[:] = dydz
PhotonArray.dydz = property(PhotonArray._getDYDZArray, PhotonArray_setdydz)

def PhotonArray_setwavelength(self, wavelength): self._getWavelengthArray()[:] = wavelength
PhotonArray.wavelength = property(PhotonArray._getWavelengthArray, PhotonArray_setwavelength)

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
    ud = galsim.UniformDeviate(rng)
    max_flux = float(max_flux)
    if (max_flux <= 0):
        raise ValueError("max_flux must be positive")
    photons = galsim._galsim.MakePhotonsFromImage(image._image, max_flux, ud)
    if image.scale != 1.:
        photons.scaleXY(image.scale)
    return photons

PhotonArray.makeFromImage = classmethod(PhotonArray_makeFromImage)

def PhotonArray_write(self, file_name):
    """Write a PhotonArray to a FITS file.

    The output file will be a FITS binary table with a row for each photon in the PhotonArray.
    Columns will include 'id' (sequential from 1 to nphotons), 'x', 'y', and 'flux'.
    Additionally, the columns 'dxdz', 'dydz', and 'wavelength' will be included if they are
    set for this PhotonArray object.

    The file can be read back in with the classmethod `PhotonArray.read`.

        >>> photons.write('photons.fits')
        >>> photons2 = galsim.PhotonArray.read('photons.fits')

    @param file_name    The file name of the output FITS file.
    """
    from galsim._pyfits import pyfits

    cols = []
    cols.append(pyfits.Column(name='id', format='J', array=range(self.size())))
    cols.append(pyfits.Column(name='x', format='D', array=self.x))
    cols.append(pyfits.Column(name='y', format='D', array=self.y))
    cols.append(pyfits.Column(name='flux', format='D', array=self.flux))

    if self.hasAllocatedAngles():
        cols.append(pyfits.Column(name='dxdz', format='D', array=self.dxdz))
        cols.append(pyfits.Column(name='dydz', format='D', array=self.dydz))

    if self.hasAllocatedWavelengths():
        cols.append(pyfits.Column(name='wavelength', format='D', array=self.wavelength))

    cols = pyfits.ColDefs(cols)
    try:
        table = pyfits.BinTableHDU.from_columns(cols)
    except AttributeError:  # pragma: no cover  (Might need this for older pyfits versions)
        table = pyfits.new_table(cols)
    galsim.fits.writeFile(file_name, table)

def PhotonArray_read(cls, file_name):
    """Create a PhotonArray, reading the photon data from a FITS file.

    The file being read in is not arbitrary.  It is expected to be a file that was written
    out with the PhotonArray `write` method.

        >>> photons.write('photons.fits')
        >>> photons2 = galsim.PhotonArray.read('photons.fits')

    @param file_name    The file name of the input FITS file.
    """
    from galsim._pyfits import pyfits, pyfits_version
    with pyfits.open(file_name) as fits:
        data = fits[1].data
    N = len(data)
    if pyfits_version > '3.0':
        names = data.columns.names
    else: # pragma: no cover
        names = data.dtype.names

    ret = cls.__new__(cls)
    _PhotonArray_empty_init(ret, N)
    ret.x = data['x']
    ret.y = data['y']
    ret.flux = data['flux']
    if 'dxdz' in names:
        ret.dxdz = data['dxdz']
        ret.dydz = data['dydz']
    if 'wavelength' in names:
        ret.wavelength = data['wavelength']
    return ret

PhotonArray.write = PhotonArray_write
PhotonArray.read = classmethod(PhotonArray_read)

orig_addTo = PhotonArray.addTo
def PhotonArray_addTo(self, image):
    """Add flux of photons to an image by binning into pixels.
    """
    if isinstance(image, galsim.Image):
        return orig_addTo(self, image._image.view())
    else:
        from .deprecated import depr
        depr("C++-layer image as argument to PhotonArray.addTo", 1.5, "Use a regular galsim.Image")
        return orig_addTo(self, image.view())
PhotonArray.addTo = PhotonArray_addTo

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
    brightness distribution between the obscuration angle and the edge of the pupil defined
    by the f/ratio of the telescope.  The angles are expressed in terms of slopes dx/dz
    and dy/dz.

    @param fratio           The f/ratio of the telescope (e.g. 1.2 for LSST)
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
        ud = galsim.UniformDeviate(rng)

        self.fratio = fratio
        self.obscuration = obscuration
        self.ud = ud


    def applyTo(self, photon_array):
        """Assign directions to the photons in photon_array."""

        dxdz = photon_array.dxdz
        dydz = photon_array.dydz
        n_photons = len(dxdz)

        # The f/ratio is the ratio of the focal length to the diameter of the aperture of
        # the telescope.  The angular radius of the field of view is defined by the
        # ratio of the radius of the aperture to the focal length
        pupil_angle = np.arctan(0.5 / self.fratio)  # radians
        obscuration_angle = np.arctan(0.5 * self.obscuration / self.fratio)

        # Generate azimuthal angles for the photons
        phi = np.empty(n_photons)
        self.ud.generate(phi)
        phi *= (2 * np.pi)

        # Generate inclination angles for the photons, which are uniform in sin(theta) between
        # the sine of the obscuration angle and the sine of the pupil radius
        sintheta = np.empty(n_photons)
        self.ud.generate(sintheta)
        sintheta = np.sin(obscuration_angle) + (np.sin(pupil_angle) - np.sin(obscuration_angle)) \
            * sintheta

        # Assign the directions to the arrays. In this class the convention for the
        # zero of phi does not matter but it would if the obscuration is dependent on
        # phi
        tantheta = np.sqrt(np.square(sintheta) / (1. - np.square(sintheta)))
        dxdz[:] = tantheta * np.sin(phi)
        dydz[:] = tantheta * np.cos(phi)
