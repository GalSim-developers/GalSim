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
"""@file photon_array.py
Implements the PhotonArray class describing a collection of photons incident on a detector.
Also includes classes that modify PhotonArray objects in a number of ways.
"""

import numpy as np

from . import _galsim
from .random import UniformDeviate, BaseDeviate
from .utilities import lazy_property
from .angle import radians, arcsec
from .errors import GalSimError, GalSimRangeError, GalSimValueError, GalSimUndefinedBoundsError
from .errors import GalSimIncompatibleValuesError, convert_cpp_errors

# Add on more methods in the python layer

class PhotonArray(object):
    """The PhotonArray class encapsulates the concept of a collection of photons incident on
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
    photons at once for efficiency.  Also, some profiles need to use negative flux photons to
    properly implement photon shooting (e.g. InterpolateImage, which uses negative flux photons to
    get the interpolation correct).  Finally, when we "remove" photons, for better efficiency, we
    actually just set the flux to 0 rather than recreate new numpy arrays.

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
    def __init__(self, N, x=None, y=None, flux=None, dxdz=None, dydz=None, wavelength=None):
        # Only x, y, flux are built by default, since these are always required.
        # The others we leave as None unless/until they are needed.
        self._x = np.zeros(N, dtype=float)
        self._y = np.zeros(N, dtype=float)
        self._flux = np.zeros(N, dtype=float)
        self._dxdz = None
        self._dydz = None
        self._wave = None
        self._is_corr = False

        # These give reasonable errors in x,y,flux are the wrong size/type
        if x is not None: self.x = x
        if y is not None: self.y = y
        if flux is not None: self.flux = flux
        if dxdz is not None: self.dxdz = dxdz
        if dydz is not None: self.dydz = dydz
        if wavelength is not None: self.wavelength = wavelength

    def size(self):
        return len(self._x)

    def __len__(self):
        return len(self._x)

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x[:] = value

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        self._y[:] = value

    @property
    def flux(self):
        return self._flux
    @flux.setter
    def flux(self, value):
        self._flux[:] = value

    @property
    def dxdz(self):
        self.allocateAngles()
        return self._dxdz
    @dxdz.setter
    def dxdz(self, value):
        self.allocateAngles()
        self._dxdz[:] = value

    @property
    def dydz(self):
        self.allocateAngles()
        return self._dydz
    @dydz.setter
    def dydz(self, value):
        self.allocateAngles()
        self._dydz[:] = value

    @property
    def wavelength(self):
        self.allocateWavelengths()
        return self._wave
    @wavelength.setter
    def wavelength(self, value):
        self.allocateWavelengths()
        self._wave[:] = value

    def hasAllocatedAngles(self):
        return self._dxdz is not None and self._dydz is not None
    def allocateAngles(self):
        if self._dxdz is None:
            self._dxdz = np.zeros_like(self._x)
            self._dydz = np.zeros_like(self._x)
            self.__dict__.pop('_pa', None)

    def hasAllocatedWavelengths(self):
        return self._wave is not None
    def allocateWavelengths(self):
        if self._wave is None:
            self._wave = np.zeros_like(self._x)
            self.__dict__.pop('_pa', None)

    def isCorrelated(self):
        "Returns whether the photons are correlated"
        return self._is_corr
    def setCorrelated(self, is_corr=True):
        "Set whether the photons are correlated"
        self._is_corr = is_corr
        self.__dict__.pop('_pa', None)

    def getTotalFlux(self):
        return self.flux.sum()

    def setTotalFlux(self, flux):
        self.scaleFlux(flux / self.getTotalFlux())

    def scaleFlux(self, scale):
        self._flux *= scale

    def scaleXY(self, scale):
        self._x *= float(scale)
        self._y *= float(scale)

    def assignAt(self, istart, rhs):
        "Assign the contents of another PhotonArray to this one starting at istart."
        if istart + rhs.size() > self.size():
            raise GalSimValueError(
                "The given rhs does not fit into this array starting at %d"%istart, rhs)
        s = slice(istart, istart + rhs.size())
        self.x[s] = rhs.x
        self.y[s] = rhs.y
        self.flux[s] = rhs.flux
        if rhs.hasAllocatedAngles():
            self.dxdz[s] = rhs.dxdz
            self.dydz[s] = rhs.dydz
        if rhs.hasAllocatedWavelengths():
            self.wavelength[s] = rhs.wavelength

    def convolve(self, rhs, rng=None):
        "Convolve this PhotonArray with another."
        if rhs.size() != self.size():
            raise GalSimIncompatibleValuesError("PhotonArray.convolve with unequal size arrays",
                                                self_pa=self, rhs=rhs)
        if rng is None:
            rng = BaseDeviate()
        self._pa.convolve(rhs._pa, rng._rng)

    def __repr__(self):
        s = "galsim.PhotonArray(%d, x=array(%r), y=array(%r), flux=array(%r)"%(
                self.size(), self.x.tolist(), self.y.tolist(), self.flux.tolist())
        if self.hasAllocatedAngles():
            s += ", dxdz=array(%r), dydz=array(%r)"%(self.dxdz.tolist(), self.dydz.tolist())
        if self.hasAllocatedWavelengths():
            s += ", wavelength=array(%r)"%(self.wavelength.tolist())
        s += ")"
        return s

    def __str__(self):
        return "galsim.PhotonArray(%d)"%self.size()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_pa',None)
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    __hash__ = None

    def __eq__(self, other):
        return (self is other or
                (isinstance(other, PhotonArray) and
                 np.array_equal(self.x,other.x) and
                 np.array_equal(self.y,other.y) and
                 np.array_equal(self.flux,other.flux) and
                 self.hasAllocatedAngles() == other.hasAllocatedAngles() and
                 self.hasAllocatedWavelengths() == other.hasAllocatedWavelengths() and
                 (np.array_equal(self.dxdz,other.dxdz) if self.hasAllocatedAngles() else True) and
                 (np.array_equal(self.dydz,other.dydz) if self.hasAllocatedAngles() else True) and
                 (np.array_equal(self.wavelength,other.wavelength)
                    if self.hasAllocatedWavelengths() else True) ))

    def __ne__(self, other):
        return not self == other

    @lazy_property
    def _pa(self):
        #assert(self._x.strides[0] == self._x.itemsize)
        #assert(self._y.strides[0] == self._y.itemsize)
        #assert(self._flux.strides[0] == self._flux.itemsize)
        _x = self._x.ctypes.data
        _y = self._y.ctypes.data
        _flux = self._flux.ctypes.data
        _dxdz = _dydz = _wave = 0
        if self.hasAllocatedAngles():
            #assert(self._dxdz.strides[0] == self._dxdz.itemsize)
            #assert(self._dydz.strides[0] == self._dydz.itemsize)
            _dxdz = self._dxdz.ctypes.data
            _dydz = self._dydz.ctypes.data
        if self.hasAllocatedWavelengths():
            #assert(self._wave.strides[0] == self._wave.itemsize)
            _wave = self._wave.ctypes.data
        with convert_cpp_errors():
            return _galsim.PhotonArray(int(self.size()), _x, _y, _flux, _dxdz, _dydz, _wave,
                                       self._is_corr)

    def addTo(self, image):
        """Add flux of photons to an image by binning into pixels.

        Photon in this PhotonArray are binned into the pixels of the input
        Image and their flux summed into the pixels.  Image is assumed to represent
        surface brightness, so photons' fluxes are divided by image pixel area.
        Photons past the edges of the image are discarded.

        @param image    the Image to which the photons' flux will be added.

        @returns the total flux of photons the landed inside the image bounds.
        """
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Attempting to PhotonArray::addTo an Image with undefined Bounds")
        return self._pa.addTo(image._image)

    @classmethod
    def makeFromImage(cls, image, max_flux=1., rng=None):
        """Turn an existing image into a PhotonArray that would accumulate into this image.

        The flux in each non-zero pixel will be turned into 1 or more photons with random positions
        within the pixel bounds.  The `max_flux` parameter (which defaults to 1) sets an upper
        limit for the absolute value of the flux of any photon.  Pixels with abs values > maxFlux
        will spawn multiple photons.

        TODO: This corresponds to the `Nearest` interpolant.  It would be worth figuring out how
              to implement other (presumably better) interpolation options here.

        @param image        The image to turn into a PhotonArray
        @param max_flux     The maximum flux value to use for any output photon [default: 1]
        @param rng          A BaseDeviate to use for the random number generation [default: None]

        @returns a PhotonArray
        """
        max_flux = float(max_flux)
        if (max_flux <= 0):
            raise GalSimRangeError("max_flux must be positive", max_flux, 0.)
        total_flux = image.array.sum(dtype=float)

        # This goes a bit over what we actually need, but not by much.  Worth it to not have to
        # worry about array reallocations.
        N = int(np.prod(image.array.shape) + total_flux / max_flux)
        photons = cls(N)

        if rng is None:
            rng = BaseDeviate()
        N = photons._pa.setFrom(image._image, max_flux, rng._rng)
        photons._x = photons.x[:N]
        photons._y = photons.y[:N]
        photons._flux = photons.flux[:N]

        if image.scale != 1. and image.scale is not None:
            photons.scaleXY(image.scale)
        return photons

    def write(self, file_name):
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
        from ._pyfits import pyfits
        from . import fits

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
        fits.writeFile(file_name, table)

    @classmethod
    def read(cls, file_name):
        """Create a PhotonArray, reading the photon data from a FITS file.

        The file being read in is not arbitrary.  It is expected to be a file that was written
        out with the PhotonArray `write` method.

            >>> photons.write('photons.fits')
            >>> photons2 = galsim.PhotonArray.read('photons.fits')

        @param file_name    The file name of the input FITS file.
        """
        from ._pyfits import pyfits
        with pyfits.open(file_name) as fits:
            data = fits[1].data
        N = len(data)
        names = data.columns.names

        photons = cls(N, x=data['x'], y=data['y'], flux=data['flux'])
        if 'dxdz' in names:
            photons.dxdz = data['dxdz']
            photons.dydz = data['dydz']
        if 'wavelength' in names:
            photons.wavelength = data['wavelength']
        return photons

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
                        tables. [default: None, which uses the DistDeviate default]
    """
    def __init__(self, sed, bandpass, rng=None, npoints=None):
        self.sed = sed
        self.bandpass = bandpass
        self.rng = BaseDeviate(rng)
        self.npoints = npoints

    def applyTo(self, photon_array, local_wcs=None):
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
            raise GalSimRangeError("The f-ratio must be positive.", fratio, 0.)
        if obscuration < 0 or obscuration >= 1:
            raise GalSimRangeError("Invalid obscuration.", obscuration, 0., 1.)
        ud = UniformDeviate(rng)

        self.fratio = fratio
        self.obscuration = obscuration
        self.ud = ud


    def applyTo(self, photon_array, local_wcs=None):
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

class PhotonDCR(object):
    """A surface-layer operator that applies the effect of differential chromatic refraction (DCR)
    and optionally the chromatic dilation due to atmospheric seeing.

    Due to DCR, blue photons land closer to the zenith than red photons.  Kolmogorov turbulence
    also predicts that blue photons get spread out more by the atmosphere than red photons,
    specifically FWHM is proportional to wavelength^(-0.2).  Both of these effects can be
    implemented by wavelength-dependent shifts of the photons.

    Since DCR depends on the zenith angle and the parallactic angle (which is the position angle of
    the zenith measured from North through East) of the object being drawn, these must be specified
    via keywords.  There are four ways to specify these values:
      1) explicitly provide `zenith_angle = ...` as a keyword of type Angle, and
         `parallactic_angle` will be assumed to be 0 by default.
      2) explicitly provide both `zenith_angle = ...` and `parallactic_angle = ...` as
         keywords of type Angle.
      3) provide the coordinates of the object `obj_coord = ...` and the coordinates of the zenith
         `zenith_coord = ...` as keywords of type CelestialCoord.
      4) provide the coordinates of the object `obj_coord = ...` as a CelestialCoord, the
         hour angle of the object `HA = ...` as an Angle, and the latitude of the observer
         `latitude = ...` as an Angle.

    DCR also depends on temperature, pressure and water vapor pressure of the atmosphere.  The
    default values for these are expected to be appropriate for LSST at Cerro Pachon, Chile, but
    they are broadly reasonable for most observatories.

    This surface op is intended to match the functionality of ChromaticAtmosphere, but acting
    on the photon array rather than as a ChromaticObject.  The photons will need to have
    wavelengths defined in order to work.

    IMPORTANT: The alpha parameter is only appropriate for stars.  This surface op will act on
               all of the photons, so applying a chromatic dilation according to the chromatic
               seeing is the wrong thing to do when the surface brightness being rendered is
               not a pure PSF.  As such, the default is alpha=0, not -0.2, which would be
               appropriate for Kolmogorov turbulence.

    @param base_wavelength      Wavelength (in nm) represented by the fiducial photon positions
    @param scale_unit           Units used for the positions of the photons.  [default:
                                galsim.arcsec]
    @param alpha                Power law index for wavelength-dependent seeing.  This should only
                                be used if doing a star-only simulation.  It is not correct when
                                drawing galaxies. [default: 0.]
    @param zenith_angle         Angle from object to zenith, expressed as an Angle
                                [default: 0]
    @param parallactic_angle    Parallactic angle, i.e. the position angle of the zenith, measured
                                from North through East.  [default: 0]
    @param obj_coord            Celestial coordinates of the object being drawn as a
                                CelestialCoord. [default: None]
    @param zenith_coord         Celestial coordinates of the zenith as a CelestialCoord.
                                [default: None]
    @param HA                   Hour angle of the object as an Angle. [default: None]
    @param latitude             Latitude of the observer as an Angle. [default: None]
    @param pressure             Air pressure in kiloPascals.  [default: 69.328 kPa]
    @param temperature          Temperature in Kelvins.  [default: 293.15 K]
    @param H2O_pressure         Water vapor pressure in kiloPascals.  [default: 1.067 kPa]
    """
    def __init__(self, base_wavelength, scale_unit=arcsec, **kwargs):
        from . import dcr
        from . import angle

        # This matches the code in ChromaticAtmosphere.
        self.base_wavelength = base_wavelength

        if isinstance(scale_unit, str):
            scale_unit = angle.AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self.alpha = kwargs.pop('alpha', 0.)

        self.zenith_angle, self.parallactic_angle, self.kw = dcr.parse_dcr_angles(**kwargs)

        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in self.kw:
            if kw not in ('temperature', 'pressure', 'H2O_pressure'):
                raise TypeError("Got unexpected keyword: {0}".format(kw))

        self.base_refraction = dcr.get_refraction(self.base_wavelength, self.zenith_angle,
                                                         **self.kw)

    def applyTo(self, photon_array, local_wcs):
        """Apply the DCR effect to the photons
        """
        from . import dcr
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("PhotonDCR requires that wavelengths be set")

        w = photon_array.wavelength
        cenx = local_wcs.origin.x
        ceny = local_wcs.origin.y

        # Apply the wavelength-dependent scaling
        if self.alpha != 0.:
            scale = (w/self.base_wavelength)**self.alpha
            photon_array.x = scale * (photon_array.x - cenx) + cenx
            photon_array.y = scale * (photon_array.y - ceny) + ceny

        # Apply DCR
        shift_magnitude = dcr.get_refraction(w, self.zenith_angle, **self.kw)
        shift_magnitude -= self.base_refraction
        shift_magnitude = shift_magnitude * (radians / self.scale_unit)
        sinp, cosp = self.parallactic_angle.sincos()

        du = -shift_magnitude * sinp
        dv = shift_magnitude * cosp

        dx = local_wcs._x(du, dv)
        dy = local_wcs._y(du, dv)
        photon_array.x += dx
        photon_array.y += dy
