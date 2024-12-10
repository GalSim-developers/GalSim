# Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

__all__ = [ 'PhotonArray', 'PhotonOp', 'WavelengthSampler', 'FRatioAngles',
            'PhotonDCR', 'Refraction', 'FocusDepth',
            'PupilImageSampler', 'PupilAnnulusSampler', 'TimeSampler',
            'ScaleFlux', 'ScaleWavelength' ]

import numpy as np
import astropy.units as u

from . import _galsim
from .random import BaseDeviate
from .celestial import CelestialCoord
from ._utilities import lazy_property
from .angle import radians, arcsec, Angle, AngleUnit
from .errors import GalSimError, GalSimRangeError, GalSimValueError, GalSimUndefinedBoundsError
from .errors import GalSimIncompatibleValuesError, galsim_warn
from ._pyfits import pyfits
from . import dcr

# Add on more methods in the python layer

class PhotonArray:
    """The PhotonArray class encapsulates the concept of a collection of photons incident on
    a detector.

    A PhotonArray object is not typically constructed directly by the user.  Rather, it is
    typically constructed as the return value of the `GSObject.shoot` method.
    At this point, the photons only have x,y,flux values.

    Then there are a number of `Photon Operators`, which perform various modifications to the
    photons such as giving them wavelengths (`WavelengthSampler` or inclination angles
    (`FRatioAngles`) or move them around according to the effect of differential chromatic
    refraction (DCR; `PhotonDCR`).

    One could also add functionality to remove some photons due to fringing or vignetting,
    but these are not yet implemented.

    A PhotonArray instance has the following attributes, each of which is a numpy array:

    Attributes:
        x:          The incidence x position of the photons in image coordinates (pixels),
                    typically measured at the top of the detector.
        y:          The incidence y position of the photons in image coordinates (pixels),
                    typically measured at the top of the detector.
        flux:       The flux of the photons in units of photons. Typically, these are all 1,
                    but see the note below for reasons some photons might have flux != 1.
        dxdz:       The tangent of the inclination angles in the x direction.  Note that we define
                    the +z direction as towards towards the dielectric medium of the detector and
                    -z as towards vacuum; consequently, a photon with increasing x in time has
                    positive dxdz.
        dydz:       The tangent of the inclination angles in the y direction.  Note that we define
                    the +z direction as towards towards the dielectric medium of the detector and
                    -z as towards vacuum; consequently, a photon with increasing y in time has
                    positive dydz.
        wavelength  The wavelength of the photons (in nm)
        pupil_u:    Horizontal location of photon as it intersected the entrance pupil plane
                    (meters).
        pupil_v:    Vertical location of photon as it intersected the entrance pupil plane
                    (meters).
        time:       Time stamp for photon impacting the pupil plane (seconds).

    Unlike most GalSim objects (but like `Image`), PhotonArrays are mutable.  It is permissible
    to write values to the above attributes with code like::

        >>> photon_array.x += numpy.random.random(1000) * 0.01
        >>> photon_array.flux *= 20.
        >>> photon_array.wavelength = sed.sampleWavelength(photonarray.size(), bandpass)
        etc.

    All of these will update the existing numpy arrays being used by the photon_array instance.

    .. note::

        Normal photons have flux=1, but we allow for "fat" photons that combine the effect of
        several photons at once for efficiency.  Also, some profiles need to use negative flux
        photons to properly implement photon shooting (e.g. `InterpolatedImage`, which uses negative
        flux photons to get the interpolation correct).  Finally, when we "remove" photons, for
        better efficiency, we actually just set the flux to 0 rather than recreate new numpy arrays.

    The initialization constructs a PhotonArray to hold N photons, but does not set the values of
    anything yet.  The constructor allocates space for the x,y,flux arrays, since those are always
    needed.  The other arrays are only allocated on demand if the user accesses these attributes.

    Parameters:
        N:          The number of photons to store in this PhotonArray.  This value cannot be
                    changed.
        x:          Optionally, the initial x values. [default: None]
        y:          Optionally, the initial y values. [default: None]
        flux:       Optionally, the initial flux values. [default: None]
        dxdz:       Optionally, the initial dxdz values. [default: None]
        dydz:       Optionally, the initial dydz values. [default: None]
        wavelength: Optionally, the initial wavelength values (in nm). [default: None]
        pupil_u:    Optionally, the initial pupil_u values. [default: None]
        pupil_v:    Optionally, the initial pupil_v values. [default: None]
        time:       Optionally, the initial time values. [default: None]
    """
    def __init__(
        self, N, x=None, y=None, flux=None, dxdz=None, dydz=None, wavelength=None,
        pupil_u=None, pupil_v=None, time=None
    ):
        # Only x, y, flux are built by default, since these are always required.
        # The others we leave as None unless/until they are needed.
        self._x = np.zeros(N, dtype=float)
        self._y = np.zeros(N, dtype=float)
        self._flux = np.zeros(N, dtype=float)
        self._dxdz = None
        self._dydz = None
        self._wave = None
        self._pupil_u = None
        self._pupil_v = None
        self._time = None
        self._is_corr = False

        # These give reasonable errors if x,y,flux are the wrong size/type
        if x is not None: self.x = x
        if y is not None: self.y = y
        if flux is not None: self.flux = flux
        if dxdz is not None: self.dxdz = dxdz
        if dydz is not None: self.dydz = dydz
        if wavelength is not None: self.wavelength = wavelength
        if pupil_u is not None: self.pupil_u = pupil_u
        if pupil_v is not None: self.pupil_v = pupil_v
        if time is not None: self.time = time

    @classmethod
    def fromArrays(
        cls, x, y, flux, dxdz=None, dydz=None, wavelength=None, pupil_u=None, pupil_v=None,
        time=None, is_corr=False,
    ):
        """Create a PhotonArray from pre-allocated numpy arrays without any copying.

        The normal PhotonArray constructor always allocates new arrays and copies any provided
        initial values into those new arrays.  This class method, by constrast, constructs a
        PhotonArray that references existing numpy arrays, so that any PhotonOps or photon shooting
        of GSObjects applied to the resulting PhotonArray will also be reflected in the original
        arrays.

        Note that the input arrays must all be the same length, have dtype float64 and be
        c_contiguous.

        Parameters:
            x:          X values.
            y:          X values.
            flux:       Flux values.
            dxdz:       Optionally, the initial dxdz values. [default: None]
            dydz:       Optionally, the initial dydz values. [default: None]
            wavelength: Optionally, the initial wavelength values (in nm). [default: None]
            pupil_u:    Optionally, the initial pupil_u values (in m). [default: None]
            pupil_v:    Optionally, the initial pupil_v values (in m). [default: None]
            time:       Optionally, the initial time values (in s). [default: None]
            is_corr:    Whether or not the photons are correlated. [default: False]
        """
        args = [x, y, flux]
        argnames = ['x', 'y', 'flux']
        for a, aname in zip(
            [dxdz, dydz, wavelength, pupil_u, pupil_v, time],
            ['dxdz', 'dydz', 'wavelength', 'pupil_u', 'pupil_v', 'time']
        ):
            if a is not None:  # don't check optional args that are None
                args.append(a)
                argnames.append(aname)

        N = len(x)
        for a, aname in zip(args, argnames):
            if not isinstance(a, np.ndarray):
                raise TypeError("Argument {} must be an ndarray".format(aname))
            if not a.dtype == np.float64:
                raise TypeError("Array {} dtype must be np.float64".format(aname))
            if not len(a) == N:
                raise ValueError("Arrays must all be the same length")
            if not a.flags.c_contiguous:
                raise ValueError("Array {} must be c_contiguous".format(aname))

        return cls._fromArrays(x, y, flux, dxdz, dydz, wavelength, pupil_u, pupil_v, time, is_corr)

    @classmethod
    def _fromArrays(
        cls, x, y, flux, dxdz=None, dydz=None, wavelength=None, pupil_u=None, pupil_v=None,
        time=None, is_corr=False
    ):
        """Same as `fromArrays`, but no sanity checking of inputs.
        """
        ret = PhotonArray.__new__(PhotonArray)
        ret._x = x
        ret._y = y
        ret._flux = flux
        ret._dxdz = dxdz
        ret._dydz = dydz
        ret._wave = wavelength
        ret._pupil_u = pupil_u
        ret._pupil_v = pupil_v
        ret._time = time
        ret._is_corr = False
        if is_corr:
            from .deprecated import depr
            depr('is_corr=True', 2.5, '',
                "We don't think this is necessary anymore.  If you have a use case that "
                "requires it, please open an issue.")
            ret._is_corr = is_corr
        return ret

    @classmethod
    def concatenate(cls, photon_arrays):
        """Create a single PhotonArray from a list of multiple PhotonArrays.

        The size of the created PhotonArray will be the sum of the sizes of the given arrays,
        and the values will be concatenations of the values in each.

        .. note::
            The optional value arrays (e.g. dxdz, wavelength, etc.) must be given in
            all the given photon_arrays or in none of them.  This is not checked.

        Parameters:
            photon_arrays:  A list of PhotonArray objects to be concatenated.
        """
        p1 = photon_arrays[0]
        kwargs = {
            'x': np.concatenate([p.x for p in photon_arrays]),
            'y': np.concatenate([p.y for p in photon_arrays]),
            'flux': np.concatenate([p.flux for p in photon_arrays]),
        }

        if p1.hasAllocatedAngles():
            kwargs['dxdz'] = np.concatenate([p.dxdz for p in photon_arrays])
            kwargs['dydz'] = np.concatenate([p.dydz for p in photon_arrays])
        if p1.hasAllocatedWavelengths():
            kwargs['wavelength'] = np.concatenate([p.wavelength for p in photon_arrays])
        if p1.hasAllocatedPupil():
            kwargs['pupil_u'] = np.concatenate([p.pupil_u for p in photon_arrays])
            kwargs['pupil_v'] = np.concatenate([p.pupil_v for p in photon_arrays])
        if p1.hasAllocatedTimes():
            kwargs['time'] = np.concatenate([p.time for p in photon_arrays])

        return cls._fromArrays(**kwargs)

    def size(self):
        """Return the size of the photon array.  Equivalent to ``len(self)``.
        """
        return len(self._x)

    def __len__(self):
        return len(self._x)

    @property
    def x(self):
        """The incidence x position in image coordinates (pixels), typically at the top of
        the detector.
        """
        return self._x
    @x.setter
    def x(self, value):
        self._x[:] = value

    @property
    def y(self):
        """The incidence y position in image coordinates (pixels), typically at the top of
        the detector.
        """
        return self._y
    @y.setter
    def y(self, value):
        self._y[:] = value

    @property
    def flux(self):
        """The flux of the photons.
        """
        return self._flux
    @flux.setter
    def flux(self, value):
        self._flux[:] = value

    @property
    def dxdz(self):
        """The tangent of the inclination angles in the x direction: dx/dz.
        """
        if not self.hasAllocatedAngles():
            from .deprecated import depr
            depr('dxdz accessed before being set.', 2.5,
                 'Angle arrays should be set or explicitly allocated before being accessed.',
                 'For now, accessing dxdz allocates an array with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocateAngles()
        return self._dxdz
    @dxdz.setter
    def dxdz(self, value):
        self.allocateAngles()
        self._dxdz[:] = value

    @property
    def dydz(self):
        """The tangent of the inclination angles in the y direction: dy/dz.
        """
        if not self.hasAllocatedAngles():
            from .deprecated import depr
            depr('dydz accessed before being set.', 2.5,
                 'Angle arrays should be set or explicitly allocated before being accessed.',
                 'For now, accessing dydz allocates an array with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocateAngles()
        return self._dydz
    @dydz.setter
    def dydz(self, value):
        self.allocateAngles()
        self._dydz[:] = value

    @property
    def wavelength(self):
        """The wavelength of the photons (in nm).
        """
        if not self.hasAllocatedWavelengths():
            from .deprecated import depr
            depr('wavelength accessed before being set.', 2.5,
                 'Wavelength array should be set or explicitly allocated before being accessed.',
                 'For now, accessing wavelength allocates arrays with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocateWavelengths()
        return self._wave
    @wavelength.setter
    def wavelength(self, value):
        self.allocateWavelengths()
        self._wave[:] = value

    @property
    def pupil_u(self):
        """Horizontal location of photon as it intersected the entrance pupil plane.
        """
        if not self.hasAllocatedPupil():
            from .deprecated import depr
            depr('pupil_u accessed before being set.', 2.5,
                 'Pupil arrays should be set or explicitly allocated before being accessed.',
                 'For now, accessing pupil_u allocates arrays with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocatePupil()
        return self._pupil_u
    @pupil_u.setter
    def pupil_u(self, value):
        self.allocatePupil()
        self._pupil_u[:] = value

    @property
    def pupil_v(self):
        """Vertical location of photon as it intersected the entrance pupil plane.
        """
        if not self.hasAllocatedPupil():
            from .deprecated import depr
            depr('pupil_v accessed before being set.', 2.5,
                 'Pupil arrays should be set or explicitly allocated before being accessed.',
                 'For now, accessing pupil_v allocates arrays with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocatePupil()
        return self._pupil_v
    @pupil_v.setter
    def pupil_v(self, value):
        self.allocatePupil()
        self._pupil_v[:] = value

    @property
    def time(self):
        """Time stamp of when photon encounters the pupil plane.
        """
        if not self.hasAllocatedTimes():
            from .deprecated import depr
            depr('time accessed before being set.', 2.5,
                 'Time array should be set or explicitly allocated before being accessed.',
                 'For now, accessing time allocates arrays with all zeros. '
                 'This will become an error in a future version (probably 3.0).')
            self.allocateTimes()
        return self._time
    @time.setter
    def time(self, value):
        self.allocateTimes()
        self._time[:] = value

    def hasAllocatedAngles(self):
        """Returns whether the arrays for the incidence angles `dxdz` and `dydz` have been
        allocated.
        """
        return self._dxdz is not None and self._dydz is not None

    def allocateAngles(self):
        """Allocate memory for the incidence angles, `dxdz` and `dydz`.
        """
        if self._dxdz is None:
            self._dxdz = np.zeros_like(self._x)
            self._dydz = np.zeros_like(self._x)
            self.__dict__.pop('_pa', None)

    def hasAllocatedWavelengths(self):
        """Returns whether the `wavelength` array has been allocated.
        """
        return self._wave is not None

    def allocateWavelengths(self):
        """Allocate the memory for the `wavelength` array.
        """
        if self._wave is None:
            self._wave = np.zeros_like(self._x)
            self.__dict__.pop('_pa', None)

    def hasAllocatedPupil(self):
        """Returns whether the arrays for the pupil coordinates `pupil_u` and `pupil_v` have been
        allocated.
        """
        return self._pupil_u is not None and self._pupil_v is not None

    def allocatePupil(self):
        """Allocate the memory for the pupil coordinates, `pupil_u` and `pupil_v`.
        """
        if self._pupil_u is None:
            self._pupil_u = np.zeros_like(self._x)
            self._pupil_v = np.zeros_like(self._x)

    def hasAllocatedTimes(self):
        """Returns whether the array for the time stamps `time` has been allocated.
        """
        return self._time is not None

    def allocateTimes(self):
        """Allocate the memory for the time stamps, `time`.
        """
        if self._time is None:
            self._time = np.zeros_like(self._x)

    def isCorrelated(self):
        """Returns whether the photons are correlated
        """
        from .deprecated import depr
        depr('isCorrelated', 2.5, '',
                "We don't think this is necessary anymore.  If you have a use case that "
                "requires it, please open an issue.")
        return self._is_corr

    def setCorrelated(self, is_corr=True):
        """Set whether the photons are correlated
        """
        from .deprecated import depr
        depr('setCorrelated', 2.5, '',
                "We don't think this is necessary anymore.  If you have a use case that "
                "requires it, please open an issue.")
        self._is_corr = is_corr
        self.__dict__.pop('_pa', None)

    def getTotalFlux(self):
        """Return the total flux of all the photons.
        """
        return self.flux.sum()

    def setTotalFlux(self, flux):
        """Rescale the photon fluxes to achieve the given total flux.

        Parameter:
            flux:       The target flux
        """
        self.scaleFlux(flux / self.getTotalFlux())

    def scaleFlux(self, scale):
        """Rescale the photon fluxes by the given factor.

        Parameter:
            scale:      The factor by which to scale the fluxes.
        """
        self._flux *= scale

    def scaleXY(self, scale):
        """Scale the photon positions (`x` and `y`) by the given factor.

        Parameter:
            scale:      The factor by which to scale the positions.
        """
        self._x *= scale
        self._y *= scale

    def assignAt(self, istart, rhs):
        """Assign the contents of another `PhotonArray` to this one starting at istart.

        Parameters:
            istart:     The first index at which to insert new values.
            rhs:        The other `PhotonArray` from which to get the values.
        """
        from .deprecated import depr
        depr("PhotonArray.assignAt", 2.5, "copyFrom(rhs, slice(istart, istart+rhs.size()))")
        if istart + rhs.size() > self.size():
            raise GalSimValueError(
                "The given rhs does not fit into this array starting at %d"%istart, rhs)
        s = slice(istart, istart + rhs.size())
        self._copyFrom(rhs, s, slice(None))

    def copyFrom(self, rhs, target_indices=slice(None), source_indices=slice(None),
                 do_xy=True, do_flux=True, do_other=True):
        """Copy some contents of another `PhotonArray` to some elements of this one.

        Specifically each element of rhs[source_indices] is mapped to self[target_indices].
        The values s1 and s2 may be slices, list of indices, or anything else that is a valid
        key for a numpy array.

        Parameters:
            rhs:            The `PhotonArray` from which to get values.
            target_indices: The indices at which to assign values in the current PhotonArray (self).
                            [default: slice(None)]
            source_indices: The indices from which to get values from the PhotonArray, ``rhs``.
                            [default: slice(None)]
            do_xy:          Whether to include copying the x and y arrays. [default: True]
            do_flux:        Whether to include copying the flux array. [default: True]
            do_other:       Whether to include copying the other arrays (angles, wavelength,
                            pupil positions, time). [default: True]
        """
        try:
            a1 = self.flux[target_indices]
            # Numpy is flexible about allowing slices outside the range of the array.
            # Rather than try to check all possible ways the indices can be invalid, we
            # just make sure that at least some elements come back from the numpy call.
            n1 = len(np.atleast_1d(a1))
            assert n1 > 0
        except (IndexError, AssertionError):
            raise GalSimValueError("target_indices is invalid for the target PhotonArray",
                                   target_indices)
        try:
            a2 = rhs.flux[source_indices]
            n2 = len(np.atleast_1d(a2))
            assert n2 > 0
        except (IndexError, AssertionError) as e:
            raise GalSimValueError("source_indices is invalid for the source PhotonArray",
                                   source_indices)
        if n1 != n2:
            raise GalSimIncompatibleValuesError(
                "target_indices and source_indices do not reference the same number of elements"
                "in their respective PhotonArrays ({} and {} respectively)".format(n1, n2),
                dict(target_indices=target_indices, source_indices=source_indices))

        self._copyFrom(rhs, target_indices, source_indices, do_xy, do_flux, do_other)

    def _copyFrom(self, rhs, target_indices, source_indices, do_xy=True, do_flux=True,
                  do_other=True):
        """Equivalent to self.copyFrom(rhs, target_indices, source_indices), but without any
        checks that the indices are valid.
        """
        # Aliases for notational convenience.
        s1 = target_indices
        s2 = source_indices

        if do_xy:
            self.x[s1] = rhs.x[s2]
            self.y[s1] = rhs.y[s2]
        if do_flux:
            self.flux[s1] = rhs.flux[s2]
        if do_other and rhs.hasAllocatedAngles():
            self.allocateAngles()
            self.dxdz[s1] = rhs.dxdz[s2]
            self.dydz[s1] = rhs.dydz[s2]
        if do_other and rhs.hasAllocatedWavelengths():
            self.allocateWavelengths()
            self.wavelength[s1] = rhs.wavelength[s2]
        if do_other and rhs.hasAllocatedPupil():
            self.allocatePupil()
            self.pupil_u[s1] = rhs.pupil_u[s2]
            self.pupil_v[s1] = rhs.pupil_v[s2]
        if do_other and rhs.hasAllocatedTimes():
            self.allocateTimes()
            self.time[s1] = rhs.time[s2]

    def convolve(self, rhs, rng=None):
        """Convolve this `PhotonArray` with another.

        ..note::

            If both self and rhs have wavelengths, angles, pupil coordinates or times assigned,
            then the values from the first array (i.e. self) take precedence.
        """
        if rhs.size() != self.size():
            raise GalSimIncompatibleValuesError("PhotonArray.convolve with unequal size arrays",
                                                self_pa=self, rhs=rhs)
        if rhs.hasAllocatedAngles() and not self.hasAllocatedAngles():
            self.dxdz = rhs.dxdz
            self.dydz = rhs.dydz

        if rhs.hasAllocatedWavelengths() and not self.hasAllocatedWavelengths():
            self.wavelength = rhs.wavelength

        if rhs.hasAllocatedPupil() and not self.hasAllocatedPupil():
            self.pupil_u = rhs.pupil_u
            self.pupil_v = rhs.pupil_v

        if rhs.hasAllocatedTimes() and not self.hasAllocatedTimes():
            self.time = rhs.time

        rng = BaseDeviate(rng)
        self._pa.convolve(rhs._pa, rng._rng)

    def __repr__(self):
        s = "galsim.PhotonArray(%d, x=array(%r), y=array(%r), flux=array(%r)"%(
                self.size(), self.x.tolist(), self.y.tolist(), self.flux.tolist())
        if self.hasAllocatedAngles():
            s += ", dxdz=array(%r), dydz=array(%r)"%(self.dxdz.tolist(), self.dydz.tolist())
        if self.hasAllocatedWavelengths():
            s += ", wavelength=array(%r)"%(self.wavelength.tolist())
        if self.hasAllocatedPupil():
            s += ", pupil_u=array(%r), pupil_v=array(%r)"%(self.pupil_u.tolist(), self.pupil_v.tolist())
        if self.hasAllocatedTimes():
            s += ", time=array(%r)"%(self.time.tolist())
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
                 self.hasAllocatedPupil() == other.hasAllocatedPupil() and
                 self.hasAllocatedTimes() == other.hasAllocatedTimes() and
                 (np.array_equal(self.dxdz,other.dxdz) if self.hasAllocatedAngles() else True) and
                 (np.array_equal(self.dydz,other.dydz) if self.hasAllocatedAngles() else True) and
                 (np.array_equal(self.wavelength,other.wavelength)
                    if self.hasAllocatedWavelengths() else True) and
                 (np.array_equal(self.pupil_u,other.pupil_u)
                    if self.hasAllocatedPupil() else True) and
                 (np.array_equal(self.pupil_v,other.pupil_v)
                    if self.hasAllocatedPupil() else True) and
                 (np.array_equal(self.time,other.time)
                    if self.hasAllocatedTimes() else True)
                ))

    def __ne__(self, other):
        return not self == other

    @lazy_property
    def _pa(self):
        #assert(self._x.strides[0] == self._x.itemsize)
        #assert(self._y.strides[0] == self._y.itemsize)
        #assert(self._flux.strides[0] == self._flux.itemsize)
        _x = self._x.__array_interface__['data'][0]
        _y = self._y.__array_interface__['data'][0]
        _flux = self._flux.__array_interface__['data'][0]
        _dxdz = _dydz = _wave = 0
        if self.hasAllocatedAngles():
            #assert(self._dxdz.strides[0] == self._dxdz.itemsize)
            #assert(self._dydz.strides[0] == self._dydz.itemsize)
            _dxdz = self._dxdz.__array_interface__['data'][0]
            _dydz = self._dydz.__array_interface__['data'][0]
        if self.hasAllocatedWavelengths():
            #assert(self._wave.strides[0] == self._wave.itemsize)
            _wave = self._wave.__array_interface__['data'][0]
        return _galsim.PhotonArray(int(self.size()), _x, _y, _flux, _dxdz, _dydz, _wave,
                                   self._is_corr)

    def addTo(self, image):
        """Add flux of photons to an image by binning into pixels.

        Photons in this `PhotonArray` are binned into the pixels of the input
        `Image` and their flux summed into the pixels.  The `Image` is assumed to represent
        surface brightness, so photons' fluxes are divided by image pixel area.
        Photons past the edges of the image are discarded.

        Parameters:
            image:      The `Image` to which the photons' flux will be added.

        Returns:
            the total flux of photons the landed inside the image bounds.
        """
        if not image.bounds.isDefined():
            raise GalSimUndefinedBoundsError(
                "Attempting to PhotonArray::addTo an Image with undefined Bounds")
        return self._pa.addTo(image._image)

    @classmethod
    def makeFromImage(cls, image, max_flux=1., rng=None):
        """Turn an existing `Image` into a `PhotonArray` that would accumulate into this image.

        The flux in each non-zero pixel will be turned into 1 or more photons with random positions
        within the pixel bounds.  The ``max_flux`` parameter (which defaults to 1) sets an upper
        limit for the absolute value of the flux of any photon.  Pixels with abs values > maxFlux
        will spawn multiple photons.

        Parameters:
            image:      The image to turn into a `PhotonArray`
            max_flux:   The maximum flux value to use for any output photon [default: 1]
            rng:        A `BaseDeviate` to use for the random number generation [default: None]

        Returns:
            a `PhotonArray`
        """
        # TODO: This corresponds to the Nearest interpolant.  It would be worth figuring out how
        #       to implement other (presumably better) interpolation options here.

        max_flux = float(max_flux)
        if (max_flux <= 0):
            raise GalSimRangeError("max_flux must be positive", max_flux, 0.)
        total_flux = np.abs(image.array).sum(dtype=float)

        # This goes a bit over what we actually need, but not by much.  Worth it to not have to
        # worry about array reallocations.
        N = int(np.prod(image.array.shape) + total_flux / max_flux)
        photons = cls(N)

        rng = BaseDeviate(rng)
        N = photons._pa.setFrom(image._image, max_flux, rng._rng)
        photons._x = photons.x[:N]
        photons._y = photons.y[:N]
        photons._flux = photons.flux[:N]

        if image.scale != 1. and image.scale is not None:
            photons.scaleXY(image.scale)
        return photons

    def write(self, file_name):
        """Write a `PhotonArray` to a FITS file.

        The output file will be a FITS binary table with a row for each photon in the `PhotonArray`.
        Columns will include 'id' (sequential from 1 to nphotons), 'x', 'y', and 'flux'.
        Additionally, the columns 'dxdz', 'dydz', and 'wavelength' will be included if they are
        set for this `PhotonArray` object.

        The file can be read back in with the classmethod `PhotonArray.read`::

            >>> photons.write('photons.fits')
            >>> photons2 = galsim.PhotonArray.read('photons.fits')

        Parameters:
            file_name:  The file name of the output FITS file.
        """
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

        if self.hasAllocatedPupil():
            cols.append(pyfits.Column(name='pupil_u', format='D', array=self.pupil_u))
            cols.append(pyfits.Column(name='pupil_v', format='D', array=self.pupil_v))

        if self.hasAllocatedTimes():
            cols.append(pyfits.Column(name='time', format='D', array=self.time))

        cols = pyfits.ColDefs(cols)
        table = pyfits.BinTableHDU.from_columns(cols)
        fits.writeFile(file_name, table)

    @classmethod
    def read(cls, file_name):
        """Create a `PhotonArray`, reading the photon data from a FITS file.

        The file being read in is not arbitrary.  It is expected to be a file that was written
        out with the `PhotonArray.write` method.::

            >>> photons.write('photons.fits')
            >>> photons2 = galsim.PhotonArray.read('photons.fits')

        Parameters:
            file_name:  The file name of the input FITS file.
        """
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
        if 'pupil_u' in names:
            photons.pupil_u = data['pupil_u']
            photons.pupil_v = data['pupil_v']
        if 'time' in names:
            photons.time = data['time']
        return photons


class PhotonOp:
    """A base class for photon operators, which just defines the interface.

    Photon operators are designed to apply some physical effect to a bundle of photons.  They
    may adjust the fluxes in some way, or the positions, maybe in a wavelength-dependent way, etc.
    They are typically applied via a ``photon_ops`` argument to the `GSObject.drawImage` method.
    The order typically matters, so the operators are applied in the order they appear in the list.
    """
    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the photon operator to a PhotonArray.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        raise NotImplementedError("Cannot call applyTo on a pure PhotonOp object")

    # These simpler versions of == and hash are fine.
    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class WavelengthSampler(PhotonOp):
    """A photon operator that uses sed.sampleWavelength to set the wavelengths array of a
    `PhotonArray`.

    Parameters:
        sed:        The `SED` to use for the objects spectral energy distribution.
        bandpass:   A `Bandpass` object representing a filter, or None to sample over the full
                    `SED` wavelength range.
        rng:        If provided, a random number generator that is any kind of `BaseDeviate`
                    object. If ``rng`` is None, one will be automatically created, using the
                    time as a seed. [default: None]
        npoints:    Number of points `DistDeviate` should use for its internal interpolation
                    tables. [default: None, which uses the `DistDeviate` default]
    """
    _opt_params = { 'npoints' : int }

    def __init__(self, sed, bandpass=None, rng=None, npoints=None):
        if rng is not None:
            from .deprecated import depr
            depr('WavelengthSampler(..., rng)', 2.3, '',
                 'Instead provide rng when calling applyTo, drawImage, etc.')
        self.sed = sed
        self.bandpass = bandpass
        self.rng = rng
        self.npoints = npoints

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Assign wavelengths to the photons sampled from the SED * Bandpass.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        rng = rng if rng is not None else self.rng
        if photon_array.hasAllocatedWavelengths():
            galsim_warn("Wavelengths already set before applying WavelengthSampler. "
                        "This is most likely an error.")
        photon_array.wavelength = self.sed.sampleWavelength(
                photon_array.size(), self.bandpass, rng=rng, npoints=self.npoints)

    def __str__(self):
        return "galsim.WavelengthSampler(sed=%s, bandpass=%s, rng=%s, npoints=%s)"%(
            self.sed, self.bandpass, self.rng, self.npoints)

    def __repr__(self):
        return "galsim.WavelengthSampler(sed=%r, bandpass=%r, rng=%r, npoints=%r)"%(
            self.sed, self.bandpass, self.rng, self.npoints)


class FRatioAngles(PhotonOp):
    """A photon operator that assigns photon directions based on the f/ratio and obscuration.

    Assigns arrival directions at the focal plane for photons, drawing from a uniform
    brightness distribution between the obscuration angle and the edge of the pupil defined
    by the f/ratio of the telescope.  The angles are expressed in terms of slopes dx/dz
    and dy/dz.

    Parameters:
        fratio:         The f/ratio of the telescope (e.g. 1.2 for LSST)
        obscuration:    Linear dimension of central obscuration as fraction of aperture
                        linear dimension. [0., 1.).  [default: 0.0]
        rng:            A random number generator to use or None, in which case an rng
                        will be automatically constructed for you. [default: None]
    """
    _req_params = { 'fratio' : float }
    _opt_params = { 'obscuration' : float }

    def __init__(self, fratio, obscuration=0.0, rng=None):
        if fratio < 0:
            raise GalSimRangeError("The f-ratio must be positive.", fratio, 0.)
        if obscuration < 0 or obscuration >= 1:
            raise GalSimRangeError("Invalid obscuration.", obscuration, 0., 1.)
        if rng is not None:
            from .deprecated import depr
            depr('FRatioAngles(..., rng)', 2.3, '',
                 'Instead provide rng when calling applyTo, drawImage, etc.')

        self.fratio = fratio
        self.obscuration = obscuration
        self.rng = rng


    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Assign directions to the photons in photon_array.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        gen = BaseDeviate(rng).as_numpy_generator()

        n_photons = len(photon_array)

        # The f/ratio is the ratio of the focal length to the diameter of the aperture of
        # the telescope.  The angular radius of the field of view is defined by the
        # ratio of the radius of the aperture to the focal length
        pupil_angle = np.arctan(0.5 / self.fratio)  # radians
        obscuration_angle = np.arctan(0.5 * self.obscuration / self.fratio)

        # Generate azimuthal angles for the photons
        phi = gen.uniform(0, 2*np.pi, size=n_photons)

        # Generate inclination angles for the photons, which are uniform in sin(theta) between
        # the sine of the obscuration angle and the sine of the pupil radius
        sintheta = gen.uniform(np.sin(obscuration_angle), np.sin(pupil_angle), size=n_photons)

        # Assign the directions to the arrays. In this class the convention for the
        # zero of phi does not matter but it would if the obscuration is dependent on
        # phi
        tantheta = np.sqrt(np.square(sintheta) / (1. - np.square(sintheta)))
        photon_array.dxdz = tantheta * np.sin(phi)
        photon_array.dydz = tantheta * np.cos(phi)

    def __str__(self):
        return "galsim.FRatioAngles(fratio=%s, obscration=%s, rng=%s)"%(
            self.fratio, self.obscuration, self.rng)

    def __repr__(self):
        return "galsim.FRatioAngles(fratio=%r, obscration=%r, rng=%r)"%(
            self.fratio, self.obscuration, self.rng)


class PhotonDCR(PhotonOp):
    r"""A photon operator that applies the effect of differential chromatic refraction (DCR)
    and optionally the chromatic dilation due to atmospheric seeing.

    Due to DCR, blue photons land closer to the zenith than red photons.  Kolmogorov turbulence
    also predicts that blue photons get spread out more by the atmosphere than red photons,
    specifically FWHM is proportional to :math:`\lambda^{-0.2}`.  Both of these effects can be
    implemented by wavelength-dependent shifts of the photons.

    Since DCR depends on the zenith angle and the parallactic angle (which is the position angle of
    the zenith measured from North through East) of the object being drawn, these must be specified
    via keywords.  There are four ways to specify these values:

    1) explicitly provide ``zenith_angle`` as a keyword of type `Angle`, and ``parallactic_angle``
       will be assumed to be 0 by default.
    2) explicitly provide both ``zenith_angle`` and ``parallactic_angle`` as keywords of type
       `Angle`.
    3) provide the coordinates of the object ``obj_coord`` and the coordinates of the zenith
       ``zenith_coord`` as keywords of type `CelestialCoord`.
    4) provide the coordinates of the object ``obj_coord`` as a `CelestialCoord`, the hour angle
       of the object ``HA`` as an `Angle`, and the latitude of the observer ``latitude`` as an
       `Angle`.

    DCR also depends on temperature, pressure and water vapor pressure of the atmosphere.  The
    default values for these are expected to be appropriate for LSST at Cerro Pachon, Chile, but
    they are broadly reasonable for most observatories.

    This photon op is intended to match the functionality of `ChromaticAtmosphere`, but acting
    on the photon array rather than as a `ChromaticObject`.  The photons will need to have
    wavelengths defined in order to work.

    .. warning::
        The alpha parameter is only appropriate for stars.  This photon op will act on
        all of the photons, so applying a chromatic dilation according to the chromatic
        seeing is the wrong thing to do when the surface brightness being rendered is
        not a pure PSF.  As such, the default is alpha=0, not -0.2, which would be
        appropriate for Kolmogorov turbulence.

    Parameters:
        base_wavelength:    Wavelength (in nm) represented by the fiducial photon positions
        scale_unit:         Units used for the positions of the photons.  [default: galsim.arcsec]
        alpha:              Power law index for wavelength-dependent seeing.  This should only
                            be used if doing a star-only simulation.  It is not correct when
                            drawing galaxies. [default: 0.]
        zenith_angle:       `Angle` from object to zenith, expressed as an `Angle`. [default: 0]
        parallactic_angle:  Parallactic angle, i.e. the position angle of the zenith, measured
                            from North through East.  [default: 0]
        obj_coord:          Celestial coordinates of the object being drawn as a `CelestialCoord`.
                            [default: None]
        zenith_coord:       Celestial coordinates of the zenith as a `CelestialCoord`.
                            [default: None]
        HA:                 Hour angle of the object as an `Angle`. [default: None]
        latitude:           Latitude of the observer as an `Angle`. [default: None]
        pressure:           Air pressure, either as an astropy Quantity or a float in units of
                            kiloPascals.  [default: 69.328 kPa]
        temperature:        Temperature, either as an astropy Quantity or a float in units of
                            Kelvin.  [default: 293.15 K]
        H2O_pressure:       Water vapor pressure, either as an astropy Quantity or a float in units
                            of kiloPascals.  [default: 1.067 kPa]
    """
    _req_params = { 'base_wavelength' : float }
    _opt_params = { 'scale_unit' : str,
                    'alpha' : float,
                    'parallactic_angle' : Angle,
                    'latitude' : Angle,
                    'pressure' : (float, u.Quantity),
                    'temperature' : (float, u.Quantity),
                    'H2O_pressure' : (float, u.Quantity)
                  }
    _single_params = [ { 'zenith_angle' : Angle, 'HA' : Angle, 'zenith_coord' : CelestialCoord } ]

    def __init__(self, base_wavelength, scale_unit=arcsec, **kwargs):
        # This matches the code in ChromaticAtmosphere.
        self.base_wavelength = base_wavelength

        if isinstance(scale_unit, str):
            scale_unit = AngleUnit.from_name(scale_unit)
        self.scale_unit = scale_unit
        self.alpha = kwargs.pop('alpha', 0.)

        self.zenith_angle, self.parallactic_angle, self.kw = dcr.parse_dcr_angles(**kwargs)
        # Convert any weather data to the appropriate units
        p = self.kw.get('pressure', None)
        if p is not None and isinstance(p, u.Quantity):
            self.kw['pressure'] = p.to_value(u.kPa)
        t = self.kw.get('temperature', None)
        if t is not None and isinstance(t, u.Quantity):
            self.kw['temperature'] = t.to_value(u.K)
        h = self.kw.get('H2O_pressure', None)
        if h is not None and isinstance(h, u.Quantity):
            self.kw['H2O_pressure'] = h.to_value(u.kPa)

        # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
        # Check that they're valid
        for kw in self.kw:
            if kw not in ('temperature', 'pressure', 'H2O_pressure'):
                raise TypeError("Got unexpected keyword: {0}".format(kw))

        self.base_refraction = dcr.get_refraction(self.base_wavelength, self.zenith_angle,
                                                  **self.kw)

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the DCR effect to the photons

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("PhotonDCR requires that wavelengths be set")
        if local_wcs is None:
            raise TypeError("PhotonDCR requires a local_wcs to be provided to applyTo")

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
        shift_magnitude *= radians / self.scale_unit
        sinp, cosp = self.parallactic_angle.sincos()

        du = -shift_magnitude * sinp
        dv = shift_magnitude * cosp

        dx = local_wcs._x(du, dv)
        dy = local_wcs._y(du, dv)
        photon_array.x += dx
        photon_array.y += dy

    def __repr__(self):
        s = "galsim.PhotonDCR(base_wavelength=%r, scale_unit=%r, alpha=%r, "%(
                self.base_wavelength, self.scale_unit, self.alpha)
        s += "zenith_angle=%r, parallactic_angle=%r"%(self.zenith_angle, self.parallactic_angle)
        for k in sorted(self.kw):
            s += ", %s=%r"%(k, self.kw[k])
        s += ")"
        return s


class Refraction(PhotonOp):
    """A photon operator that refracts photons (manipulating their dxdz and dydz values) at an
    interface, commonly the interface between vacuum and silicon at the surface of a CCD.

    Assumes that the surface normal is along the z-axis.  If the refraction would result in total
    internal reflection, then those photon's dxdz and dydz values are set to NaN, and flux values
    set to 0.0.

    Parameters:
        index_ratio:    The ratio of the refractive index on the far side of the interface to the
                        near side.  Can be given as a number or a callable function.  In the
                        latter case, the function should accept a numpy array of vacuum wavelengths
                        as input and return a numpy array of refractive index ratios.
    """
    _req_params = { 'index_ratio' : float }

    def __init__(self, index_ratio):
        self.index_ratio = index_ratio

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Refract photons

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        if hasattr(self.index_ratio, '__call__'):
            index_ratio = self.index_ratio(photon_array.wavelength)
        else:
            index_ratio = self.index_ratio
        # Here's the math avoiding any actual trig function calls:
        #
        #        x1 = dr/dz
        #        ------+
        #         \    |
        #          \   |
        #           \  |  dz/dz = 1
        #            \ |
        #             \|         n1
        #  ------------------------
        #              |\        n2
        #              |\
        # dz'/dz' = 1  | \
        #              | \
        #              |  \
        #              +---
        #              x2 = dr'/dz'
        #
        # Solve Snell's law for x2 as fn of x1:
        #   n1 sin(th1) = n2 sin(th2)
        #   n1 x1 / sqrt(1 + x1^2) = n2 x2 / sqrt(1 + x2^2)
        #   n1^2 x1^2 (1 + x2^2) = n2^2 x2^2 (1 + x1^2)
        #   n1^2 x1^2 = x2^2 (n2^2 (1 + x1^2) - n1^2 x1^2)
        #   x1^2 = x2^2 ((n2/n1)^2 (1 + x1^2) - x1^2)
        #   x1 = x2 sqrt( (n2/n1)^2 (1 + x1^2) - x1^2 )
        #      = x2 sqrt( (n2/n1)^2 (1 + x1^2) - (1 + x1^2) + 1 )
        #      = x2 sqrt( 1 - (1 + x1^2) (1 - (n2/n1)^2) )
        normsqr = 1 + photon_array.dxdz**2 + photon_array.dydz**2  # (1 + x1^2)
        with np.errstate(invalid='ignore'):
            # NaN below <=> total internal reflection
            factor = np.sqrt(1 - normsqr*(1-index_ratio**2))
        photon_array.dxdz /= factor
        photon_array.dydz /= factor
        photon_array.flux = np.where(np.isnan(factor), 0.0, photon_array.flux)

    def __repr__(self):
        return "galsim.Refraction(index_ratio=%r)"%self.index_ratio


class FocusDepth(PhotonOp):
    """A photon operator that focuses/defocuses photons by changing the height of the focal
    surface with respect to the conical beam.

    Parameters:
        depth:   The z-distance by which to displace the focal surface, in units of pixels.  A
                 positive (negative) number here indicates an extra- (intra-) focal sensor height.
                 I.e., depth > 0 means the sensor surface intersects the beam after it has
                 converged, and depth < 0 means the sensor surface intersects the beam before it
                 has converged.
    """
    _req_params = { 'depth' : float }

    def __init__(self, depth):
        self.depth = depth

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Adjust a photon bundle to account for the change in focal depth.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        if not photon_array.hasAllocatedAngles():
            raise GalSimError("FocusDepth requires that angles be set")
        photon_array.x += self.depth * photon_array.dxdz
        photon_array.y += self.depth * photon_array.dydz

    def __repr__(self):
        return "galsim.FocusDepth(depth=%r)"%self.depth


class PupilImageSampler(PhotonOp):
    """A photon operator that samples the pupil-plane positions given a pupil-plane image.
    Samples are drawn discretely from pupil plane image pixels marked as illuminated.

    Parameters:
        diam:               Aperture diameter in meters.
        lam:                Wavelength in nanometers.  [default: None]
        circular_pupil:     Adopt a circular pupil?  [default: True]
        obscuration:        Linear dimension of central obscuration as fraction of aperture
                            linear dimension. [0., 1.).  [default: 0.0]
        nstruts:            Number of radial support struts to add to the central obscuration.
                            [default: 0]
        strut_thick:        Thickness of support struts as a fraction of aperture diameter.
                            [default: 0.05]
        strut_angle:        `Angle` made between the vertical and the strut starting closest to it,
                            defined to be positive in the counter-clockwise direction; must be an
                            `Angle` instance. [default: 0. * galsim.degrees]
        oversampling:       Optional oversampling factor *in the image plane* for the PSF
                            eventually constructed using this `Aperture`.  Setting
                            ``oversampling < 1`` will produce aliasing in the PSF (not good).
                            [default: 1.0]
        pad_factor:         Additional multiple by which to extend the PSF image to avoid
                            folding.  [default: 1.0]
        pupil_plane_im:     The GalSim.Image, NumPy array, or name of file containing the pupil
                            plane image, to be used instead of generating one based on the
                            obscuration and strut parameters.  [default: None]
        pupil_angle:        If ``pupil_plane_im`` is not None, rotation angle for the pupil plane
                            (positive in the counter-clockwise direction).  Must be an `Angle`
                            instance. [default: 0. * galsim.degrees]
        pupil_plane_scale:  Sampling interval in meters to use for the pupil plane array.  In
                            most cases, it's a good idea to leave this as None, in which case
                            GalSim will attempt to find a good value automatically.  The
                            exception is when specifying the pupil arrangement via an image, in
                            which case this keyword can be used to indicate the sampling of that
                            image.  See also ``pad_factor`` for adjusting the pupil sampling scale.
                            [default: None]
        pupil_plane_size:   Size in meters to use for the pupil plane array.  In most cases, it's
                            a good idea to leave this as None, in which case GalSim will attempt
                            to find a good value automatically.  See also ``oversampling`` for
                            adjusting the pupil size.  [default: None]
    """
    _req_params = {
        "diam": float,
    }
    _opt_params = {
        "lam": float,
        "circular_pupil": bool,
        "obscuration": float,
        "nstruts": int,
        "strut_thick": float,
        "strut_angle": Angle,
        "oversampling": float,
        "pad_factor": float,
        "pupil_plane_im": str,
        "pupil_angle": Angle,
        "pupil_plane_scale": float,
        "pupil_plane_size": float,
    }
    def __init__(self, diam, **kwargs):
        from .phase_psf import Aperture
        self.aper = Aperture(diam, **kwargs)
        # Save these for the repr
        self.diam = diam
        self.kwargs = kwargs

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Sample the pupil plane u,v positions for each photon.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        self.aper.samplePupil(photon_array, rng)

    def __repr__(self):
        s =  "galsim.PupilImageSampler(diam=%s"%self.diam
        for k,v in self.kwargs.items():
            s += ', %s=%r'%(k,v)
        s += ')'
        return s


class PupilAnnulusSampler(PhotonOp):
    """A photon operator that uniformly samples an annular entrance pupil.

    Parameters:
        R_outer:  Annulus outer radius in meters.
        R_inner:  Annulus inner radius in meters.  [default: 0.0]
    """
    _req_params = {
        "R_outer": float,
    }
    _opt_params = {
        "R_inner": float,
    }
    def __init__(self, R_outer, R_inner=0.0):
        self.R_outer = R_outer
        self.R_inner = R_inner

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Sample the pupil plane u,v positions for each photon.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        gen = BaseDeviate(rng).as_numpy_generator()
        r = gen.uniform(self.R_inner**2, self.R_outer**2, size=len(photon_array))
        np.sqrt(r, out=r)
        phi = gen.uniform(0, 2*np.pi, size=len(photon_array))
        photon_array.pupil_u = r * np.cos(phi)
        photon_array.pupil_v = r * np.sin(phi)

    def __repr__(self):
        s = "galsim.PupilAnnulusSampler(R_outer=%r"%self.R_outer
        if self.R_inner != 0.0:
            s += ", R_inner=%r"%self.R_inner
        s += ")"
        return s


class TimeSampler(PhotonOp):
    """A photon operator that uniformly samples photon time stamps within some interval.

    Parameters:
        t0:         The nominal start time of the observation in seconds. [default: 0]
        exptime:    The exposure time in seconds. [default: 0]
    """
    _opt_params = {
        "t0": float,
        "exptime": float
    }
    def __init__(self, t0=0.0, exptime=0.0):
        self.t0 = t0
        self.exptime = exptime

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Add time stamps to photons.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """
        gen = BaseDeviate(rng).as_numpy_generator()
        photon_array.time = gen.uniform(self.t0, self.t0+self.exptime, size=len(photon_array))

    def __repr__(self):
        s = "galsim.TimeSampler("
        if self.t0 != 0.0:
            s += "t0=%r"%self.t0
            if self.exptime != 0.0:
                s += ", exptime=%r"%self.exptime
        else:
            if self.exptime != 0.0:
                s += "exptime=%r"%self.exptime
        s += ")"
        return s


class ScaleFlux(PhotonOp):
    """A simple photon operator that multiplies all flux values by a constant.

    Parameters:
        x:          The constant by which to multiply all flux values.
    """
    _req_params = { "x": float }
    def __init__(self, x):
        self.x = x

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the scaling.
        """
        photon_array.flux *= self.x

    def __repr__(self):
        return f"galsim.ScaleFlux({self.x})"


class ScaleWavelength(PhotonOp):
    """A simple photon operator that multiplies all wavelength values by a constant.

    Parameters:
        x:          The constant by which to multiply all wavelength values.
    """
    _req_params = { "x": float }
    def __init__(self, x):
        self.x = x

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the scaling.
        """
        photon_array.wavelength *= self.x

    def __repr__(self):
        return f"galsim.ScaleWavelength({self.x})"


# Put these at the end to avoid circular imports
from . import fits
