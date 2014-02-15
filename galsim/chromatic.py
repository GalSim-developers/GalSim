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
"""@file chromatic.py
Define wavelength-dependent surface brightness profiles.

Implementation is done by constructing GSObjects as functions of wavelength.  `draw` methods then
integrate over wavelength while also multiplying in a throughput function.

Possible uses include galaxies with color gradients, automatically drawing a given galaxy through
different filters, or implementing wavelength-dependent point spread functions.
"""

import numpy
import copy

import galsim
import galsim.integ
import galsim.dcr

class ChromaticObject(object):
    """Base class for defining wavelength dependent objects."""
    def __init__(self, gsobj):
        """ Create the simplest type of ChromaticObject of all, which is just a wrapper around a
        GSObject.  At this point, the newly created ChromaticObject will act just like the GSObject
        it wraps.  The difference is that its methods: applyExpansion, applyDilation, and applyShift
        can now accept functions of wavelength as arguments, as opposed to the simple scalars that
        GSObjects are limited to.  These methods can be used to effect a variety of physical chromatic
        effects, such as differential chromatic refraction, chromatic seeing, and diffraction-limited
        seeing.

        @param gsobj  The GSObject to be chromaticized.
        @returns      The chromaticized GSObject as a ChromaticObject.
        """
        if not isinstance(gsobj, galsim.GSObject):
            raise TypeError("Can only directly instantiate ChromaticObject with a GSObject "+
                            "argument.")
        self.obj = gsobj.copy()
        self.A = lambda w: numpy.matrix(numpy.identity(3), dtype=float)
        self.fluxFactor = lambda w: 1.0
        self.separable = True
        self.SED = lambda w: 1.0

    def draw(self, bandpass, image=None, scale=None, gain=1.0, wmult=1.0,
             add_to_image=False, use_true_center=True, offset=None,
             integrator=None, **kwargs):
        """Base chromatic image draw method.  Some subclasses may choose to override this for
        specific efficiency gains.  For instance, most GalSim use cases will probably finish with
        a convolution, in which case ChromaticConvolution.draw() will be used.

        The task of ChromaticObject.draw(bandpass) is to integrate a chromatic surface brightness
        profile multiplied by the throughput of `bandpass`, over the wavelength interval indicated
        by `bandpass`.  Several integrators are available in galsim.integ to do this integration.
        By default, galsim.integ.trapz_sample_integrator will be used if `bandpass.wave_list`
        exists, and galsim.integ.trapz_continuous_integrator will be used otherwise.  Both of these
        functions use the Trapezoidal rule.  The difference is that `trapz_sample_integrator`
        evaluates the integrand at precisely the wavelengths defined by `bandpass.wave_list`, while
        `trapz_continuous_integrator` evaluates the integrand at the midpoints of N equally sized
        subintervals in the interval between `bandpass.blue_limit` and `bandpass.red_limit`.  In
        the latter case, the number of subintervals N is an optional argument to
        `ChromaticObject.draw`.  Other available integrators include `midpt_sample_integrator` and
        `midpt_continuous_integrator`, which replace the Trapezoidal rule above with the midpoint
        rule.

        @param bandpass           A galsim.Bandpass object representing the filter
                                  against which to integrate.
        @param image              see GSObject.draw()
        @param scale              see GSObject.draw()
        @param gain               see GSObject.draw()
        @param wmult              see GSObject.draw()
        @param add_to_image       see GSObject.draw()
        @param use_true_center    see GSObject.draw()
        @param offset             see GSObject.draw()
        @param integrator         One of the image integrators from galsim.integ

        @returns                  galsim.Image drawn through filter.
        """

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength)
        prof1 = prof0._fix_center(image, scale, offset, use_true_center, reverse=False)
        image = prof1._draw_setup_image(image, scale, wmult, add_to_image)

        if self.separable:
            multiplier = galsim.integ.int1d(lambda w: self.SED(w) * bandpass(w),
                                            bandpass.blue_limit, bandpass.red_limit)
            prof0 *= multiplier/self.SED(bandpass.effective_wavelength)
            prof0.draw(image=image, gain=gain, wmult=wmult,
                       add_to_image=add_to_image, offset=offset)
            return image

        # decide on integrator
        if integrator is None:
            if hasattr(bandpass, 'wave_list'):
                integrator = galsim.integ.trapz_sample_integrator
            else:
                integrator = galsim.integ.trapz_continuous_integrator

        integral = integrator(self.evaluateAtWavelength, bandpass, image, gain, wmult,
                              use_true_center, offset, **kwargs)

        # Clear image if add_to_image is False
        if not add_to_image:
            image.setZero()
        image += integral
        return image

    def _getScaleEtaBetaThetaDxDy(self, w):
        A0 = self.A(w)
        A = A0[0,0]
        B = A0[0,1]
        C = A0[1,0]
        D = A0[1,1]
        dx = A0[0,2]
        dy = A0[1,2]
        scale = numpy.sqrt(numpy.linalg.det(A0))
        theta = numpy.arctan2(C-B, A+D) #need to worry about A+D == 0 ?
        if A-D == 0.0:
            eta = 0.0
            beta = 0.0
        else:
            beta = 0.5 * (numpy.arctan2(C+B, A-D) + theta)
            eta = 2.0*numpy.arcsinh((A-D)/(2.0*scale*numpy.cos(2.0*beta-theta)))
        if eta < 0.0:
            eta = -eta
            beta += numpy.pi/2.0
        return scale, eta, beta, theta, dx, dy

    def evaluateAtWavelength(self, w):
        if not hasattr(self, 'A'):
            raise AttributeError("Attempting to evaluate ChromaticObject before affine transform " +
                                 "matrix has been created!")
        scale, eta, beta, theta, dx, dy = self._getScaleEtaBetaThetaDxDy(w)
        tmpobj = self.obj.evaluateAtWavelength(w).copy()
        tmpobj.applyRotation(theta * galsim.radians)
        tmpobj.applyShear(eta=eta, beta=beta*galsim.radians)
        tmpobj.applyExpansion(scale)
        tmpobj.applyShift(dx, dy)
        tmpobj.scaleFlux(self.fluxFactor(w))
        return tmpobj

    def scaleFlux(self, scale):
        if hasattr(scale, '__call__'):
            fluxFactor = self.fluxFactor
            self.fluxFactor = lambda w: fluxFactor(w) * scale(w)
        else:
            self.obj.scaleFlux(scale)

    # Subclasses must override scaleFlux() and evaluateAtWavelength()
    # def scaleFlux(self, scale):
    #     raise NotImplementedError

    # def evaluateAtWavelength(self, wave):
    #     raise NotImplementedError

    # Add together `ChromaticObject`s and/or `GSObject`s
    def __add__(self, other):
        return galsim.ChromaticSum([self, other])

    # Subtract `ChromaticObject`s and/or `GSObject`s
    def __sub__(self, other):
        return galsim.ChromaticSum([self, (-1. * other)])

    # Make op* and op*= work to adjust the flux of the object
    def __imul__(self, other):
        self.scaleFlux(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # Make a new copy of a `ChromaticObject`.
    def copy(self):
        """Returns a copy of an object.  This preseved the original type of the object."""
        cls = self.__class__
        ret = cls.__new__(cls)
        for k, v in self.__dict__.iteritems():
            if k == 'objlist':
                # explicity request that individual items of objlist are copied,
                # not just the list itself
                ret.__dict__[k] = [o.copy() for o in v]
            else:
                ret.__dict__[k] = copy.copy(v)
        return ret

    # Following functions work to apply affine transformations to a ChromaticObject.
    #

    def applyExpansion(self, scale):
        """Rescale the linear size of the profile by the given (possibly wavelength-dependent)
        scale factor.

        This doesn't correspond to either of the normal operations one would typically want to
        do to a galaxy.  See the functions applyDilation and applyMagnification for the more
        typical usage.  But this function is conceptually simple.  It rescales the linear
        dimension of the profile, while preserving surface brightness.  As a result, the flux
        will necessarily change as well.

        See applyDilation for a version that applies a linear scale factor in the size while
        preserving flux.

        See applyMagnification for a version that applies a scale factor to the area while
        preserving surface brightness.

        After this call, the caller's type will be a ChromaticObject.

        @param scale The factor by which to scale the linear dimension of the object.  `scale` may
                     be callable, in which case the argument should be wavelength in nanometers and
                     the return value the scale for that wavelength.
        """
        if isinstance(self, ChromaticSum):
            for obj in self.objlist:
                obj.applyExpansion(scale)
        else:
            if hasattr(scale, '__call__'):
                expand = lambda w: numpy.matrix(numpy.diag([scale(w), scale(w), 1]))
                self.separable = False
            else:
                expand = lambda w: numpy.matrix(numpy.diag([scale, scale, 1]))
            if hasattr(self, 'A'):
                A = self.A
                self.A = lambda w:expand(w) * A(w)
            else:
                self.obj = self.copy()
                self.__class__ = ChromaticObject
                self.A = expand
                self.fluxFactor = lambda w: 1.0
                self.separable = self.obj.separable
                if hasattr(self, 'gsobj'):
                    del self.gsobj
                # note, self.SED should already exist if this is a separable object

    def applyDilation(self, scale):
        """Apply a dilation of the linear size by the given (possibly wavelength-dependent) scale.

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves flux.
        See applyMagnification() for a version that preserves surface brightness, and thus
        changes the flux.

        After this call, the caller's type will be a ChromaticObject.

        @param scale The linear rescaling factor to apply.  `scale` may be callable, in which case
                     the argument should be wavelength in nanometers and the return value the scale
                     for that wavelength.
        """
        self.applyExpansion(scale)
        # conserve flux
        if hasattr(scale, '__call__'):
            self.scaleFlux(lambda w: 1./scale(w)**2)
        else:
            self.scaleFlux(1./scale**2)

    def applyMagnification(self, mu):
        """Apply a lensing magnification, scaling the area and flux by mu at fixed surface
        brightness.

        This process applies a lensing magnification mu, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., `half_light_radius` <-- `half_light_radius * sqrt(mu)`
        while increasing the flux by a factor of mu.  Thus, applyMagnification preserves surface
        brightness.

        See applyDilation for a version that applies a linear scale factor in the size while
        preserving flux.

        After this call, the caller's type will be a ChromaticObject.

        @param mu The lensing magnification to apply.
        """

        self.applyExpansion(numpy.sqrt(mu))

    def applyShear(self, *args, **kwargs):
        """Apply an area-preserving shear to this object, where arguments are either a galsim.Shear,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        The applyShear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use applyShear() with applyMagnification(), or use
        applyLensing() which combines both operations.

        After this call, the caller's type will be a ChromaticObject.
        """
        if isinstance(self, ChromaticSum):
            for obj in self.objlist:
                obj.applyShear(*args, **kwargs)
        else:
            if len(args) == 1:
                if kwargs:
                    raise TypeError("Error, gave both unnamed and named arguments to applyShear!")
                if not isinstance(args[0], galsim.Shear):
                    raise TypeError("Error, unnamed argument to applyShear is not a Shear!")
                shear = args[0]
            elif len(args) > 1:
                raise TypeError("Error, too many unnamed arguments to applyShear!")
            else:
                shear = galsim.Shear(**kwargs)
            c2b = numpy.cos(2.0 * shear.beta.rad())
            s2b = numpy.sin(2.0 * shear.beta.rad())
            ce2 = numpy.cosh(shear.eta/2.0)
            se2 = numpy.sinh(shear.eta/2.0)
            #Bernstein&Jarvis (2002) equation 2.9
            S = numpy.matrix([[ce2+c2b*se2, s2b*se2, 0],
                              [s2b*se2, ce2-c2b*se2, 0],
                              [0, 0, 1]], dtype=float)
            if hasattr(self, 'A'):
                A = self.A
                self.A = lambda w: S * A(w)
            else:
                self.obj = self.copy()
                self.__class__ = ChromaticObject
                self.A = lambda w: S
                self.fluxFactor = lambda w: 1.0
                self.separable = self.obj.separable
                if hasattr(self, 'gsobj'):
                    del self.gsobj
                # note, self.SED should already exist if this is a separable object

    def applyLensing(self, g1, g2, mu):
        """Apply a lensing shear and magnification to this object.

        This ChromaticObject method applies a lensing (reduced) shear and magnification.  The shear
        must be specified using the g1, g2 definition of shear (see galsim.Shear documentation for
        more details).  This is the same definition as the outputs of the galsim.PowerSpectrum and
        galsim.NFWHalo classes, which compute shears according to some lensing power spectrum or
        lensing by an NFW dark matter halo.  The magnification determines the rescaling factor for
        the object area and flux, preserving surface brightness.

        After this call, the caller's type will be a ChromaticObject.

        @param g1      First component of lensing (reduced) shear to apply to the object.
        @param g2      Second component of lensing (reduced) shear to apply to the object.
        @param mu      Lensing magnification to apply to the object.  This is the factor by which
                       the solid angle subtended by the object is magnified, preserving surface
                       brightness.
        """
        self.applyShear(g1=g1, g2=g2)
        self.applyMagnification(mu)

    def applyRotation(self, theta):
        """Apply a rotation theta to this object.

        After this call, the caller's type will be a ChromaticObject.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        """
        if isinstance(self, ChromaticSum):
            for obj in self.objlist:
                obj.applyRotation(theta)
        else:
            cth = numpy.cos(theta.rad())
            sth = numpy.sin(theta.rad())
            R = numpy.matrix([[cth, -sth, 0],
                              [sth, cth, 0],
                              [0, 0, 1]], dtype=float)
            if hasattr(self, 'A'):
                A = self.A
                self.A = lambda w: R * A(w)
            else:
                self.obj = self.copy()
                self.__class__ = ChromaticObject
                self.A = lambda w: R
                self.fluxFactor = lambda w: 1.0
                self.separable = self.obj.separable
                if hasattr(self, 'gsobj'):
                    del self.gsobj
                # note, self.SED should already exist if this is a separable object

    def applyShift(self, *args, **kwargs):
        """Apply a (possibly wavelength-dependent) (dx, dy) shift to this chromatic object.

        After this call, the caller's type will be a ChromaticObject.

        @param dx Horizontal shift to apply (float).
        @param dy Vertical shift to apply (float).

        Note:
        You may supply dx,dy as either two arguments, as a tuple, as a
        galsim.PositionD or galsim.PositionI object, or finally, as a callable object that accepts
        wavelength in nanometers as its argument and returns one of 2-tuple, galsim.PositionD, or
        galsim.PositionI as its result representing a wavelength-dependent shift.
        """
        if isinstance(self, ChromaticSum):     # Don't wrap `ChromaticSum`s, easier to just
            for obj in self.objlist:           # wrap their arguments.
                obj.applyShift(*args, **kwargs)
        else:
            # First unpack args/kwargs
            if len(args) == 0:
                # Then dx,dy need to be kwargs
                # If not, then python will raise an appropriate error.
                dx = kwargs.pop('dx')
                dy = kwargs.pop('dy')
            elif len(args) == 1:
                if hasattr(args[0], '__call__'):
                    def dx(w):
                        try:
                            return args[0](w).x
                        except:
                            return args[0](w)[0]
                    def dy(w):
                        try:
                            return args[0](w).y
                        except:
                            return args[0](w)[1]
                    self.separable = False
                elif isinstance(args[0], galsim.PositionD) or isinstance(args[0], galsim.PositionI):
                    dx = args[0].x
                    dy = args[0].y
                else:
                    # Let python raise the appropriate exception if this isn't valid.
                    dx = args[0][0]
                    dy = args[0][1]
            elif len(args) == 2:
                dx = args[0]
                dy = args[1]
            else:
                raise TypeError("Too many arguments supplied to applyShift ")
            if kwargs:
                raise TypeError("applyShift() got unexpected keyword arguments: %s",kwargs.keys())
            # Then create augmented affine transform matrix and multiply or set as necessary
            if hasattr(dx, '__call__'):
                shift = lambda w: numpy.matrix([[1,0,dx(w)],[0,1,dy(w)],[0,0,1]], dtype=float)
            else:
                shift = lambda w: numpy.matrix([[1,0,dx],[0,1,dy],[0,0,1]], dtype=float)
            if hasattr(self, 'A'):
                A = self.A
                self.A = lambda w: shift(w) * A(w)
            else:
                self.obj = self.copy()
                self.__class__ = ChromaticObject
                self.A = shift
                self.fluxFactor = lambda w: 1.0
                self.separable = self.obj.separable
                if hasattr(self, 'gsobj'):
                    del self.gsobj
                # note, self.SED should already exist if this is a separable object

    # Also add methods which create a new ChromaticObject with the transformations
    # applied...
    def createExpanded(self, scale):
        """Returns a new ChromaticObject by applying an expansion of the linear size by the
        given scale.

        This doesn't correspond to either of the normal operations one would typically want to
        do to a galaxy.  See the functions createDilated() and createMagnified() for the more
        typical usage.  But this function is conceptually simple.  It rescales the linear
        dimension of the profile, while preserving surface brightness.  As a result, the flux
        will necessarily change as well.

        See createDilated() for a version that applies a linear scale factor in the size while
        preserving flux.

        See createMagnified() for a version that applies a scale factor to the area while
        preserving surface brightness.

        @param scale The linear rescaling factor to apply.
        @returns The rescaled ChromaticObject
        """
        ret = self.copy()
        ret.applyExpansion(scale)
        return ret

    def createDilated(self, scale):
        """Returns a new ChromaticObject by applying a dilation of the linear size by the
        given scale.

        Scales the linear dimensions of the image by the factor scale.
        e.g. `half_light_radius` <-- `half_light_radius * scale`

        This operation preserves flux.
        See createMagnified() for a version that preserves surface brightness, and thus
        changes the flux.

        @param scale The linear rescaling factor to apply.
        @returns The rescaled ChromaticObject.
        """
        ret = self.copy()
        ret.applyDilation(scale)
        return ret

    def createMagnified(self, mu):
        """Returns a new ChromaticObject by applying a lensing magnification, scaling the area and
        flux by mu at fixed surface brightness.

        This process returns a new object with a lensing magnification mu, which scales the linear
        dimensions of the image by the factor sqrt(mu), i.e., `half_light_radius` <--
        `half_light_radius * sqrt(mu)` while increasing the flux by a factor of mu.  Thus, the new
        object has the same surface brightness as the original, but different size and flux.

        See createDilated() for a version that preserves flux.

        @param mu The lensing magnification to apply.
        @returns The rescaled ChromaticObject.
        """
        ret = self.copy()
        ret.applyMagnification(mu)
        return ret

    def createSheared(self, *args, **kwargs):
        """Returns a new ChromaticObject by applying an area-preserving shear, where arguments are
        either a galsim.Shear or keyword arguments that can be used to create one.

        For more details about the allowed keyword arguments, see the documentation of galsim.Shear
        (for doxygen documentation, see galsim.shear.Shear).

        The createSheared() method precisely preserves the area.  To include a lensing distortion
        with the appropriate change in area, either use createSheared() with createMagnified(), or
        use createLensed() which combines both operations.

        @returns The sheared ChromaticObject.
        """
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    def createLensed(self, g1, g2, mu):
        """Returns a new ChromaticObject by applying a lensing shear and magnification.

        This method returns a new ChromaticObject to which the supplied lensing (reduced) shear and
        magnification has been applied.  The shear must be specified using the g1, g2 definition of
        shear (see galsim.Shear documentation for more details).  This is the same definition as
        the outputs of the galsim.PowerSpectrum and galsim.NFWHalo classes, which compute shears
        according to some lensing power spectrum or lensing by an NFW dark matter halo. The
        magnification determines the rescaling factor for the object area and flux, preserving
        surface brightness.

        @param g1      First component of lensing (reduced) shear to apply to the object.
        @param g2      Second component of lensing (reduced) shear to apply to the object.
        @param mu      Lensing magnification to apply to the object.  This is the factor by which
                       the solid angle subtended by the object is magnified, preserving surface
                       brightness.
        @returns       The lensed ChromaticObject.
        """
        ret = self.copy()
        ret.applyLensing(g1, g2, mu)
        return ret

    def createRotated(self, theta):
        """Returns a new ChromaticObject by applying a rotation.

        @param theta Rotation angle (Angle object, +ve anticlockwise).
        @returns The rotated ChromaticObject.
        """
        ret = self.copy()
        ret.applyRotation(theta)
        return ret

    def createShifted(self, *args, **kwargs):
        """Returns a new ChromaticObject by applying a shift.

        @param dx Horizontal shift to apply (float).
        @param dy Vertical shift to apply (float).

        Note: you may supply dx,dy as either two arguments, as a tuple, or as a
        galsim.PositionD or galsim.PositionI object.

        @returns The shifted ChromaticObject.
        """
        ret = self.copy()
        ret.applyShift(*args, **kwargs)
        return ret


class Chromatic(ChromaticObject):
    """Construct chromatic versions of galsim GSObjects.

    This class attaches an SED to a galsim GSObject.  This is useful to consistently generate
    the same galaxy observed through different filters, or, with the ChromaticSum class, to construct
    multi-component galaxies, each with a different SED. For example, a bulge+disk galaxy could be
    constructed:

    >>> bulge_SED = user_function_to_get_bulge_spectrum()
    >>> disk_SED = user_function_to_get_disk_spectrum()
    >>> bulge_mono = galsim.DeVaucouleurs(half_light_radius=1.0)
    >>> bulge = galsim.Chromatic(bulge_mono, bulge_SED)
    >>> disk_mono = galsim.Exponential(half_light_radius=2.0)
    >>> disk = galsim.Chromatic(disk_mono, disk_SED)
    >>> gal = bulge + disk

    The SED is usually specified as a galsim.SED object, though any callable that returns
    spectral density in photons/nanometer as a function of wavelength in nanometers should work.

    The flux normalization comes from a combination of the `flux` attribute of the GSObject being
    chromaticized and the SED.  The SED implicitly has units of photons per nanometer.  Multiplying
    this by a dimensionless throughput function (see galsim.Bandpass) and then integrating over
    wavelength gives the relative number of photons contributed by the SED/bandpass combination.
    This integral is then effectively multiplied by the `flux` attribute of the chromaticized
    GSObject to give the final number of drawn photons.
    """
    def __init__(self, gsobj, SED):
        """Attach an SED to a GSObject.

        @param gsobj    An GSObject instance to be chromaticized.
        @param SED      Typically a galsim.SED object, though any callable that returns
                        spectral density in photons/nanometer as a function of wavelength
                        in nanometers should work.
        """
        self.SED = SED
        self.gsobj = gsobj.copy()
        # Chromaticized GSObjects are separable into spatial (x,y) and spectral (lambda) factors.
        self.separable = True

    # Apply following transformations to the underlying GSObject
    def scaleFlux(self, scale):
        self.gsobj.scaleFlux(scale)

    def evaluateAtWavelength(self, wave):
        """Evaluate underlying GSObject scaled by self.SED(`wave`).

        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return self.SED(wave) * self.gsobj


class ChromaticSum(ChromaticObject):
    """Add ChromaticObjects and/or GSObjects together.  If a GSObject is part of a sum, then its
    SED is assumed to be flat with spectral density of 1 photon per nanometer.
    """
    def __init__(self, objlist):
        self.objlist = [o.copy() for o in objlist]
        self.separable = False

    def evaluateAtWavelength(self, wave):
        """Evaluate underlying GSObjects scaled by their attached SEDs evaluated at `wave`.

        @param wave  Wavelength in nanometers.
        @returns     galsim.Sum GSObject for profile at specified wavelength.
        """
        return galsim.Add([obj.evaluateAtWavelength(wave) for obj in self.objlist])

    def draw(self, bandpass, image=None, scale=None, gain=1.0, wmult=1.0,
             add_to_image=False, use_true_center=True, offset=None,
             integrator=None, **kwargs):
        """ Slightly optimized draw method for ChromaticSum's.  Draw each summand individually
        and add resulting images together.  This will waste time if both summands have the same
        associated SED, in which case the summands should be added together first and the
        resulting galsim.Sum object can then be chromaticized.  In general, however, drawing
        individual sums independently can help with speed by identifying chromatic profiles that
        are separable into spectral and spatial factors.

        @param bandpass           A galsim.Bandpass object representing the filter
                                  against which to integrate.
        @param image              see GSObject.draw()
        @param scale              see GSObject.draw()
        @param gain               see GSObject.draw()
        @param wmult              see GSObject.draw()
        @param add_to_image       see GSObject.draw()
        @param use_true_center    see GSObject.draw()
        @param offset             see GSObject.draw()
        @param integrator         One of the image integrators from galsim.integ

        @returns                  galsim.Image drawn through filter.
        """
        image = self.objlist[0].draw(bandpass, image=image, scale=scale, gain=gain, wmult=wmult,
                                     add_to_image=add_to_image, use_true_center=use_true_center,
                                     offset=offset, integrator=integrator, **kwargs)
        for obj in self.objlist[1:]:
            image = obj.draw(bandpass, image=image, scale=scale, gain=gain, wmult=wmult,
                             add_to_image=True, use_true_center=use_true_center,
                             offset=offset, integrator=integrator, **kwargs)

    # apply following transformations to all underlying summands
    def scaleFlux(self, scale):
        for obj in self.objlist:
            obj.scaleFlux(scale)


class ChromaticConvolution(ChromaticObject):
    """Convolve ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat
    spectra.
    """
    def __init__(self, objlist):
        self.objlist = []
        for obj in objlist:
            if isinstance(obj, ChromaticConvolution):
                self.objlist.extend([o.copy() for o in obj.objlist])
            else:
                self.objlist.append(obj.copy())
        if all([obj.separable for obj in self.objlist]):
            self.separable = True
            self.SED = lambda w: reduce(lambda x,y:x*y, [obj.SED(w) for obj in self.objlist])
        else:
            self.separable = False

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return galsim.Convolve([obj.evaluateAtWavelength(wave) for obj in self.objlist])

    def draw(self, bandpass, image=None, scale=None, gain=1.0, wmult=1.0,
             add_to_image=False, use_true_center=True, offset=None,
             integrator=None, iimult=None, **kwargs):
        """ Optimized draw method for ChromaticConvolution.  Works by finding sums of profiles
        which include separable portions, which can then be integrated before doing any
        convolutions, which are pushed to the end.

        @param bandpass           A galsim.Bandpass object representing the filter
                                  against which to integrate.
        @param image              see GSObject.draw()
        @param scale              see GSObject.draw()
        @param gain               see GSObject.draw()
        @param wmult              see GSObject.draw()
        @param add_to_image       see GSObject.draw()
        @param use_true_center    see GSObject.draw()
        @param offset             see GSObject.draw()
        @param integrator         One of the image integrators from galsim.integ
        @param iimult             Oversample any intermediate InterpolatedImages created to hold
                                  effective profiles by this amount.

        @returns                  galsim.Image drawn through filter.
        """
        # `ChromaticObject.draw()` can just as efficiently handle separable cases.
        if self.separable:
            return ChromaticObject.draw(self, bandpass, image=image, scale=scale, gain=gain,
                                        wmult=wmult, add_to_image=add_to_image,
                                        use_true_center=use_true_center, offset=offset,
                                        integrator=integrator, **kwargs)
        # Only make temporary changes to objlist...
        objlist = [o.copy() for o in self.objlist]

        # Now split up any `ChromaticSum`s:
        # This is the tricky part.  Some notation first:
        #     int(f(x,y,lambda)) denotes the integral over wavelength of chromatic surface brightness
        #         profile f(x,y,lambda).
        #     (f1 * f2) denotes the convolution of surface brightness profiles f1 & f2.
        #     (f1 + f2) denotes the addition of surface brightness profiles f1 & f2.
        #
        # In general, chromatic s.b. profiles can be classified as either separable or inseparable,
        # depending on whether they can be factored into spatial and spectral components or not.
        # Write separable profiles as g(x,y) * h(lambda), and leave inseparable profiles as
        # f(x,y,lambda).
        # We will suppress the arguments `x`, `y`, `lambda`, hereforward, but generally an `f` refers
        # to an inseparable profile, a `g` refers to the spatial part of a separable profile, and an
        # `h` refers to the spectral part of a separable profile.
        #
        # Now, analyze a typical scenario, a bulge+disk galaxy model (each of which is separable,
        # e.g., an SED times an exponential profile for the disk, and a different SED times a DeV
        # profile for the bulge).  Suppose the PSF is inseparable.  (Chromatic PSF's will generally
        # be inseparable since we usually think of the spatial part of the PSF being normalized to
        # unit integral for any fixed wavelength.)  Say there's also an achromatic pixel to
        # convolve with.
        # The formula for this might look like:
        #
        # img = int((bulge + disk) * PSF * pix)
        #     = int((g1 h1 + g2 h2) * f3 * g4)               # note pix is lambda-independent
        #     = int(g1 h1 * f3 * g4 + g2 h2 * f3 * g4)       # distribute the + over the *
        #     = int(g1 h1 * f3 * g4) + int(g2 h2 * f3 * g4)  # distribute the + over the int
        #     = g1 * g4 * int(h1 f3) + g2 * g4 * int(h2 f3)  # move lambda-indep terms out of int
        #
        # The result is that the integral is now inside the convolution, meaning we only have to
        # compute two convolutions instead of a convolution for each wavelength at which we evaluate
        # the integrand.  This technique, making an `effective` PSF profile for each of the bulge and
        # disk, is a significant time savings in most cases.
        # In general, we make effective profiles by splitting up `ChromaticSum`s and collecting the
        # inseparable terms on which to do integration first, and then finish with convolution last.

        # Here is the logic to turn int((g1 h1 + g2 h2) * f3) -> g1 * int(h1 f3) + g2 * int(h2 f3)
        for i, obj in enumerate(objlist):
            if isinstance(obj, ChromaticSum):
                # say obj.objlist = [A,B,C], where obj is a ChromaticSum object
                del objlist[i] # remove the add object from objlist
                tmplist = list(objlist) # collect remaining items to be convolved with each of A,B,C
                tmplist.append(obj.objlist[0]) # add A to this convolve list
                tmpobj = ChromaticConvolution(tmplist) # draw image
                image = tmpobj.draw(bandpass, image=image, gain=gain, wmult=wmult,
                                    add_to_image=add_to_image, use_true_center=use_true_center,
                                    offset=offset, integrator=integrator, iimult=iimult, **kwargs)
                for summand in obj.objlist[1:]: # now do the same for B and C
                    tmplist = list(objlist)
                    tmplist.append(summand)
                    tmpobj = ChromaticConvolution(tmplist)
                    # add to previously started image
                    image = tmpobj.draw(bandpass, image=image, gain=gain, wmult=wmult,
                                        add_to_image=True, use_true_center=use_true_center,
                                        offset=offset, integrator=integrator, iimult=iimult,
                                        **kwargs)
                return image

        # If program gets this far, the objects in objlist should be atomic (non-ChromaticSum
        # and non-ChromaticConvolution).

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength)
        prof0 = prof0._fix_center(image, scale, offset, use_true_center, reverse=False)
        image = prof0._draw_setup_image(image, scale, wmult, add_to_image)

        # Sort these atomic objects into separable and inseparable lists, and collect
        # the spectral parts of the separable profiles.
        sep_profs = []
        insep_profs = []
        sep_SED = []
        for obj in objlist:
            if obj.separable:
                if isinstance(obj, galsim.GSObject):
                    sep_profs.append(obj) # The g(x,y)'s (see above)
                else:
                    sep_profs.append(obj.evaluateAtWavelength(bandpass.effective_wavelength)
                                     /obj.SED(bandpass.effective_wavelength)) # more g(x,y)'s
                    sep_SED.append(obj.SED) # The h(lambda)'s (see above)
            else:
                insep_profs.append(obj) # The f(x,y,lambda)'s (see above)
        # insep_profs should never be empty, since separable cases were farmed out to
        # ChromaticObject.draw() above.

        # Collapse inseparable profiles into one effective profile
        SED = lambda w: reduce(lambda x,y:x*y, [s(w) for s in sep_SED], 1)
        insep_obj = galsim.Convolve(insep_profs)
        iiscale = insep_obj.evaluateAtWavelength(bandpass.effective_wavelength).nyquistDx()
        if iimult is not None:
            iiscale /= iimult
        if len(insep_profs) == 1:
            effective_prof_image = insep_profs[0].draw(bandpass*SED, wmult=wmult, scale=iiscale,
                                                       integrator=integrator, **kwargs)
        else:
            effective_prof_image = ChromaticObject.draw(insep_obj, bandpass*SED, wmult=wmult,
                                                        scale=iiscale, integrator=integrator,
                                                        **kwargs)

        # Image -> InterpolatedImage
        # It could be useful to cache this result if drawing more than one object with the same
        # PSF+SED combination.  This naturally happens in a ring test or when fitting the
        # parameters of a galaxy profile to an image when the PSF is constant.
        effective_prof = galsim.InterpolatedImage(effective_prof_image)
        # append effective profile to separable profiles (which should all be GSObjects)
        sep_profs.append(effective_prof)
        # finally, convolve and draw.
        final_prof = galsim.Convolve(sep_profs)
        return final_prof.draw(image=image, gain=gain, wmult=wmult, add_to_image=add_to_image,
                               use_true_center=use_true_center, offset=offset)

    def scaleFlux(self, scale):
        self.objlist[0].scaleFlux(scale)


class ChromaticDeconvolution(ChromaticObject):
    """A class for deconvolving a ChromaticObject.

    The ChromaticDeconvolution class represents a wavelength-dependent deconvolution kernel.

    You may also specify a gsparams argument.  See the docstring for galsim.GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if gsparams is unspecified
    (or None), then the ChromaticDeconvolution instance inherits the same GSParams as the object
    being deconvolved.

    @param obj        The object to deconvolve.
    @param gsparams   Optional gsparams argument.
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = obj.SED

    def evaluateAtWavelength(self, wave):
        return galsim.Deconvolve(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def scaleFlux(self, scale):
        self.obj.scaleFlux(scale)


class ChromaticAutoConvolution(ChromaticObject):
    """A special class for convolving a ChromaticObject with itself.

    It is equivalent in functionality to galsim.Convolve([obj,obj]), but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    @param obj        The object to be convolved with itself.
    @param real_space Whether to use real space convolution.  Default is to automatically select
                      this according to whether the object has hard edges.
    @param gsparams   Optional gsparams argument.
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = lambda w: (obj.SED(w))**2

    def evaluateAtWavelength(self, wave):
        return galsim.AutoConvolve(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def scaleFlux(self, scale):
        self.obj.scaleFlux(numpy.sqrt(scale))


class ChromaticAutoCorrelation(ChromaticObject):
    """A special class for correlating a ChromaticObject with itself.

    It is equivalent in functionality to
        galsim.Convolve([obj,obj.createRotated(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    @param obj        The object to be convolved with itself.
    @param real_space Whether to use real space convolution.  Default is to automatically select
                      this according to whether the object has hard edges.
    @param gsparams   Optional gsparams argument.
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = lambda w: (obj.SED(w))**2

    def evaluateAtWavelength(self, wave):
        return galsim.AutoCorrelate(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def scaleFlux(self, scale):
        self.obj.scaleFlux(numpy.sqrt(scale))


class ChromaticAtmosphere(ChromaticObject):
    """Class implementing two atmospheric chromatic effects: differential chromatic refraction
    (DCR) and wavelength-dependent seeing.
    Due to DCR, blue photons land closer to the zenith than red photons.
    Kolmogorov turbulence also predicts that blue photons get spread out more by the atmosphere
    than red photons, specifically FWHM is proportional to wavelength^(-0.2).
    """
    def __init__(self, base_obj, base_wavelength, zenith_angle, alpha=-0.2,
                 position_angle=0*galsim.radians, **kwargs):
        """
        @param base_obj           Fiducial PSF, equal to the monochromatic PSF at base_wavelength
        @param base_wavelength    Wavelength represented by the fiducial PSF.
        @param zenith_angle       Angle from object to zenith, expressed as a galsim.Angle
        @param alpha              Power law index for wavelength-dependent seeing.  Default of -0.2
                                  is the prediction for Kolmogorov turbulence.
        @param position_angle     Angle pointing toward zenith, measured from "up" through "right".
        @param **kwargs           Additional arguments are passed to dcr.get_refraction, and can
                                  include temperature, pressure, and H20_pressure.
        """
        self.obj = base_obj.copy()
        self.A = lambda w: numpy.matrix(numpy.identity(3), dtype=float)
        self.fluxFactor = lambda w: 1.0
        self.separable = False
        self.applyDilation(lambda w: (w/base_wavelength)**(alpha))
        base_refraction = galsim.dcr.get_refraction(base_wavelength, zenith_angle, **kwargs)
        def shift_fn(w):
            shift_magnitude = galsim.dcr.get_refraction(w, zenith_angle, **kwargs)
            shift_magnitude -= base_refraction
            shift_magnitude = shift_magnitude / galsim.arcsec
            shift = (shift_magnitude*numpy.sin(position_angle.rad()),
                     shift_magnitude*numpy.cos(position_angle.rad()))
            return shift
        self.applyShift(shift_fn)
