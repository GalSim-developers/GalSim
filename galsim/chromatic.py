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
"""@file chromatic.py
Define wavelength-dependent surface brightness profiles.

Implementation is done by constructing GSObjects as functions of wavelength. The drawImage()
method then integrates over wavelength while also multiplying by a throughput function.

Possible uses include galaxies with color gradients, automatically drawing a given galaxy through
different filters, or implementing wavelength-dependent point spread functions.
"""

import numpy as np
import copy

import galsim

class ChromaticObject(object):
    """Base class for defining wavelength dependent objects.

    This class primarily serves as the base class for chromatic subclasses, including Chromatic,
    ChromaticSum, and ChromaticConvolution.  See the docstrings for these classes for more details.
    The ChromaticAtmosphere() function also creates a ChromaticObject.

    Initialization
    --------------

    A ChromaticObject can also be instantiated directly from an existing GSObject.
    In this case, the newly created ChromaticObject will act nearly the same way the original
    GSObject works, except that it has access to the ChromaticObject methods described below;
    especially expand(), dilate() and shift().

    @param gsobj  The GSObject to be chromaticized.

    Methods
    -------

    gsobj = chrom_obj.evaluateAtWavelength(lambda) returns the monochromatic surface brightness
    profile (as a GSObject) at a given wavelength (in nanometers).

    Also, ChromaticObject has most of the same methods as GSObjects with the following exceptions:

    The GSObject access methods (e.g. xValue(), maxK(), etc.) are not available.  Instead,
    you would need to evaluate the profile at a particular wavelength and access what you want
    from that.

    There is no withFlux() method, since this is in general undefined for a chromatic object.
    See the SED class for how to set a chromatic flux density function.

    The transformation methods: transform(), expand(), dilate(), magnify(), shear(), rotate(),
    lens(), and shift() can now accept functions of wavelength as arguments, as opposed to the
    constants that GSObjects are limited to.  These methods can be used to effect a variety of
    physical chromatic effects, such as differential chromatic refraction, chromatic seeing, and
    diffraction-limited wavelength-dependence.

    The drawImage() method draws the object as observed through a particular bandpass, so the
    function parameters are somewhat different.  See the docstring for ChromaticObject.drawImage()
    for more details.

    The drawKImage() method is not yet implemented.
    """

    # In general, `ChromaticObject` and subclasses should provide the following interface:
    # 1) Define an `evaluateAtWavelength` method, which returns a GSObject representing the
    #    profile at a particular wavelength.
    # 2) Define a `withScaledFlux` method, which scales the flux at all wavelengths by a fixed
    #    multiplier.
    # 3) Initialize a `separable` attribute.  This marks whether (`separable = True`) or not
    #    (`separable = False`) the given chromatic profile can be factored into a spatial profile
    #    and a spectral profile.  Separable profiles can be drawn quickly by evaluating at a single
    #    wavelength and adjusting the flux via a (fast) 1D integral over the spectral component.
    #    Inseparable profiles, on the other hand, need to be evaluated at multiple wavelengths
    #    in order to draw (slow).
    # 4) Separable objects must initialize an `SED` attribute, which is a callable object (often a
    #    `galsim.SED` instance) that returns the _relative_ flux of the profile at a given
    #    wavelength. (The _absolute_ flux is controlled by both the `SED` and the `.flux` attribute
    #    of the underlying chromaticized GSObject(s).  See `galsim.Chromatic` docstring for details
    #    concerning normalization.)
    # 5) Initialize a `wave_list` attribute, which specifies wavelengths at which the profile (or
    #    the SED in the case of separable profiles) will be evaluated when drawing a
    #    ChromaticObject.  The type of `wave_list` should be a numpy array, and may be empty, in
    #    which case either the Bandpass object being drawn against, or the integrator being used
    #    will determine at which wavelengths to evaluate.

    # Additionally, instances of `ChromaticObject` and subclasses will usually have either an `obj`
    # attribute representing a manipulated `GSObject` or `ChromaticObject`, or an `objlist`
    # attribute in the case of compound classes like `ChromaticSum` and `ChromaticConvolution`.

    def __init__(self, obj):
        if not isinstance(obj, (galsim.GSObject, ChromaticObject)):
            raise TypeError("Can only directly instantiate ChromaticObject with a GSObject "+
                            "or ChromaticObject argument.")
        # TODO: Once we convert to a fully immutable style (when the mutating methods are
        #       eventually removed), we can get rid of this copy() call.  Probably lots of others
        #       as well...
        self.obj = obj.copy()
        if isinstance(obj, galsim.GSObject):
            self.separable = True
            self.SED = lambda w: 1.0
            self.wave_list = np.array([], dtype=float)
        elif obj.separable:
            self.separable = True
            self.SED = obj.SED
            self.wave_list = obj.wave_list
        else:
            self.separable = False
            self.SED = lambda w: 1.0
            self.wave_list = np.array([], dtype=float)
        # Some private attributes to handle affine transformations
        # _A is a 3x3 augmented affine transformation matrix that holds both translation and
        # shear/rotate/dilate specifications.
        # (see http://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix)
        self._A = lambda w: np.matrix(np.identity(3), dtype=float)
        # _fluxFactor holds a wavelength-dependent flux rescaling.  This is only needed because
        # a wavelength-dependent dilate(f(w)) is implemented as a combination of a
        # wavelength-dependent expansion and wavelength-dependent flux rescaling.
        self._fluxFactor = lambda w: 1.0

    def drawImage(self, bandpass, image=None, integrator=None, **kwargs):
        """Base implementation for drawing an image of a ChromaticObject.

        Some subclasses may choose to override this for specific efficiency gains.  For instance,
        most GalSim use cases will probably finish with a convolution, in which case
        ChromaticConvolution.drawImage() will be used.

        The task of drawImage() in a chromatic context is to integrate a chromatic surface
        brightness profile multiplied by the throughput of `bandpass`, over the wavelength interval
        indicated by `bandpass`.

        Several integrators are available in galsim.integ to do this integration.  By default,
        `galsim.integ.SampleIntegrator(rule=np.trapz)` will be used if either
        `bandpass.wave_list` or `self.wave_list` have len() > 0.  If lengths of both are zero, which
        may happen if both the bandpass throughput and the SED associated with `self` are analytic
        python functions, for example, then `galsim.integ.ContinuousIntegrator(rule=np.trapz)`
        will be used instead.  This latter case by default will evaluate the integrand at 250
        equally-spaced wavelengths between `bandpass.blue_limit` and `bandpass.red_limit`.

        By default, the above two integrators will use the trapezoidal rule for integration.  The
        midpoint rule for integration can be specified instead by passing an integrator that has
        been initialized with the `rule=galsim.integ.midpt` argument.  Finally, when creating a
        ContinuousIntegrator, the number of samples `N` is also an argument.  For example:

            >>> integrator = galsim.ContinuousIntegrator(rule=galsim.integ.midpt, N=100)
            >>> image = chromatic_obj.drawImage(bandpass, integrator=integrator)

        @param bandpass         A Bandpass object representing the filter against which to
                                integrate.
        @param image            Optionally, the Image to draw onto.  (See GSObject.drawImage()
                                for details.)  [default: None]
        @param integrator       One of the image integrators from galsim.integ [default: None,
                                which will try to select an appropriate integrator automatically.]
        @param **kwargs         For all other kwarg options, see GSObject.drawImage()

        @returns the drawn Image.
        """
        # To help developers debug extensions to ChromaticObject, check that ChromaticObject has
        # the expected attributes
        if self.separable: assert hasattr(self, 'SED')
        assert hasattr(self, 'wave_list')

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength)
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)
        # Remove from kwargs anything that is only used for setting up image:
        if 'dtype' in kwargs: kwargs.pop('dtype')
        if 'scale' in kwargs: kwargs.pop('scale')
        if 'wcs' in kwargs: kwargs.pop('wcs')

        # determine combined self.wave_list and bandpass.wave_list
        wave_list = self._getCombinedWaveList(bandpass)

        if self.separable:
            if len(wave_list) > 0:
                multiplier = np.trapz(self.SED(wave_list) * bandpass(wave_list), wave_list)
            else:
                multiplier = galsim.integ.int1d(lambda w: self.SED(w) * bandpass(w),
                                                bandpass.blue_limit, bandpass.red_limit)
            prof0 *= multiplier/self.SED(bandpass.effective_wavelength)
            image = prof0.drawImage(image=image, **kwargs)
            return image

        # decide on integrator
        if integrator is None:
            if len(wave_list) > 0:
                integrator = galsim.integ.SampleIntegrator(np.trapz)
            else:
                integrator = galsim.integ.ContinuousIntegrator(np.trapz)

        # merge self.wave_list into bandpass.wave_list if using a sampling integrator
        if isinstance(integrator, galsim.integ.SampleIntegrator):
            bandpass = galsim.Bandpass(galsim.LookupTable(wave_list, bandpass(wave_list),
                                                          interpolant='linear'))

        add_to_image = kwargs.pop('add_to_image', False)
        integral = integrator(self.evaluateAtWavelength, bandpass, image, kwargs)

        # For performance profiling, store the number of evaluations used for the last integration
        # performed.  Note that this might not be very useful for ChromaticSum instances, which are
        # drawn one profile at a time, and hence _last_n_eval will only represent the final
        # component drawn.
        self._last_n_eval = integrator.last_n_eval

        # Apply integral to the initial image appropriately.
        # Note: Don't do image = integral and return that for add_to_image==False.
        #       Remember that python doesn't actually do assignments, so this won't update the
        #       original image if the user provided one.  The following procedure does work.
        if not add_to_image:
            image.setZero()
        image += integral
        return image

    def _getCombinedWaveList(self, bandpass):
        wave_list = bandpass.wave_list
        wave_list = np.union1d(wave_list, self.wave_list)
        wave_list = wave_list[wave_list <= bandpass.red_limit]
        wave_list = wave_list[wave_list >= bandpass.blue_limit]
        return wave_list

    def draw(self, *args, **kwargs):
        """An obsolete synonym for obj.drawImage(method='no_pixel')
        """
        normalization = kwargs.pop('normalization','f')
        if normalization in ['flux','f']:
            return self.drawImage(*args, method='no_pixel', **kwargs)
        else:
            return self.drawImage(*args, method='sb', **kwargs)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        @param wave     Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        if self.__class__ != ChromaticObject:
            raise NotImplementedError(
                    "Subclasses of ChromaticObject must override evaluateAtWavelength()")
        if not hasattr(self, '_A'):
            raise AttributeError(
                    "Attempting to evaluate ChromaticObject before affine transform " +
                    "matrix has been created!")
        ret = self.obj.evaluateAtWavelength(wave).copy()
        A0 = self._A(wave)
        ret = ret.transform(A0[0,0], A0[0,1], A0[1,0], A0[1,1])
        ret = ret.shift(A0[0,2], A0[1,2])
        ret = ret * self._fluxFactor(wave)
        return ret

    def __mul__(self, flux_ratio):
        """Scale the flux of the object by the given flux ratio.

        obj * flux_ratio is equivalent to obj.withFlux(obj.getFlux() * flux_ratio)

        It creates a new object that has the same profile as the original, but with the
        surface brightness at every location scaled by the given amount.

        @param flux_ratio   The factor by which to scale the linear dimension of the object.
                            In addition, `flux_ratio` may be a callable function, in which case
                            the argument should be wavelength in nanometers and the return value
                            the scale for that wavelength.

        @returns a new object with the flux scaled appropriately.
        """
        if hasattr(flux_ratio, '__call__'):
            ret = self.copy()
            ret._fluxFactor = lambda w: self._fluxFactor(w) * flux_ratio(w)
            return ret
        else:
            return self.withScaledFlux(flux_ratio)

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        ret = self.copy()
        ret.obj = self.obj.withScaledFlux(flux_ratio)
        return ret

    def centroid(self, bandpass):
        """ Determine the centroid of the wavelength-integrated surface brightness profile.

        @param bandpass  The bandpass through which the observation is made.

        @returns the centroid of the integrated surface brightness profile, as a PositionD.
        """
        # if either the Bandpass or self maintain a wave_list, evaluate integrand only at
        # those wavelengths.
        if len(bandpass.wave_list) > 0 or len(self.wave_list) > 0:
            w = np.union1d(bandpass.wave_list, self.wave_list)
            w = w[(w <= bandpass.red_limit) & (w >= bandpass.blue_limit)]
            objs = [self.evaluateAtWavelength(y) for y in w]
            fluxes = [o.getFlux() for o in objs]
            centroids = [o.centroid() for o in objs]
            xcentroids = np.array([c.x for c in centroids])
            ycentroids = np.array([c.y for c in centroids])
            bp = bandpass(w)
            flux = np.trapz(bp * fluxes, w)
            xcentroid = np.trapz(bp * fluxes * xcentroids, w) / flux
            ycentroid = np.trapz(bp * fluxes * ycentroids, w) / flux
            return galsim.PositionD(xcentroid, ycentroid)
        else:
            flux_integrand = lambda w: self.evaluateAtWavelength(w).getFlux() * bandpass(w)
            def xcentroid_integrand(w):
                mono = self.evaluateAtWavelength(w)
                return mono.centroid().x * mono.getFlux() * bandpass(w)
            def ycentroid_integrand(w):
                mono = self.evaluateAtWavelength(w)
                return mono.centroid().y * mono.getFlux() * bandpass(w)
            flux = galsim.integ.int1d(flux_integrand, bandpass.blue_limit, bandpass.red_limit)
            xcentroid = 1./flux * galsim.integ.int1d(xcentroid_integrand,
                                                     bandpass.blue_limit,
                                                     bandpass.red_limit)
            ycentroid = 1./flux * galsim.integ.int1d(ycentroid_integrand,
                                                     bandpass.blue_limit,
                                                     bandpass.red_limit)
            return galsim.PositionD(xcentroid, ycentroid)

    # Add together `ChromaticObject`s and/or `GSObject`s
    def __add__(self, other):
        return galsim.ChromaticSum([self, other])

    # Subtract `ChromaticObject`s and/or `GSObject`s
    def __sub__(self, other):
        return galsim.ChromaticSum([self, (-1. * other)])

    # Make op* and op*= work to adjust the flux of the object
    def __rmul__(self, other):
        return self.__mul__(other)

    # Likewise for op/ and op/=
    def __div__(self, other):
        return self.__mul__(1./other)

    def __truediv__(self, other):
        return self.__div__(other)

    # Make a new copy of a `ChromaticObject`.
    def copy(self):
        """Returns a copy of an object.  This preserves the original type of the object."""
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

    # Helper function
    def _applyMatrix(self, J):
        if isinstance(self, ChromaticSum):
            # Don't wrap ChromaticSum object, easier to just wrap its arguments.
            return ChromaticSum([ obj._applyMatrix(J) for obj in self.objlist ])
        else:
            # return a copy with the Jacobian matrix J applied to the ChromaticObject.
            if hasattr(self, '_A'):
                ret = self.copy()
                if hasattr(J, '__call__'):
                    ret._A = lambda w: J(w) * self._A(w)
                    ret.separable = False
                else:
                    ret._A = lambda w: J * self._A(w)
            else:
                ret = ChromaticObject(self)
                if hasattr(J, '__call__'):
                    ret._A = J
                    ret.separable = False
                else:
                    ret._A = lambda w: J
            # To help developers debug extensions to ChromaticObject, check that this object
            # already has a few expected attributes
            if ret.separable: assert hasattr(ret, 'SED')
            assert hasattr(ret, 'wave_list')
            return ret

    def expand(self, scale):
        """Expand the linear size of the profile by the given (possibly wavelength-dependent)
        scale factor `scale`, while preserving surface brightness.

        This doesn't correspond to either of the normal operations one would typically want to
        do to a galaxy.  The functions dilate() and magnify() are the more typical usage.  But this
        function is conceptually simple.  It rescales the linear dimension of the profile, while
        preserving surface brightness.  As a result, the flux will necessarily change as well.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        See magnify() for a version that applies a scale factor to the area while preserving surface
        brightness.

        @param scale    The factor by which to scale the linear dimension of the object.  In
                        addition, `scale` may be a callable function, in which case the argument
                        should be wavelength in nanometers and the return value the scale for that
                        wavelength.

        @returns the expanded object
        """
        if hasattr(scale, '__call__'):
            E = lambda w: np.matrix(np.diag([scale(w), scale(w), 1]))
        else:
            E = np.diag([scale, scale, 1])
        return self._applyMatrix(E)

    def dilate(self, scale):
        """Dilate the linear size of the profile by the given (possibly wavelength-dependent)
        `scale`, while preserving flux.

        e.g. `half_light_radius` <-- `half_light_radius * scale`

        See expand() and magnify() for versions that preserve surface brightness, and thus
        change the flux.

        @param scale    The linear rescaling factor to apply.  In addition, `scale` may be a
                        callable function, in which case the argument should be wavelength in
                        nanometers and the return value the scale for that wavelength.

        @returns the dilated object.
        """
        if hasattr(scale, '__call__'):
            return self.expand(scale) * (lambda w: 1./scale(w)**2)
        else:
            return self.expand(scale) * (1./scale**2)  # conserve flux

    def magnify(self, mu):
        """Apply a lensing magnification, scaling the area and flux by `mu` at fixed surface
        brightness.

        This process applies a lensing magnification `mu`, which scales the linear dimensions of the
        image by the factor sqrt(mu), i.e., `half_light_radius` <-- `half_light_radius * sqrt(mu)`
        while increasing the flux by a factor of `mu`.  Thus, magnify() preserves surface
        brightness.

        See dilate() for a version that applies a linear scale factor while preserving flux.

        @param mu       The lensing magnification to apply.  In addition, `mu` may be a callable
                        function, in which case the argument should be wavelength in nanometers
                        and the return value the magnification for that wavelength.

        @returns the magnified object.
        """
        import math
        if hasattr(mu, '__call__'):
            return self.expand(lambda w: math.sqrt(mu(w)))
        else:
            return self.expand(math.sqrt(mu))

    def shear(self, *args, **kwargs):
        """Apply an area-preserving shear to this object, where arguments are either a Shear,
        or arguments that will be used to initialize one.

        For more details about the allowed keyword arguments, see the documentation for Shear
        (for doxygen documentation, see galsim.shear.Shear).

        The shear() method precisely preserves the area.  To include a lensing distortion with
        the appropriate change in area, either use shear() with magnify(), or use lens(), which
        combines both operations.

        Note that, while gravitational shear is monochromatic, the shear method may be used for 
        many other use cases including some which may be wavelength-dependent, such as 
        intrinsic galaxy shape, telescope dilation, atmospheric PSF shape, etc.  Thus, the
        shear argument is allowed to be a function of wavelength like other transformations.

        @param shear    The shear to be applied. Or, as described above, you may instead supply
                        parameters to construct a Shear directly.  eg. `obj.shear(g1=g1,g2=g2)`.
                        In addition, the `shear` parameter may be a callable function, in which
                        case the argument should be wavelength in nanometers and the return value
                        the shear for that wavelength, returned as a galsim.Shear instance.

        @returns the sheared object.
        """
        if len(args) == 1:
            if kwargs:
                raise TypeError("Gave both unnamed and named arguments!")
            if not hasattr(args[0], '__call__') and not isinstance(args[0], galsim.Shear):
                raise TypeError("Unnamed argument is not a Shear or function returning Shear!")
            shear = args[0]
        elif len(args) > 1:
            raise TypeError("Too many unnamed arguments!")
        elif 'shear' in kwargs:
            # Need to break this out specially in case it is a function of wavelength
            shear = kwargs.pop('shear')
            if kwargs:
                raise TypeError("Too many kwargs provided!")
        else:
            shear = galsim.Shear(**kwargs)
        if hasattr(shear, '__call__'):
            def buildSMatrix(w):
                S = np.matrix(np.identity(3), dtype=float)
                S[0:2,0:2] = shear(w)._shear.getMatrix()
                return S
            S = buildSMatrix
        else:
            S = np.matrix(np.identity(3), dtype=float)
            S[0:2,0:2] = shear._shear.getMatrix()
        return self._applyMatrix(S)

    def lens(self, g1, g2, mu):
        """Apply a lensing shear and magnification to this object.

        This ChromaticObject method applies a lensing (reduced) shear and magnification.  The shear
        must be specified using the g1, g2 definition of shear (see Shear documentation for more
        details).  This is the same definition as the outputs of the PowerSpectrum and NFWHalo
        classes, which compute shears according to some lensing power spectrum or lensing by an NFW
        dark matter halo.  The magnification determines the rescaling factor for the object area and
        flux, preserving surface brightness.

        While gravitational lensing is achromatic, we do allow the parameters `g1`, `g2`, and `mu`
        to be callable functions to be parallel to all the other transformations of chromatic
        objects.  In this case, the functions should take the wavelength in nanometers as the 
        argument, and the return values are the corresponding value at that wavelength.

        @param g1       First component of lensing (reduced) shear to apply to the object.
        @param g2       Second component of lensing (reduced) shear to apply to the object.
        @param mu       Lensing magnification to apply to the object.  This is the factor by which
                        the solid angle subtended by the object is magnified, preserving surface
                        brightness.

        @returns the lensed object.
        """
        if any(hasattr(g, '__call__') for g in [g1,g2]):
            _g1 = g1
            _g2 = g2
            if not hasattr(g1, '__call__'): _g1 = lambda w: g1
            if not hasattr(g2, '__call__'): _g2 = lambda w: g2
            S = lambda w: galsim.Shear(g1=_g1(w), g2=_g2(w))
            sheared = self.shear(S)
        else:
            sheared = self.shear(g1=g1,g2=g2)
        return sheared.magnify(mu)

    def rotate(self, theta):
        """Rotate this object by an Angle `theta`.

        @param theta    Rotation angle (Angle object, +ve anticlockwise). In addition, `theta` may
                        be a callable function, in which case the argument should be wavelength in
                        nanometers and the return value the rotation angle for that wavelength,
                        returned as a galsim.Angle instance.

        @returns the rotated object.
        """
        import math
        if hasattr(theta, '__call__'):
            def buildRMatrix(w):
                cth = math.cos(theta(w).rad())
                sth = math.sin(theta(w).rad())
                R = np.matrix([[cth, -sth, 0],
                               [sth,  cth, 0],
                               [  0,    0, 1]], dtype=float)
                return R
            R = buildRMatrix
        else:
            cth = math.cos(theta.rad())
            sth = math.sin(theta.rad())
            R = np.matrix([[cth, -sth, 0],
                           [sth,  cth, 0],
                           [  0,    0, 1]], dtype=float)
        return self._applyMatrix(R)

    def transform(self, dudx, dudy, dvdx, dvdy):
        """Apply a transformation to this object defined by an arbitrary Jacobian matrix.

        This works the same as GSObject.transform(), so see that method's docstring for more
        details.

        As with the other more specific chromatic trasnformations, dudx, dudy, dvdx, and dvdy
        may be callable functions, in which case the argument should be wavelength in nanometers
        and the return value the appropriate value for that wavelength.

        @param dudx     du/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dudy     du/dy, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdx     dv/dx, where (x,y) are the current coords, and (u,v) are the new coords.
        @param dvdy     dv/dy, where (x,y) are the current coords, and (u,v) are the new coords.

        @returns the transformed object.
        """
        if any(hasattr(dd, '__call__') for dd in [dudx, dudy, dvdx, dvdy]):
            _dudx = dudx
            _dudy = dudy
            _dvdx = dvdx
            _dvdy = dvdy
            if not hasattr(dudx, '__call__'): _dudx = lambda w: dudx
            if not hasattr(dudy, '__call__'): _dudy = lambda w: dudy
            if not hasattr(dvdx, '__call__'): _dvdx = lambda w: dvdx
            if not hasattr(dvdy, '__call__'): _dvdy = lambda w: dvdy
            J = lambda w: np.matrix([[_dudx(w), _dudy(w), 0],
                                     [_dvdx(w), _dvdy(w), 0],
                                     [       0,        0, 1]], dtype=float)
        else:
            J = np.matrix([[dudx, dudy, 0],
                           [dvdx, dvdy, 0],
                           [   0,    0, 1]], dtype=float)
        return self._applyMatrix(J)

    def shift(self, *args, **kwargs):
        """Apply a (possibly wavelength-dependent) (dx, dy) shift to this chromatic object.

        For a wavelength-independent shift, you may supply `dx,dy` as either two arguments, as a
        tuple, or as a PositionD or PositionI object.

        For a wavelength-dependent shift, you may supply two functions of wavelength in nanometers
        which will be interpreted as `dx(wave)` and `dy(wave)`, or a single function of wavelength
        in nanometers that returns either a 2-tuple, PositionD, or PositionI.

        @param dx   Horizontal shift to apply (float or function).
        @param dy   Vertical shift to apply (float or function).

        @returns the shifted object.

        """
        # This follows along the galsim.utilities.pos_args function, but has some
        # extra bits to account for the possibility of dx,dy being functions.
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
            raise TypeError("Too many arguments supplied!")
        if kwargs:
            raise TypeError("Got unexpected keyword arguments: %s",kwargs.keys())
        if hasattr(dx, '__call__') or hasattr(dy, '__call__'):
            # Functionalize dx, dy as needed.
            if not hasattr(dx, '__call__'):
                tmpdx = dx
                dx = lambda w: tmpdx
            if not hasattr(dy, '__call__'):
                tmpdy = dy
                dy = lambda w: tmpdy
            # Then create augmented affine transform matrix and multiply or set as necessary
            T = lambda w: np.matrix([[1, 0, dx(w)],
                                     [0, 1, dy(w)],
                                     [0, 0,     1]], dtype=float)
        else:
            T = np.matrix([[1, 0, dx],
                           [0, 1, dy],
                           [0, 0,  1]], dtype=float)
        return self._applyMatrix(T)

def ChromaticAtmosphere(base_obj, base_wavelength, **kwargs):
    """Return a ChromaticObject implementing two atmospheric chromatic effects: differential
    chromatic refraction (DCR) and wavelength-dependent seeing.

    Due to DCR, blue photons land closer to the zenith than red photons.  Kolmogorov turbulence
    also predicts that blue photons get spread out more by the atmosphere than red photons,
    specifically FWHM is proportional to wavelength^(-0.2).  Both of these effects can be
    implemented by wavelength-dependent shifts and dilations.

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

    Note that this function implicitly assumes that lengths are in arcseconds.  Thus, to use this
    function, you should specify properties like FWHM, half_light_radius, and pixel scales in
    arcsec.  This is unlike the rest of GalSim, in which Position units only need to be internally
    consistent.

    @param base_obj             Fiducial PSF, equal to the monochromatic PSF at `base_wavelength`
    @param base_wavelength      Wavelength represented by the fiducial PSF.
    @param alpha                Power law index for wavelength-dependent seeing.  [default: -0.2,
                                the prediction for Kolmogorov turbulence]
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

    @returns a ChromaticObject representing a chromatic atmospheric PSF.
    """
    alpha = kwargs.pop('alpha', -0.2)
    # Determine zenith_angle and parallactic_angle from kwargs
    if 'zenith_angle' in kwargs:
        zenith_angle = kwargs.pop('zenith_angle')
        parallactic_angle = kwargs.pop('parallactic_angle', 0.0*galsim.degrees)
    elif 'obj_coord' in kwargs:
        obj_coord = kwargs.pop('obj_coord')
        if 'zenith_coord' in kwargs:
            zenith_coord = kwargs.pop('zenith_coord')
            zenith_angle, parallactic_angle = galsim.dcr.zenith_parallactic_angles(
                obj_coord=obj_coord, zenith_coord=zenith_coord)
        else:
            if 'HA' not in kwargs or 'latitude' not in kwargs:
                raise TypeError("ChromaticAtmosphere requires either zenith_coord or (HA, "
                                +"latitude) when obj_coord is specified!")
            HA = kwargs.pop('HA')
            latitude = kwargs.pop('latitude')
            zenith_angle, parallactic_angle = galsim.dcr.zenith_parallactic_angles(
                obj_coord=obj_coord, HA=HA, latitude=latitude)
    else:
        raise TypeError("Need to specify zenith_angle and parallactic_angle!")
    # Any remaining kwargs will get forwarded to galsim.dcr.get_refraction
    # Check that they're valid
    for kw in kwargs.keys():
        if kw not in ['temperature', 'pressure', 'H2O_pressure']:
            raise TypeError("Got unexpected keyword: {0}".format(kw))

    ret = ChromaticObject(base_obj)
    ret = ret.dilate(lambda w: (w/base_wavelength)**alpha)
    base_refraction = galsim.dcr.get_refraction(base_wavelength, zenith_angle, **kwargs)
    def shift_fn(w):
        shift_magnitude = galsim.dcr.get_refraction(w, zenith_angle, **kwargs)
        shift_magnitude -= base_refraction
        shift_magnitude = shift_magnitude * (galsim.radians / galsim.arcsec)
        shift = (-shift_magnitude*np.sin(parallactic_angle.rad()),
                 shift_magnitude*np.cos(parallactic_angle.rad()))
        return shift
    ret = ret.shift(shift_fn)
    return ret


class Chromatic(ChromaticObject):
    """Construct chromatic versions of galsim GSObjects.

    This class attaches an SED to a galsim GSObject.  This is useful to consistently generate
    the same galaxy observed through different filters, or, with the ChromaticSum class, to
    construct multi-component galaxies, each with a different SED. For example, a bulge+disk galaxy
    could be constructed:

        >>> bulge_SED = user_function_to_get_bulge_spectrum()
        >>> disk_SED = user_function_to_get_disk_spectrum()
        >>> bulge_mono = galsim.DeVaucouleurs(half_light_radius=1.0)
        >>> disk_mono = galsim.Exponential(half_light_radius=2.0)
        >>> bulge = galsim.Chromatic(bulge_mono, bulge_SED)
        >>> disk = galsim.Chromatic(disk_mono, disk_SED)
        >>> gal = bulge + disk

    Some syntactic sugar available for creating Chromatic instances is simply to multiply
    a GSObject instance by an SED instance.  Thus the last three lines above are equivalent to:

        >>> gal = bulge_mono * bulge_SED + disk_mono * disk_SED

    The SED is usually specified as a galsim.SED object, though any callable that returns
    spectral density in photons/nanometer as a function of wavelength in nanometers should work.

    Typically, the SED describes the flux in photons per nanometer of an object with a particular
    magnitude, possibly normalized with either the method sed.withFlux() or sed.withMagnitude()
    (see the docstrings in the SED class for details about these and other normalization options).
    Then the `flux` attribute of the GSObject should just be the _relative_ flux scaling of the
    current object compared to that normalization.  This implies (at least) two possible
    conventions.
    1. You can normalize the SED to have unit flux with `sed = sed.withFlux(bandpass, 1.0)`. Then
    the `flux` of each GSObject would be the actual flux in photons when observed in the given
    bandpass.
    2. You can leave the object flux as 1 (the default for most types when you construct them) and
    set the flux in the SED with `sed = sed.withFlux(bandpass, flux)`.  Then if the object had
    `flux` attribute different from 1, it would just refer to the factor by which that particular
    object is brighter than the value given in the normalization command.

    Initialization
    --------------

    @param gsobj    A GSObject instance to be chromaticized.
    @param SED      Typically an SED object, though any callable that returns
                    spectral density in photons/nanometer as a function of wavelength
                    in nanometers should work.
    """
    def __init__(self, gsobj, SED):
        self.SED = SED
        self.wave_list = SED.wave_list
        self.obj = gsobj.copy()
        # Chromaticized GSObjects are separable into spatial (x,y) and spectral (lambda) factors.
        self.separable = True

    # Apply following transformations to the underlying GSObject
    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`.

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        return Chromatic(self.obj.withScaledFlux(flux_ratio), self.SED)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        if isinstance(self.obj, InterpolatedChromaticObject):
            return self.SED(wave) * self.obj.evaluateAtWavelength(wave)
        else:
            return self.SED(wave) * self.obj


class ChromaticSum(ChromaticObject):
    """Add ChromaticObjects and/or GSObjects together.  If a GSObject is part of a sum, then its
    SED is assumed to be flat with spectral density of 1 photon per nanometer.

    This is the type returned from `galsim.Add(objects)` if any of the objects are a
    ChromaticObject.

    Initialization
    --------------

    Typically, you do not need to construct a ChromaticSum object explicitly.  Normally, you
    would just use the + operator, which returns a ChromaticSum when used with chromatic objects:

        >>> bulge = galsim.Sersic(n=3, half_light_radius=0.8) * bulge_sed
        >>> disk = galsim.Exponential(half_light_radius=1.4) * disk_sed
        >>> gal = bulge + disk

    You can also use the Add() factory function, which returns a ChromaticSum object if any of
    the individual objects are chromatic:

        >>> gal = galsim.Add([bulge,disk])

    @param args             Unnamed args should be a list of objects to add.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, *args, **kwargs):
        # Check kwargs first:
        self.gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())

        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Must provide at least one GSObject or ChromaticObject.")
        elif len(args) == 1:
            # 1 argument.  Should be either a GSObject, ChromaticObject or a list of these.
            if isinstance(args[0], (galsim.GSObject, ChromaticObject)):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError("Single input argument must be a GSObject, a ChromaticObject,"
                                +" or list of them.")
        # else args is already the list of objects

        # check for separability
        self.separable = False # assume not separable, then test for the opposite
        if all([obj.separable for obj in args]): # needed to assure that obj.SED is always defined.
            SED1 = args[0].SED
            # sum is separable if all summands have the same SED.
            if all([obj.SED == SED1 for obj in args[1:]]):
                self.separable = True
                self.SED = SED1
                self.objlist = [o.copy() for o in args]
        # if not all the same SED, try to identify groups of summands with the same SED.
        if not self.separable:
            # Dictionary of: SED -> List of objs with that SED.
            SED_dict = {}
            # Fill in objlist as we go.
            self.objlist = []
            for obj in args:
                # if separable, then add to one of the dictionary lists
                if obj.separable:
                    if obj.SED not in SED_dict:
                        SED_dict[obj.SED] = []
                    SED_dict[obj.SED].append(obj)
                # otherwise, just add to self.objlist
                else:
                    self.objlist.append(obj.copy())
            # go back and populate self.objlist with separable items, grouping objs with the
            # same SED.
            for v in SED_dict.values():
                if len(v) == 1:
                    self.objlist.append(v[0].copy())
                else:
                    self.objlist.append(ChromaticSum(v))
        # finish up by constructing self.wave_list
        self.wave_list = np.array([], dtype=float)
        for obj in self.objlist:
            self.wave_list = np.union1d(self.wave_list, obj.wave_list)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength `wave`.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return galsim.Add([obj.evaluateAtWavelength(wave) for obj in self.objlist],
                          gsparams=self.gsparams)

    def drawImage(self, bandpass, image=None, integrator=None, **kwargs):
        """Slightly optimized draw method for ChromaticSum instances.

        Draws each summand individually and add resulting images together.  This might waste time if
        two or more summands are separable and have the same SED, and another summand with a
        different SED is also added, in which case the summands should be added together first and
        the resulting Sum object can then be chromaticized.  In general, however, drawing individual
        sums independently can help with speed by identifying chromatic profiles that are separable
        into spectral and spatial factors.

        @param bandpass         A Bandpass object representing the filter against which to
                                integrate.
        @param image            Optionally, the Image to draw onto.  (See GSObject.drawImage()
                                for details.)  [default: None]
        @param integrator       One of the image integrators from galsim.integ [default: None,
                                which will try to select an appropriate integrator automatically.]
        @param **kwargs         For all other kwarg options, see GSObject.drawImage()

        @returns the drawn Image.
        """
        add_to_image = kwargs.pop('add_to_image', False)
        # Use given add_to_image for the first one, then add_to_image=False for the rest.
        image = self.objlist[0].drawImage(
                bandpass, image=image, add_to_image=add_to_image, **kwargs)
        for obj in self.objlist[1:]:
            image = obj.drawImage(
                    bandpass, image=image, add_to_image=True, **kwargs)
        return image

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        return ChromaticSum([ obj.withScaledFlux(flux_ratio) for obj in self.objlist ])


class ChromaticConvolution(ChromaticObject):
    """Convolve ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat
    spectra.

    This is the type returned from `galsim.Convolve(objects)` if any of the objects is a
    ChromaticObject.

    Initialization
    --------------

    The normal way to use this class is to use the Convolve() factory function:

        >>> gal = galsim.Sersic(n, half_light_radius) * galsim.SED(sed_file)
        >>> psf = galsim.ChromaticAtmosphere(...)
        >>> final = galsim.Convolve([gal, psf])

    The objects to be convolved may be provided either as multiple unnamed arguments (e.g.
    `Convolve(psf, gal, pix)`) or as a list (e.g. `Convolve([psf, gal, pix])`).  Any number of
    objects may be provided using either syntax.  (Well, the list has to include at least 1 item.)

    @param args             Unnamed args should be a list of objects to convolve.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Must provide at least one GSObject or ChromaticObject")
        elif len(args) == 1:
            if isinstance(args[0], (galsim.GSObject, ChromaticObject)):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError(
                    "Single input argument must be a GSObject, or a ChromaticObject,"
                    +" or list of them.")

        # Check kwargs
        # real space convolution is not implemented for chromatic objects.
        real_space = kwargs.pop("real_space", None)
        if real_space:
            raise NotImplementedError(
                "Real space convolution of chromatic objects not implemented.")
        self.gsparams = kwargs.pop("gsparams", None)

        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())

        self.objlist = []
        for obj in args:
            if isinstance(obj, ChromaticConvolution):
                self.objlist.extend([o.copy() for o in obj.objlist])
            else:
                self.objlist.append(obj.copy())
        if all([obj.separable for obj in self.objlist]):
            self.separable = True
            # in practice, probably only one object in self.objlist has a nontrivial SED
            # but go through all of them anyway.
            self.SED = lambda w: reduce(lambda x,y:x*y, [obj.SED(w) for obj in self.objlist])
        else:
            self.separable = False

        # Assemble wave_lists
        self.wave_list = np.array([], dtype=float)
        for obj in self.objlist:
            self.wave_list = np.union1d(self.wave_list, obj.wave_list)

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength `wave`.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return galsim.Convolve([obj.evaluateAtWavelength(wave) for obj in self.objlist],
                               gsparams=self.gsparams)

    def drawImage(self, bandpass, image=None, integrator=None, iimult=None, **kwargs):
        """Optimized draw method for the ChromaticConvolution class.

        Works by finding sums of profiles which include separable portions, which can then be
        integrated before doing any convolutions, which are pushed to the end.

        @param bandpass         A Bandpass object representing the filter against which to
                                integrate.
        @param image            Optionally, the Image to draw onto.  (See GSObject.drawImage()
                                for details.)  [default: None]
        @param integrator       One of the image integrators from galsim.integ [default: None,
                                which will try to select an appropriate integrator automatically.]
        @param iimult           Oversample any intermediate InterpolatedImages created to hold
                                effective profiles by this amount. [default: None]
        @param **kwargs         For all other kwarg options, see GSObject.drawImage()

        @returns the drawn Image.
        """
        # `ChromaticObject.drawImage()` can just as efficiently handle separable cases.
        if self.separable:
            return ChromaticObject.drawImage(self, bandpass, image=image, **kwargs)

        # Only make temporary changes to objlist...
        objlist = [o.copy() for o in self.objlist]

        # Now split up any `ChromaticSum`s:
        # This is the tricky part.  Some notation first:
        #     int(f(x,y,lambda)) denotes the integral over wavelength of chromatic surface
        #         brightness profile f(x,y,lambda).
        #     (f1 * f2) denotes the convolution of surface brightness profiles f1 & f2.
        #     (f1 + f2) denotes the addition of surface brightness profiles f1 & f2.
        #
        # In general, chromatic s.b. profiles can be classified as either separable or inseparable,
        # depending on whether they can be factored into spatial and spectral components or not.
        # Write separable profiles as g(x,y) * h(lambda), and leave inseparable profiles as
        # f(x,y,lambda).
        # We will suppress the arguments `x`, `y`, `lambda`, hereforward, but generally an `f`
        # refers to an inseparable profile, a `g` refers to the spatial part of a separable
        # profile, and an `h` refers to the spectral part of a separable profile.
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
        # the integrand.  This technique, making an "effective" PSF profile for each of the bulge
        # and disk, is a significant time savings in most cases.
        #
        # In general, we make effective profiles by splitting up ChromaticSum items and collecting
        # the inseparable terms on which to do integration first, and then finish with convolution
        # last.

        # Here is the logic to turn int((g1 h1 + g2 h2) * f3) -> g1 * int(h1 f3) + g2 * int(h2 f3)
        for i, obj in enumerate(objlist):
            if isinstance(obj, ChromaticSum):
                # say obj.objlist = [A,B,C], where obj is a ChromaticSum object
                del objlist[i] # remove the add object from objlist
                tmplist = list(objlist) # collect remaining items to be convolved with each of A,B,C
                tmplist.append(obj.objlist[0]) # add A to this convolve list
                tmpobj = ChromaticConvolution(tmplist) # draw image
                add_to_image = kwargs.pop('add_to_image', False)
                image = tmpobj.drawImage(bandpass, image=image, integrator=integrator,
                                         iimult=iimult, add_to_image=add_to_image, **kwargs)
                for summand in obj.objlist[1:]: # now do the same for B and C
                    tmplist = list(objlist)
                    tmplist.append(summand)
                    tmpobj = ChromaticConvolution(tmplist)
                    # add to previously started image
                    image = tmpobj.drawImage(bandpass, image=image, integrator=integrator,
                                             iimult=iimult, add_to_image=True, **kwargs)
                # Return the image here, breaking the loop early.  If there are two ChromaticSum
                # instances in objlist, then the next pass through will repeat the procedure
                # on the other one, effectively distributing the multiplication over both sums.
                return image

        # If program gets this far, the objects in objlist should be atomic (non-ChromaticSum
        # and non-ChromaticConvolution).  (The latter case was dealt with in the constructor.)

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength)
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)

        # Sort these atomic objects into separable and inseparable lists, and collect
        # the spectral parts of the separable profiles.
        sep_profs = []
        insep_profs = []
        sep_SED = []
        wave_list = np.array([], dtype=float)
        for obj in objlist:
            if obj.separable:
                if isinstance(obj, galsim.GSObject):
                    sep_profs.append(obj) # The g(x,y)'s (see above)
                else:
                    sep_profs.append(obj.evaluateAtWavelength(bandpass.effective_wavelength)
                                     /obj.SED(bandpass.effective_wavelength)) # more g(x,y)'s
                    sep_SED.append(obj.SED) # The h(lambda)'s (see above)
                    wave_list = np.union1d(wave_list, obj.wave_list)
            else:
                insep_profs.append(obj) # The f(x,y,lambda)'s (see above)
        # insep_profs should never be empty, since separable cases were farmed out to
        # ChromaticObject.drawImage() above.

        # Collapse inseparable profiles into one effective profile
        SED = lambda w: reduce(lambda x,y:x*y, [s(w) for s in sep_SED], 1)
        insep_obj = galsim.Convolve(insep_profs, gsparams=self.gsparams)
        # Find scale at which to draw effective profile
        iiscale = insep_obj.evaluateAtWavelength(bandpass.effective_wavelength).nyquistScale()
        if iimult is not None:
            iiscale /= iimult
        # Create the effective bandpass.
        wave_list = np.union1d(wave_list, bandpass.wave_list)
        wave_list = wave_list[wave_list >= bandpass.blue_limit]
        wave_list = wave_list[wave_list <= bandpass.red_limit]
        effective_bandpass = galsim.Bandpass(
            galsim.LookupTable(wave_list, bandpass(wave_list) * SED(wave_list),
                               interpolant='linear'))
        # If there's only one inseparable profile, let it draw itself.
        wmult = kwargs.get('wmult', 1)
        if len(insep_profs) == 1:
            if isinstance(insep_profs[0], InterpolatedChromaticObject):
                effective_prof_image = insep_profs[0].drawImage(
                    effective_bandpass, wmult=wmult, scale=iiscale,
                    method='no_pixel')
            else:
                effective_prof_image = insep_profs[0].drawImage(
                    effective_bandpass, wmult=wmult, scale=iiscale, integrator=integrator,
                    method='no_pixel')
        # Otherwise, use superclass ChromaticObject to draw convolution of inseparable profiles.
        else:
            effective_prof_image = ChromaticObject.drawImage(
                    insep_obj, effective_bandpass, wmult=wmult, scale=iiscale,
                    integrator=integrator, method='no_pixel')

        # Image -> InterpolatedImage
        # It could be useful to cache this result if drawing more than one object with the same
        # PSF+SED combination.  This naturally happens in a ring test or when fitting the
        # parameters of a galaxy profile to an image when the PSF is constant.
        effective_prof = galsim.InterpolatedImage(effective_prof_image, gsparams=self.gsparams)
        # append effective profile to separable profiles (which should all be GSObjects)
        sep_profs.append(effective_prof)
        # finally, convolve and draw.
        final_prof = galsim.Convolve(sep_profs, gsparams=self.gsparams)
        return final_prof.drawImage(image=image,**kwargs)

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`.

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        ret = self.copy()
        ret.objlist[0] *= flux_ratio
        return ret


class ChromaticDeconvolution(ChromaticObject):
    """A class for deconvolving a ChromaticObject.

    The ChromaticDeconvolution class represents a wavelength-dependent deconvolution kernel.

    You may also specify a gsparams argument.  See the docstring for GSParams using
    help(galsim.GSParams) for more information about this option.  Note: if `gsparams` is
    unspecified (or None), then the ChromaticDeconvolution instance inherits the same GSParams as
    the object being deconvolved.

    Initialization
    --------------

    @param obj              The object to deconvolve.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = lambda w: 1./obj.SED(w)
        self.wave_list = obj.wave_list

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength `wave`.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return galsim.Deconvolve(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`.

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        return ChromaticDeconvolution(self.obj / flux_ratio, **self.kwargs)


class ChromaticAutoConvolution(ChromaticObject):
    """A special class for convolving a ChromaticObject with itself.

    It is equivalent in functionality to `galsim.Convolve([obj,obj])`, but takes advantage of
    the fact that the two profiles are the same for some efficiency gains.

    Initialization
    --------------

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = lambda w: (obj.SED(w))**2
        self.wave_list = obj.wave_list

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength `wave`.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return galsim.AutoConvolve(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`.

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        import math
        if flux_ratio >= 0.:
            return ChromaticAutoConvolution( self.obj * math.sqrt(flux_ratio), **self.kwargs )
        else:
            return ChromaticObject(self).withScaledFlux(flux_ratio)


class ChromaticAutoCorrelation(ChromaticObject):
    """A special class for correlating a ChromaticObject with itself.

    It is equivalent in functionality to
        galsim.Convolve([obj,obj.rotate(180.*galsim.degrees)])
    but takes advantage of the fact that the two profiles are the same for some efficiency gains.

    Initialization
    --------------

    @param obj              The object to be convolved with itself.
    @param real_space       Whether to use real space convolution.  [default: None, which means
                            to automatically decide this according to whether the objects have hard
                            edges.]
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """
    def __init__(self, obj, **kwargs):
        self.obj = obj.copy()
        self.kwargs = kwargs
        self.separable = obj.separable
        if self.separable:
            self.SED = lambda w: (obj.SED(w))**2
        self.wave_list = obj.wave_list

    def evaluateAtWavelength(self, wave):
        """Evaluate this chromatic object at a particular wavelength `wave`.

        @param wave  Wavelength in nanometers.

        @returns the monochromatic object at the given wavelength.
        """
        return galsim.AutoCorrelate(self.obj.evaluateAtWavelength(wave), **self.kwargs)

    def withScaledFlux(self, flux_ratio):
        """Multiply the flux of the object by `flux_ratio`.

        @param flux_ratio   The factor by which to scale the flux.

        @returns the object with the new flux.
        """
        import math
        if flux_ratio >= 0.:
            return ChromaticAutoCorrelation( self.obj * math.sqrt(flux_ratio), **self.kwargs )
        else:
            return ChromaticObject(self).withScaledFlux(flux_ratio)

class InterpolatedChromaticObject(ChromaticObject):
    def __init__(self, waves):
        self.waves = waves
        self.separable = False
        self.SED = lambda w: 1.0
        self.wave_list = np.array([], dtype=float)
        self.base_norm = 1.0

        if self.waves is not None:
            self.waves = np.sort(self.waves)

            # Make the objects and their images between which we are going to interpolate.
            self.objs = [ self.simpleEvaluateAtWavelength(wave) for wave in waves ]

            nyquist_dx_vals = [ obj.nyquistScale() for obj in self.objs ]
            use_dx = min(nyquist_dx_vals)

            possible_im_sizes = [ obj.SBProfile.getGoodImageSize(use_dx, 1.0) for obj in self.objs ]
            use_n = max(possible_im_sizes)

            self.stepK_vals = [obj.stepK() for obj in self.objs ]
            self.maxK_vals = [obj.maxK() for obj in self.objs ]
            self.ims = [ obj.drawImage(scale=use_dx, nx=use_n, ny=use_n, method='no_pixel') for obj in self.objs ]
            self.dx = use_dx
            self.n_im = use_n

    def evaluateAtWavelength(self, wave, force_eval = False):
        if self.waves is not None and not force_eval:
            im, stepk, maxk = self._image_at_wavelength(wave)
            return galsim.InterpolatedImage(im, _force_stepk = stepk*self.dx, _force_maxk = maxk*self.dx)
        else:
            return self.base_norm*self.simpleEvaluateAtWavelength(wave)

    def _image_at_wavelength(self, wave):
        if self.waves is None:
            raise RuntimeError("Requested image at some wavelength when doing direct calculation!")
        if wave < min(self.waves) or wave > max(self.waves):
            raise RuntimeError("Requested wavelength is outside the allowed range: %f to %f nm"%(min(self.waves),max(self.waves)))
        lower_idx = np.searchsorted(self.waves, wave) - 1
        frac = (wave - self.waves[lower_idx]) / (self.waves[lower_idx+1] - self.waves[lower_idx])
        im = frac*self.ims[lower_idx+1] + (1.0-frac)*self.ims[lower_idx]
        stepk = frac*self.stepK_vals[lower_idx+1] + (1.0-frac)*self.stepK_vals[lower_idx]
        maxk = frac*self.maxK_vals[lower_idx+1] + (1.0-frac)*self.maxK_vals[lower_idx]
        return self.base_norm*im, stepk, maxk

    def drawImage(self, bandpass, force_eval=False, image=None, **kwargs):
        if (self.waves is None) or force_eval:
            return ChromaticObject.drawImage(self, bandpass, image=image, **kwargs)

        # setup output image (semi-arbitrarily using the bandpass effective wavelength)
        prof0 = self.evaluateAtWavelength(bandpass.effective_wavelength) # should use simpleEvaluateAtWavelength?
        image = prof0.drawImage(image=image, setup_only=True, **kwargs)
        # Remove from kwargs anything that is only used for setting up image:
        if 'dtype' in kwargs: kwargs.pop('dtype')
        if 'scale' in kwargs: kwargs.pop('scale')
        if 'wcs' in kwargs: kwargs.pop('wcs')

        # determine combined self.wave_list and bandpass.wave_list
        wave_list = self._getCombinedWaveList(bandpass)

        # decide on integrator
        integrator = galsim.integ.DirectImageIntegrator(galsim.integ.midpt)

        # merge self.wave_list into bandpass.wave_list if using a sampling integrator
        bandpass = galsim.Bandpass(galsim.LookupTable(wave_list, bandpass(wave_list),
                                                      interpolant='linear'))

        import time
        t1 = time.time()
        integral, stepk, maxk = integrator(self._image_at_wavelength, bandpass)
        t2 = time.time()
        # For now, pretend we have no information about the maxk and stepk that should be used.
        int_im = galsim.InterpolatedImage(integral, _force_stepk = stepk*self.dx,
                                          _force_maxk = maxk*self.dx)
        t3 = time.time()

        # For performance profiling, store the number of evaluations used for the last integration
        # performed.  Note that this might not be very useful for ChromaticSum instances, which are
        # drawn one profile at a time, and hence _last_n_eval will only represent the final
        # component drawn.
        self._last_n_eval = integrator.last_n_eval

        # Apply integral to the initial image appropriately.
        # Note: Don't do image = integral and return that for add_to_image==False.
        #       Remember that python doesn't actually do assignments, so this won't update the
        #       original image if the user provided one.  The following procedure does work.
        image = int_im.drawImage(image=image, **kwargs)
        return image


class ChromaticOpticalPSF(InterpolatedChromaticObject):
    def __init__(self, diam, aberrations, min_wave, max_wave, n_wave, **kwargs):
        # First, take the basic info.
        self.diam = diam
        self.aberrations = aberrations
        self.kwargs = kwargs

        # Take user-specified choice for number of wavelengths to use for initial calculation.
        if n_wave is not None:
            waves = np.linspace(min_wave, max_wave, n_wave)
        else:
            waves = None
        super(ChromaticOpticalPSF, self).__init__(waves)

    def simpleEvaluateAtWavelength(self, wave):
        lam_over_diam = 1.e-9 * (wave / self.diam) * (galsim.radians / galsim.arcsec)
        aberrations = self.aberrations / wave
        ret = galsim.OpticalPSF(lam_over_diam=lam_over_diam, aberrations=aberrations, **self.kwargs)
        return ret

    def withScaledFlux(self, flux_ratio):
        ret = self.copy()
        ret.base_norm *= flux_ratio
        return ret
