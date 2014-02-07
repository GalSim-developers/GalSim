# Copyright 2012, 2013 The GalSim developers:
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

Implementation is done by constructing GSObjects as functions of wavelength.  Draw methods then
integrate over wavelength while also multiplying in a throughput function.

Possible uses include galaxies with color gradients, automatically drawing a given galaxy through
different filters, or  implementing wavelength-dependent point spread functions.
"""

import numpy
import copy

import galsim
import galsim.integ
import galsim.dcr

class ChromaticObject(object):
    """Base class for defining wavelength dependent objects.
    """
    def draw(self, bandpass, image=None, scale=None, gain=1.0, wmult=1.0,
             add_to_image=False, use_true_center=True, offset=None,
             integrator=None, **kwargs):
        """Base chromatic image draw function, for subclasses that don't choose to override it.
        Since most galsim use cases will probably finish with a convolution,
        ChromaticConvolution.draw() will often be the draw method used in practice.

        The task of ChromaticObject.draw(bandpass) is to integrate a chromatic surface brightness
        profile multiplied by the throughput of `bandpass`, over the wavelength interval indicated
        by `bandpass`.  To evaluate the integrand, this method creates a function that accepts a
        single argument, a wavelength in nanometers, and returns an image of the surface brightness
        profile at that wavelength using GSObject.draw().

        The wavelength integration can be done using several different schemes, all defined in
        integ.py.  The default, galsim.integ.midpoint_int_image, implements the midpoint rule,
        i.e., it makes rectangular (in wavelength) approximations to the integral over equally
        spaced subintervals.  Fancier available integrators include
        galsim.integ.trapezoidal_int_image, galsim.integ.simpsons_int_image, and
        galsim.integ.globally_adaptive_GK_int_image, which implements a globally adaptive
        Gauss-Kronrod integration scheme.

        Image integrators can receive additional keyword parameters through **kwargs.  For
        instance, galsim.integ.midpoint_int_image can accept `N`, the number of subintervals into
        which to divide the bandpass wavelength interval.  Likewise, `N` is also an option for
        galsim.integ.trapezoidal_int_image and galsim.integ.simpsons_int_image.  For
        galsim.integ.globally_adaptive_GK_int_image, a relative error tolerance `rel_err` can be
        passed instead.

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

        # default integrator uses midpoint rule.
        if integrator is None:
            integrator = galsim.integ.midpoint_int_image
        # setup output image (semi-arbitrarily using midpoint of bandpass wavelength interval)
        middle_wavelength = 0.5 * (bandpass.blue_limit + bandpass.red_limit)
        prof0 = self.evaluateAtWavelength(middle_wavelength) * bandpass(middle_wavelength)
        prof0 = prof0._fix_center(image, scale, offset, use_true_center, reverse=False)
        image = prof0._draw_setup_image(image, scale, wmult, add_to_image)

        # integrand returns an galsim.Image at each wavelength
        def f_image(w):
            prof = self.evaluateAtWavelength(w) * bandpass(w)
            tmpimage = image.copy()
            tmpimage.setZero()
            prof.draw(image=tmpimage, gain=gain, wmult=wmult,
                      add_to_image=False, use_true_center=use_true_center, offset=offset)
            return tmpimage

        # Do the wavelength integral
        integral = integrator(f_image, bandpass.blue_limit, bandpass.red_limit, **kwargs)

        # Clear image if add_to_image is False
        if not add_to_image:
            image.setZero()
        image += integral
        return image

    def __add__(self, other):
        """Add together two ChromaticObjects, or a ChromaticObject and a GSObject.
        """
        return galsim.ChromaticSum([self, other])

    def copy(self):
        cls = self.__class__
        ret = cls.__new__(cls)
        ret.__dict__.update(self.__dict__)
        return ret

    # reminder to developers to add apply* methods to subclasses below
    def createDilated(scale):
        ret = self.copy()
        ret.applyDilation(scale)
        return ret

    def createMagnified(mu):
        ret = self.copy()
        ret.applyMagnification(mu)
        return ret

    def createSheared(*args, **kwargs):
        ret = self.copy()
        ret.applyShear(*args, **kwargs)
        return ret

    def createLensed(g1, g2, mu):
        ret = self.copy()
        ret.applyLensing(g1, g2, mu)
        return ret

    def createRotated(theta):
        ret = self.copy()
        ret.applyRotation(theta)
        return ret

    def createShifted(*args, **kwargs):
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
        self.gsobj = gsobj
        # Chromaticized GSObjects are separable into spatial (x,y) and spectral (lambda) factors.
        self.separable = True

    # Make op* and op*= work to adjust the flux of the object
    def __imul__(self, other):
        self.gsobj.scaleFlux(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # Apply following transformations to the underlying GSObject
    def scaleFlux(self, scale):
        self.gsobj.scaleFlux(scale)

    def applyShear(self, *args, **kwargs):
        self.gsobj.applyShear(*args, **kwargs)

    def applyDilation(self, scale):
        self.gsobj.applyDilation(scale)

    def applyShift(self, *args, **kwargs):
        self.gsobj.applyShift(*args, **kwargs)

    def applyExpansion(self, scale):
        self.gsobj.applyExpansion(scale)

    def applyMagnification(self, mu):
        self.gsobj.applyMagnification(mu)

    def applyLensing(self, g1, g2, mu):
        self.gsobj.applyLensing(g1, g2, mu)

    def applyRotation(self, theta):
        self.gsobj.applyRotation(theta)

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
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        """Evaluate underlying GSObjects scaled by their attached SEDs evaluated at `wave`.

        @param wave  Wavelength in nanometers.
        @returns     galsim.Sum GSObject for profile at specified wavelength.
        """
        return galsim.Sum([obj.evaluateAtWavelength(wave) for obj in self.objlist])

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

    # Make op* and op*= work to adjust the flux of the object
    def __imul__(self, other):
        for obj in self.objlist:
            obj.scaleFlux(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    # apply following transformations to all underlying summands
    def scaleFlux(self, scale):
        for obj in self.objlist:
            obj.scaleFlux(scale)

    def applyShear(self, *args, **kwargs):
        for obj in self.objlist:
            obj.applyShear(*args, **kwargs)

    def applyDilation(self, scale):
        for obj in self.objlist:
            obj.applyDilation(scale)

    def applyShift(self, *args, **kwargs):
        for obj in self.objlist:
            obj.applyShift(*args, **kwargs)

    def applyExpansion(self, scale):
        for obj in self.objlist:
            obj.applyExpansion(scale)

    def applyMagnification(self, mu):
        for obj in self.objlist:
            obj.applyMagnification(mu)

    def applyLensing(self, g1, g2, mu):
        for obj in self.objlist:
            obj.applyLensing(g1, g2, mu)

    def applyRotation(self, theta):
        for obj in self.objlist:
            obj.applyRotation(theta)

class ChromaticConvolution(ChromaticObject):
    """Convolve ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat
    spectra.
    """
    def __init__(self, objlist):
        self.objlist = []
        for obj in objlist:
            if isinstance(obj, ChromaticConvolution):
                self.objlist.extend(obj.objlist)
            else:
                self.objlist.append(obj)

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
        which include separable portions, which can then be integrated without before doing any
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
        if integrator is None:
            integrator = galsim.integ.midpoint_int_image
        # Only make temporary changes to objlist...
        objlist = list(self.objlist)

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

        # setup output image (semi-arbitrarily using midpoint of bandpass wavelength interval)
        middle_wavelength = 0.5 * (bandpass.blue_limit + bandpass.red_limit)
        prof0 = self.evaluateAtWavelength(middle_wavelength) * bandpass(middle_wavelength)
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
                    sep_profs.append(obj.gsobj) # more g(x,y)'s
                    sep_SED.append(obj.SED) # The h(lambda)'s (see above)
            else:
                insep_profs.append(obj) # The f(x,y,lambda)'s (see above)

        if insep_profs == []: # didn't find any inseparable profiles
            def f(w):
                term = bandpass(w)
                for s in sep_SED:
                    term *= s(w)
                return term
            multiplier = galsim.integ.int1d(f, bandpass.blue_limit, bandpass.red_limit)
        else: # did find inseparable profiles
            multiplier = 1.0
            # setup image for effective profile
            middle_wavelength = 0.5 * (bandpass.blue_limit + bandpass.red_limit)
            mono_prof0 = galsim.Convolve([p.evaluateAtWavelength(middle_wavelength)
                                          for p in insep_profs])
            mono_prof0 = mono_prof0._fix_center(image=None, scale=None, offset=None,
                                                use_true_center=True, reverse=False)
            mono_prof_image = mono_prof0._draw_setup_image(image=None, scale=None, wmult=wmult,
                                                           add_to_image=False)
            # Modify image size/scale wrt requested oversampling
            if iimult is not None:
                mono_prof_image = galsim.ImageD(mono_prof_image.array.shape[0] * iimult,
                                                mono_prof_image.array.shape[1] * iimult,
                                                scale=(mono_prof_image.scale * 1.0/iimult))
            # integrand for effective profile
            def f_image(w):
                mono_prof = galsim.Convolve([insp.evaluateAtWavelength(w) for insp in insep_profs])
                mono_prof *= bandpass(w)
                for s in sep_SED:
                    mono_prof *= s(w)
                tmpimage = mono_prof_image.copy()
                tmpimage.setZero()
                mono_prof.draw(image=tmpimage, wmult=wmult)
                return tmpimage
            # wavelength integral
            effective_prof_image = integrator(f_image, bandpass.blue_limit, bandpass.red_limit,
                                              **kwargs)
            # Image -> InterpolatedImage
            # It could be useful to cache this result if drawing more than one object with the same
            # PSF+SED combination.  This naturally happens in a ring test or when fitting the
            # parameters of a galaxy profile to an image when the PSF is constant.
            effective_prof = galsim.InterpolatedImage(effective_prof_image)
            # append effective profile to separable profiles (which should all be GSObjects)
            sep_profs.append(effective_prof)
        # finally, convolve and draw.
        final_prof = multiplier * galsim.Convolve(sep_profs)
        return final_prof.draw(image=image, gain=gain, wmult=wmult, add_to_image=add_to_image,
                               use_true_center=use_true_center, offset=offset)

    # Make op* and op*= work to adjust the flux of the object by altering flux of the
    # first object in self.objlist
    def __imul__(self, other):
        self.objlist[0].scaleFlux(other)
        return self

    def __mul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        ret *= other
        return ret

    def scaleFlux(self, scale):
        self.objlist[0].scaleFlux(scale)


class ChromaticShiftAndDilate(ChromaticObject):
    """Class representing chromatic profiles whose wavelength dependence consists of shifting and
    dilating a fiducial profile.

    By simply shifting and dilating a fiducial PSF, a variety of physical wavelength dependencies can
    be effected.  For instance, differential chromatic refraction is just shifting the PSF center as
    a function of wavelength.  The wavelength-dependence of seeing, and the wavelength-dependence of
    the diffraction limit are dilations.  This class can compactly represent all of these effects.
    """
    def __init__(self, gsobj, shift_fn=None, dilate_fn=None, SED=None):
        """
        @param gsobj      Fiducial GSObject profile to shift and dilate.
        @param shift_fn   Function that takes wavelength in nanometers and returns a
                          galsim.Position object, or parameters which can be transformed into a
                          galsim.Position object (dx, dy).  The fiducial GSObject is then shifted
                          this amount when evaluated at specified wavelengths.
        @param dilate_fn  Function that takes wavelength in nanometers and returns a dilation
                          scale factor.  The fiducial GSObject is then dilated this amount when
                          evaluated at specified wavelengths.
        """
        self.gsobj = gsobj
        self.separable = False

        # Default is no shifting
        if shift_fn is None:
            self.shift_fn = lambda x: (0,0)
        else:
            self.shift_fn = shift_fn

        # Default is no dilating
        if dilate_fn is None:
            self.dilate_fn = lambda x: 1.0
        else:
            self.dilate_fn = dilate_fn

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        profile = self.gsobj.copy()
        profile.applyDilation(self.dilate_fn(wave))
        profile.applyShift(self.shift_fn(wave))
        return profile

    def scaleFlux(self, scale):
        self.gsobj.scaleFlux(scale)

    # The apply* transforms don't commute with shifting and dilating, so cannot simply
    # alter self.gsobj like above...

class ChromaticAtmosphere(ChromaticShiftAndDilate):
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
        self.gsobj = base_obj
        self.separable = False
        self.dilate_fn = lambda w: (w/base_wavelength)**(alpha)
        base_refraction = galsim.dcr.get_refraction(base_wavelength, zenith_angle, **kwargs)
        def shift_fn(w):
            shift_magnitude = galsim.dcr.get_refraction(w, zenith_angle, **kwargs)
            shift_magnitude -= base_refraction
            shift_magnitude = shift_magnitude / galsim.arcsec
            shift = (shift_magnitude*numpy.sin(position_angle.rad()),
                     shift_magnitude*numpy.cos(position_angle.rad()))
            return shift
        self.shift_fn = shift_fn
