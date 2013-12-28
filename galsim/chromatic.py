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
Definitions for GalSim classes implementing wavelength-dependence.

This file extends the base GalSim classes by allowing them to be wavelength dependent.  This allows
one to implement wavelength-dependent PSFs or galaxies with color gradients.
"""

import galsim

class ChromaticObject(object):
    """Base class for defining wavelength dependent objects.
    """
    def draw(self, wave, throughput, image=None, add_to_image=False):
        """Draws an Image of a chromatic object as observed through a bandpass filter.

        Draws the image wavelength by wavelength, i.e., using a Riemann sum.  This is often slow,
        especially if each wavelength involves multiple convolutions, for example.  Hence, subclasses
        should try to override this method if they can think of clever ways to reduce the number of
        convolutions needed (see ChromaticConvolve below, for instance)

        @param wave        Wavelengths in nanometers describing bandpass filter.  For now these are
                           assumed to be linearly spaced.  This needs to be indexable, iterable, and
                           have at least two elements.
        @param throughput  Dimensionless throughput at each wavelength, i.e., the probability that
                           any particular photon at the corresponding wavelength is detected.  This
                           needs to be indexable, iterable, and have the same number of elements as
                           wave.

        @returns           The drawn image.
        """

        #Assume that wave is linear, and compute constant dwave.
        dwave = wave[1] - wave[0]

        #Initialize Image from first wavelength.
        prof = self.evaluateAtWavelength(wave[0]) * throughput[0] * dwave
        image = prof.draw(image=image, add_to_image=add_to_image)

        #And now add in the remaining wavelengths drawing each one in turn
        for w, tp in zip(wave, throughput)[1:]:
            prof = self.evaluateAtWavelength(w) * tp * dwave
            prof.draw(image=image, add_to_image=True)

        return image

    def __add__(self, other):
        return galsim.ChromaticAdd([self, other])

    def __iadd__(self, other):
        self = galsim.ChromaticAdd([self, other])
        return self

class ChromaticBaseObject(ChromaticObject):
    """Construct chromatic versions of the galsim.base objects.

    This class extends the base GSObjects in basy.py by adding SEDs.  Useful to consistently generate
    the same galaxy observed through different filters, or, with the ChromaticAdd class, to construct
    multi-component galaxies, each with a different SED. For example, a bulge+disk galaxy could be
    constructed:

    >>> bulge_wave, bulge_photons = user_function_to_get_bulge_spectrum()
    >>> disk_wave, disk_photons = user_function_to_get_disk_spectrum()
    >>> bulge = galsim.ChromaticBaseObject(galsim.Sersic, bulge_wave, bulge_photons,
                                           n=4, half_light_radius=1.0)
    >>> disk = galsim.ChromaticBaseObject(galsim.Sersic, disk_wave, disk_photons,
                                          n=1, half_light_radius=2.0)
    >>> gal = galsim.ChromaticAdd([bulge, disk])

    Notice that positional and keyword arguments which apply to the specific base class being
    generalized (e.g., n=4, half_light_radius = 1.0 for the bulge component) are passed to
    ChromaticBaseObject after the base object type and SED.

    The SED is specified with a wavelength array and a photon array.  At present, the wavelength
    array is assumed to be linear.  The photon array specifies the distribution of source photons
    over wavelength, i.e., it is proportional to f_lambda * lambda.  Flux normalization is set
    such that the Riemann sum over the wavelength and photon array is equal to the total number of
    photons in an infinite aperture.
    """
    def __init__(self, gsobj, wave, photons, *args, **kwargs):
        """Initialize ChromaticBaseObject.

        @param gsobj    One of the GSObjects defined in base.py.  Possibly works with other GSObjects
                        too.
        @param wave     Wavelength array in nanometers for SED.  Must be indexable, iterable, and
                        have at least two elements.
        @param photons  Photon array in photons per nanometers.  Must be indexable, iterable, and
                        be the same length as wave.
        @param args
        @param kwargs   Additional positional and keyword arguments are forwarded to the gsobj
                        constructor.
        """
        self.photons = galsim.LookupTable(wave, photons)
        self.gsobj = gsobj(*args, **kwargs)
        # We classify ChromaticObjects as either seperable, in which case the wavelength-dependent
        # surface-brightness can be written f(x,y,lambda) = g(x,y) * h(lambda), or as inseperable,
        # in which case the profile cannot be decomposed into factors.
        self.separable = True

    def applyShear(self, *args, **kwargs):
        self.gsobj.applyShear(*args, **kwargs)

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return self.photons(wave) * self.gsobj

class ChromaticAdd(ChromaticObject):
    """Add ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat spectra.
    """
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return galsim.Add([obj.evaluateAtWavelength(wave) for obj in self.objlist])

    def applyShear(self, *args, **kwargs):
        for obj in self.objlist:
            obj.applyShear(*args, **kwargs)

    def draw(self, wave, throughput, image=None, add_to_image=False):
        # most efficient to just add up one component at a time...?
        image = self.objlist[0].draw(wave, throughput, image=image)
        for obj in self.objlist[1:]:
            image = obj.draw(wave, throughput, image=image, add_to_image=True)

class ChromaticConvolve(ChromaticObject):
    """Convolve ChromaticObjects and/or GSObjects together.  GSObjects are treated as having flat
    spectra.
    """
    def __init__(self, objlist):
        self.objlist = objlist

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        return galsim.Convolve([obj.evaluateAtWavelength(wave) for obj in self.objlist])

    def draw(self, wave, throughput, image=None, add_to_image=False):
        # expand any `ChromaticConvolve`s in the object list
        L = len(self.objlist)
        i = 0
        while i < L:
            if isinstance(self.objlist[i], ChromaticConvolve):
                # found a ChromaticConvolve object, so unpack its obj.objlist to end of self.objlist,
                # delete obj from self.objlist, and undate list length `L` and list index `i`.
                L += len(self.objlist[i].objlist) - 1
                # appending to the end of the objlist means we don't have to recurse in order to
                # expand a hierarchy of `ChromaticAdd`s; we just have to keep going until the end of
                # the ever-expanding list.
                self.objlist.extend(self.objlist[i].objlist)
                del self.objlist[i]
                i -= 1
            i += 1

        # Now split up any `ChromaticAdd`s:
        # This is the tricky part.  Some notation first:
        #     I(f(x,y,lambda)) denotes the integral over wavelength of a chromatic surface brightness
        #         profile f(x,y,lambda).
        #     C(f1, f2) denotes the convolution of surface brightness profiles f1 & f2.
        #     A(f1, f2) denotes the addition of surface brightness profiles f1 & f2.
        #
        # In general, chromatic s.b. profiles can be classified as either seperable or inseperable.
        # Write seperable profiles as g(x,y) * h(lambda), and leave inseperable profiles as
        # f(x,y,lambda).
        # We will suppress the arguments `x`, `y`, `lambda`, hereforward, but generally an `f` refers
        # to an inseperable profile, a `g` refers to the spatial part of a seperable profile, and an
        # `h` refers to the spectral part of a seperable profile.
        #
        # Now, analyze a typical scenario, a bulge+disk galaxy model (each of which is seperable,
        # e.g., an SED times an exponential profile for the disk, and another SED times a DeV profile
        # for the bulge).  Suppose the PSF is inseperable.  (Chromatic PSF's will generally be
        # inseper since we usually think of the PSF as being normalized to unit integral, no matter
        # the wavelength we evaluate it at.)  Say there's also an achromatic pix to convolve with.
        # The formula for this might look like:
        #
        # img = I(C(A(bulge, disk), PSF, pix))
        #     = I(C(A(g1*h1, g2*h2), f3, g4))                #note pix is lambda-independent
        #     = I(A(C(g1*h1, f3, g4)), C(A(g2*h2, f3, g4)))  #distribute the A over the C
        #     = A(I(C(g1*h1, f3, g4)), I(C(g2*h2, f3, g4)))  #distribute the A over the I
        #     = A(C(g1,I(h1*f3),g4), C(g2,I(h2*f3),g4))      #factor lambda-independent terms out of I
        #
        # The result is that the integral is now inside the convolution, meaning we only have to
        # compute two convolutions instead of a convolution for each wavelength at which we evaluate
        # the integrand.  This is technique, making an `effective` PSF profile for each of the bulge
        # and disk, is a significant savings for most use cases.
        # In general, we make effective profiles by splitting up `ChromaticAdd`s and collecting the
        # inseperable terms to do integration first, and convolution last when possible.

        # Here is the logic to turn I(C(A(...))) into A(C(..., I(...)))
        returnme = False
        for i, obj in enumerate(self.objlist):
            if isinstance(obj, ChromaticAdd):  #say obj.objlist = [A,B,C]
                returnme = True
                del self.objlist[i] #remove the add object from self.objlist
                convlist = self.objlist #the remaining items to be convolved with each of A,B,C
                tmplist = list(convlist)
                tmplist.append(obj.objlist[0]) #add A to convolve list
                tmpobj = ChromaticConvolve(tmplist)
                image = tmpobj.draw(wave, throughput, image=image, add_to_image=add_to_image)
                for summand in obj.objlist[1:]: #now do B, C, and so on...
                    tmplist = list(convlist)
                    tmplist.append(summand)
                    tmpobj = ChromaticConvolve(tmplist)
                    image = tmpobj.draw(wave, throughput, image=image, add_to_image=True)
        if returnme:
            return image

        # If program gets this far, the objects in self.objlist should be atomic (non-ChromaticAdd
        # and non-ChromaticConvolve), both seperable and inseperable.
        # Classify these into lists.
        sep_profs = []
        insep_profs = []
        sep_photons = []
        for obj in self.objlist:
            if obj.separable:
                if isinstance(obj, galsim.GSObject):
                    sep_profs.append(obj) # The g(x,y)'s (see above)
                else:
                    sep_profs.append(obj.gsobj)
                sep_photons.append(obj.photons) # The h(lambda)'s (see above)
            else:
                insep_profs.append(obj) # The f(x,y,lambda)'s (see above)

        # make an effective profile from inseparables and the chromatic part of separables
        dwave = wave[1] - wave[0] # assume wavelengths are linear
        # and what happens when a list is empty?  I dunno...
        # start assembling monochromatic profiles into an effective profile
        mono_prof = galsim.Convolve([insp.evaluateAtWavelength(wave[0]) for insp in insep_profs])
        mono_prof *= throughput[0] * dwave
        for s in sep_photons:
            mono_prof *= s(wave[0])
        effective_prof_image = mono_prof.draw()
        for w, tp in zip(wave, throughput)[1:]:
            mono_prof = galsim.Convolve([insp.evaluateAtWavelength(w) for insp in insep_profs])
            mono_prof *= tp * dwave
            for s in sep_photons:
                mono_prof *= s(w)
            mono_prof.draw(image=effective_prof_image, add_to_image=True)

        effective_prof = galsim.InterpolatedImage(effective_prof_image)
        sep_profs.append(effective_prof)
        final_prof = galsim.Convolve(sep_profs)
        return final_prof.draw(image=image, add_to_image=add_to_image)


class ChromaticShiftAndDilate(ChromaticObject):
    """Class representing chromatic profiles whose wavelength dependence consists of shifting and
    dilating a fiducial profile.

    By simply shifting and dilating a fiducial PSF, a variety of physical wavelength dependencies can
    be effected.  For instance, differential chromatic refraction is just shifting the PSF center as
    a function of wavelength.  The wavelength-dependence of seeing, and the wavelength-dependence of
    the diffraction limit are dilations.  This class can compactly represent all of these effects.
    See tests/test_chromatic.py for an example.
    """
    def __init__(self, gsobj,
                 shift_fn=None, dilate_fn=None,
                 **kwargs):
        """
        @param gsobj      Fiducial galsim.base profile to shift and dilate.
        @param shift_fn   Function that takes wavelength in nanometers and returns a
                          galsim.Position object, or parameters which can be transformed into a
                          galsim.Position object (dx, dy).
        @param dilate_fn  Function that takes wavelength in nanometers and returns a dilation
                          scale factor.
        """
        self.gsobj = gsobj(**kwargs)
        self.shift_fn = shift_fn
        self.dilate_fn = dilate_fn
        self.separable = False

    def applyShear(self, *args, **kwargs):
        self.gsobj.applyShear(*args, **kwargs)

    def evaluateAtWavelength(self, wave):
        """
        @param wave  Wavelength in nanometers.
        @returns     GSObject for profile at specified wavelength
        """
        PSF = self.gsobj.copy()
        PSF.applyDilation(self.dilate_fn(wave))
        PSF.applyShift(self.shift_fn(wave))
        return PSF
