
Wavelength-dependent Profiles
#############################

Real astronomical objects emit photons at a range of wavelengths according to a potentially
complicated spectral energy distribution (SED).  These photons then may be affected differently
by the atmosphere and optics as part of the point-spread function (PSF).  Then they typically
pass through a bandpass filter with a variable transmission as a function of wavelength.
Finally, there may be other wavelength-dependent effects when converting from photons to
electrons in the sensor.

GalSim supplies a number of tools to simulate these chromatic effects. 
An `SED` is used to define the SED of the objects.  There are a variety of options as to the units
of the input SED function; e.g. photons/cm^2/nm/sec, ergs/cm^2/Hz/s, etc.  There are also ways
to adjust the normalization of the SED to give a particular observed magnitude when observed
in through particular `Bandpass`.  There is also a dimensionless option, which may be appropriate
for defining chromatic PSFs. 

The `Bandpass` class represents a spectral throughput function, which could be an
entire imaging system throughput response function (reflection off of mirrors, transmission through
filters, lenses and the atmosphere, quantum efficiency of detectors), or individual pieces thereof.
Both a `Bandpass` and the `SED` are necessary to compute the relative contribution of each
wavelength of a `ChromaticObject` to a drawn image.

Then there are a number of kinds of `ChromaticObject` to define the wavelength dependence of an
object's surface brightness profile.  The simplest one is when the spatial and spectral
dependencies are separable; i.e. every part of the profile has the same SED.  In this case,
one forms the `ChromaticObject` simply by multiplying a `GSObject` by an `SED`::

    >>> obj = galsim.Sersic(n=2.3, half_light_radius=3.5)
    >>> sed = galsim.SED('CWW_Sbc_ext.sed', wave_type'Ang', flux_type='flambda')
    >>> chromatic_object = obj * sed

Other more complicated kinds of chromatic profiles are subclasses of `ChromaticObject` and
have their own initialization arguments.  See the listings below.

To draw any kind of `ChromaticObject`, you call its :meth:`~galsim.ChromaticObject.drawImage`
method, which works largely the same as :meth:`~galsim.GSObject:drawImage`, but requires a
`galsim.Bandpass` argument to define what bandpass is being used for the observation::

    >>> gband = galsim.Bandpass(lambda w:1.0, wave_type='nm', blue_limit=410, red_limit=550)
    >>> image = chromatic_obj.drawImage(gband)

The transformation methods of `ChromaticObject`, like :meth:`~ChromaticObject.dilate` and 
:meth:`~ChromaticObject.shift`,
can also accept as an argument a function of wavelength (in nanometers)
that returns a wavelength-dependent dilation, shift, etc.  These can be used to implement
chromatic PSFs.  For example, a diffraction limited PSF might look like::

    >>> psf500 = galsim.Airy(lam_over_diam=2.0)
    >>> chromatic_psf = ChromaticObject(psf500).dilate(lambda w:(w/500)**1.0)


Spectral Energy Distributions
-----------------------------

.. autoclass:: galsim.SED
    :members:

Bandpass Filters
----------------

.. autoclass:: galsim.Bandpass
    :members:

Chromatic Profiles
------------------

.. autoclass:: galsim.ChromaticObject
    :members:

.. autoclass:: galsim.ChromaticAtmosphere
    :members:

.. autoclass:: galsim.ChromaticOpticalPSF
    :members:

.. autoclass:: galsim.ChromaticAiry
    :members:

.. autoclass:: galsim.ChromaticRealGalaxy
    :members:

.. autoclass:: galsim.InterpolatedChromaticObject
    :members:

.. autoclass:: galsim.ChromaticSum
    :members:

.. autoclass:: galsim.ChromaticConvolution
    :members:

.. autoclass:: galsim.ChromaticDeconvolution
    :members:

.. autoclass:: galsim.ChromaticAutoConvolution
    :members:

.. autoclass:: galsim.ChromaticAutoCorrelation
    :members:

.. autoclass:: galsim.ChromaticTransformation
    :members:

.. autoclass:: galsim.ChromaticFourierSqrtProfile
    :members:

Spectral Correlated Noise
-------------------------

.. autoclass:: galsim.CovarianceSpectrum
    :members:
