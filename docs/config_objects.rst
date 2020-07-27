Config Objects
==============

GalSim defines a number of object types, which correspond to the GSObject types in the python code.
Some are designed to be appropriate for describing PSFs and others for describing galaxies.
GalSim does not enforce this distinction in any way; you can use any object type in the ``psf``
or ``gal`` fields.  But normally, the ``psf`` field would use types from the `PSF Types`
below, and the ``gal`` field would use types from the `Galaxy Types`.
There are also some `Generic Types` that can be appropriate for either.

Each object type sets a number of other items that either must or may be present in the dict
for the object (i.e. in the top level ``psf`` or ``gal`` field or farther down in the dict where
an object is being defined, such is in a 'List' object type).  These attributes are given as
bullet items for each type defined below.

There are also some `Other attributes` that are allowed for any object (or
sometimes just galaxies), regardless of what type they are.

And finally, it is possible to define your own object type,
which we describe in `Custom Object Types`.

PSF Types
---------

* 'Moffat'  A Moffat profile: :math:`I(r) \sim (1 + (r/r_0)^2)^{-\beta}`, where :math:`r_0` is the ``scale_radius``.

    * ``beta`` = *float_value* (required)
    * ``scale_radius`` = *float_value* (exactly one of ``scale_radius``, ``fwhm`` or ``half_light_radius`` is required)
    * ``half_light_radius`` = *float_value* (exactly one of ``scale_radius``, ``fwhm`` or ``half_light_radius`` is required)
    * ``fwhm`` = *float_value* (exactly one of ``scale_radius``, ``fwhm`` or ``half_light_radius`` is required)
    * ``trunc`` = *float_value* (optional)  The profile can be truncated to 0 at some radius if desired. The default is no truncation.

* 'Airy'  A simple Airy disk. (Typically one would convolve this by some model of the atmospheric component of the PSF.  cf. 'Convolution' below.)

    * ``lam_over_diam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required)  Lambda / telescope_diameter converted to units of arcsec (or whatever units you want your profile to use).
    * ``lam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required).  This should be the wavelength in nanometers.
    * ``diam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required).  This should be the telescope diameter in meters.
    * ``obscuration`` = *float_value* (default = 0)  The linear size of an obstructing secondary mirror as a fraction of the full mirror size.
    * ``scale_unit`` = *str_value* (default = 'arcsec') Units to be used for internal calculations when calculating lam/diam.

* 'Kolmogorov'  A Kolmogorov turbulent spectrum: :math:`T(k) \sim \exp(-D(k)/2)`, where :math:`D(k) = 6.8839 (\lambda k/2\pi r0)^{5/3}`.

    * ``lam_over_r0`` = *float_value* (exactly one of ``lam_over_r0``, ``fwhm`` or ``half_light_radius`` or both ``lam`` and ``r0`` is required) Lambda / r0 converted to units of arcsec (or whatever units you want your profile to use).
    * ``lam`` = *float_value* (exactly one of ``lam_over_r0``, ``fwhm`` or ``half_light_radius`` or both ``lam`` and ``r0`` is required) The wavelength in nanometers.
    * ``r0`` = *float_value* (exactly one of ``lam_over_r0``, ``fwhm`` or ``half_light_radius`` or both ``lam`` and ``r0`` is required) The Fried parameter in meters.
    * ``r0_500`` = *float_value* (optional, in lieu of ``r0``).  The Fried parameter in meters at a wavelength of 500 nm.  The correct ``r0`` value will be calculated using the standard relation r0 = r0_500 * (lam/500)``1.2.
    * ``fwhm`` = *float_value* (exactly one of ``lam_over_r0``, ``fwhm`` or ``half_light_radius`` or both ``lam`` and ``r0`` is required)
    * ``half_light_radius`` = *float_value* (exactly one of ``lam_over_r0``, ``fwhm`` or ``half_light_radius`` or both ``lam`` and ``r0`` is required)
    * ``scale_unit`` = *str_value* (default = 'arcsec') Units to be used for internal calculations when calculating lam/r0.

* 'OpticalPSF'  A PSF from aberrated telescope optics.

    * ``lam_over_diam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required)
    * ``lam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required).  This should be the wavelength in nanometers.
    * ``diam`` = *float_value* (either ``lam_over_diam`` or both ``lam`` and ``diam`` required).  This should be the telescope diameter in meters.
    * ``defocus`` = *float_value* (default = 0) The defocus value, using the Noll convention for the normalization. (Noll index 4)
    * ``astig1`` = *float_value* (default = 0) The astigmatism in the y direction, using the Noll convention for the normalization. (Noll index 5)
    * ``astig2`` = *float_value* (default = 0) The astigmatism in the x direction, using the Noll convention for the normalization. (Noll index 6)
    * ``coma1`` = *float_value* (default = 0)The defocus value, using the Noll convention for the normalization. (Noll index 7)
    * ``coma2`` = *float_value* (default = 0)The defocus value, using the Noll convention for the normalization. (Noll index 8)
    * ``trefoil1`` = *float_value* (default = 0)The defocus value, using the Noll convention for the normalization. (Noll index 9)
    * ``trefoil2`` = *float_value* (default = 0) The defocus value, using the Noll convention for the normalization. (Noll index 10)
    * ``spher`` = *float_value* (default = 0)The defocus value, using the Noll convention for the normalization. (Noll index 11)
    * ``aberrations`` = *list* (optional) This is an alternative way to specify the above aberrations.  You can just give them as a list of values using the Noll convention for the ordering (starting at Noll index 1, since there is no 0).  With this syntax, you may go to as high order as you want.
    * ``circular_pupil`` = *bool_value* (default = True) Whether the pupil should be circular (True, the default) or square (False).
    * ``obscuration`` = *float_value* (default = 0) The linear dimension of a central obscuration as a fraction of the pupil linear dimension.
    * ``interpolant`` = *str_value* (default = 'quintic') Which interpolant to use for the constructed InterpolatedImage object describing the PSF profile.
    * ``oversampling`` = *float_value* (default = 1.5) How much oversampling of the internal image is needed relative to the Nyquist scale of the corresponding Airy profile.  The more aberrated the PSF, the higher this needs to be.
    * ``pad_factor`` = *float_value* (default = 1.5) How much padding to put around the edge of the internal image of the PSF.
    * ``suppress_warning`` = *bool_value* (default = False) Whether to suppress warnings about possible aliasing problems due to the choices of ``oversampling`` and ``pad_factor``.
    * ``max_size`` = *float_value* (optional) If the PSF will only be used to draw images of some size, you can set the OpticalPSF class to not build the internal image of the PSF (much) larger than that.  This can help speed up calculations if GalSim natively decides to build a very large image of the PSF, when the wings never actually affect the final image.
    * ``nstruts`` = *int_value* (default = 0) How many support struts to include.
    * ``strut_thick`` = *float_value* (default = 0.05) How thick the struts should be as a fraction of the pupil diameter.
    * ``strut_angle`` = *angle_value* (default = 0 degrees) The counter-clockwise angle between the vertical and one of the struts.  The rest will be spaced equally from there.
    * ``pupil_plane_im`` = *str_value* (optional) Instead of using strut-related parameters to define the pupil plane geometry, you can use this parameter to specify a file-name containing an image of the pupil plane.
    * ``pupil_angle`` = *angle_value* (default = 0 degrees) When specifying a pupil_plane_im, use this parameter to rotate it by some angle defined counter-clockwise with respect to the vertical.
    * ``scale_unit`` = *str_value* (default = 'arcsec') Units to be used for internal calculations when calculating lam/diam.

* 'ChromaticAtmosphere'  A chromatic PSF implementing both differential chromatic diffraction (DCR) and wavelength-dependent seeing.  See `ChromaticAtmosphere` for valid combinations that can be used to set the zenith and parallactic angles needed for DCR.
    * ``base_profile`` = *object* (required) The base profile to use for the profile shape at a given reference wavelength
    * ``base_wavelength`` = *float_value* (required) The wavelength at which the PSF has the base profile
    * ``alpha`` = *float_value* (default = -0.2)  Power law index for wavelength-dependent seeing.
    * ``zenith_angle`` = *Angle_value* (optional) The zenith angle.
    * ``parallactic_angle`` = *Angle_value*  (optional) The parallactic angle.
    * ``zenith_coord`` = *CelestialCoord*  (optional)  The (ra,dec) coordinate of the zenith.
    * ``HA`` = *Angle_value* (optional)  Hour angle of the observation.
    * ``latitude`` = *Angle_value* (optional)  Latitude of the observatory.
    * ``pressure`` = *float_value* (default = 69.328)  Air pressure in kPa.
    * ``temperature`` = *float_value* (default = 293.15)  Temperature in K.
    * ``H2O_pressure`` = *float_value* (default = 1.067)  Water vapor pressure in kPa.

Galaxy Types
------------

* 'Exponential'  A radial exponential profile: :math:`I(r) \sim \exp(-r/r_0)`, where :math:`r_0` is the ``scale_radius``.

    * ``scale_radius`` = *float_value* (exactly one of ``scale_radius`` or ``half_light_radius`` is required)
    * ``half_light_radius`` = *float_value* (exactly one of ``scale_radius`` or ``half_light_radius`` is required)

* 'Sersic'  A Sersic profile: :math:`I(r) \sim \exp(-(r/r_0)^{1/n}) = \exp(-b (r/r_e)^{1/n})`, where :math:`r_0` is the ``scale_radius`` and :math:`r_e` is the ``half_light_radius``.

    * ``n`` = *float_value* (required)
    * ``half_light_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)
    * ``scale_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)
    * ``trunc`` = *float_value* (optional)  The profile can be truncated to 0 at some radius if desired. The default is no truncation.
    * ``flux_untruncated`` = *bool_value* (default = False)  Set the profile such that the specified flux corresponds to that of the untruncated profile.  Valid only when ``trunc`` > 0; ignored otherwise.

* 'DeVaucouleurs'  A DeVaucouleurs profile: :math:`I(r) \sim \exp(-(r/r_0)^{1/4}) = \exp(-b (r/r_e)^{1/4})` (aka n=4 Sersic).

    * ``scale_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)
    * ``half_light_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)
    * ``trunc`` = *float_value* (optional)  The profile can be truncated to 0 at some radius if desired. The default is no truncation.
    * ``flux_untruncated`` = *bool_value* (default = False)  Set the profile such that the specified flux corresponds to that of the untruncated profile.  Valid only when ``trunc`` > 0; ignored otherwise.

* 'Spergel'  A profile based on the Spergel (2010) paper with the form: :math:`I(r) \sim (r/r_0)^\nu * K_\nu(r/r_0)` where :math:`r_0` is the ``scale_radius`` and :math:`K_\nu` is the modified Bessel function of the second kind.

    * ``nu`` = *float_value* (required)
    * ``half_light_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)
    * ``scale_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required)

* 'RealGalaxy'  A real galaxy image, typically taken from a deep HST image, deconvolved by the original PSF.  Note that the deconvolution implies that this cannot be draw with photon shooting (``image.draw_method = 'phot'``). This requires that ``input.real_catalog`` be specified and uses the following fields:

    * ``index`` = *int_value* (default = 'Sequence' from 0 to ``real_catalog.nobjects``-1; only one of ``id`` or ``index`` may be specified)  Which item in the catalog to use. Special: If ``index`` is either a 'Sequence' or 'Random' and ``last`` or ``max`` (respectively) is not specified, then it is automatically set to ``real_catalog.nobjects-1``.
    * ``id`` = *str_value* (only one of ``id`` or ``index`` may be specified)  The ID in the catalog of the object to use.
    * ``x_interpolant`` = *str_value* (default = 'Quintic')  What to use for interpolating between pixel centers.  Options are 'Nearest', 'Linear', 'Cubic', 'Quintic', 'Sinc', or 'LanczosN', where the 'N' after 'Lanczos' should be replaced with the integer order to use for the Lanczos filter.
    * ``k_interpolant`` = *str_value* (default = 'Quintic')  What to use for interpolating between pixel centers in Fourier space, for convolution.  Options are 'Nearest', 'Linear', 'Cubic', 'Quintic', 'Sinc', or 'LanczosN', where the 'N' after 'Lanczos' should be replaced with the integer order to use for the Lanczos filter.  See docstring for this class for caveats about changing this parameter.
    * ``flux`` = *float_value* (default = catalog value) If set, this works as described below.  However, 'RealGalaxy' has a different default.  If ``flux`` is omitted, the flux of the actual galaxy in the catalog is used.
    * ``pad_factor`` = *float_value* (default = 4) Amount of zero-padding to use around the image when creating the InterpolatedImage.  See docstring for this class for caveats about changing this parameter.
    * ``noise_pad_size`` = *float* (optional) If provided, then the original image is padded to a larger image of this size using the noise profile of the original image.  This is important if you are using ``noise.whiten`` or ``noise.symmetrize``.  You want to make sure the image has the original noise all the way to the edge of the postage stamp.  Otherwise, the edges will have the wrong noise profile.
    * ``num`` = *int_value* (default = 0)  If ``input.real_catalog`` is a list, this indicates which number catalog to use.

* 'RealGalaxyOriginal'  This is the same as 'RealGalaxy' except that the profile is _not_ deconvolved by the original PSF.  So this is the galaxy *as observed* in the original image. This requires that ``input.real_catalog`` be specified and uses the same fields as 'RealGalaxy'.  This may be more useful than the deconvolved version.  For example, unlike 'RealGalaxy', it can be drawn with photon shooting (``image.draw_method = 'phot'``).
* 'COSMOSGalaxy'  Either a real or parametric galaxy from the COSMOS catalog. This requires that ``input.cosmos_catalog`` be specified and uses the following fields:

    * ``index`` = *int_value* (default = 'Sequence' from 0 to ``cosmos_catalog.nobjects``-1)  Which item in the catalog to use. Special: If ``index`` is either a 'Sequence' or 'Random' and ``last`` or ``max`` (respectively) is not specified, then it is automatically set to ``real_catalog.nobjects-1``.
    * ``gal_type`` = *str_vale* (required, unless ``real_catalog.use_real`` is False, in which case 'parametric') Which type of galaxy to use.  Options are 'real' or 'parametric'.
    * ``deep`` = *bool_value* (default = False) Whether the flux and size should be rescaled to approximate a galaxy catalog with a limiting mag of 25 in F814W, rather than 23.5.
    * ``noise_pad_size`` = *float* (optional) If provided, then the original image is padded to a larger image of this size using the noise profile of the original image.  This is important if you are using ``noise.whiten`` or ``noise.symmetrize``.  You want to make sure the image has the original noise all the way to the edge of the postage stamp.  Otherwise, the edges will have the wrong noise profile.
    * ``sersic_prec`` = *float_value* (default = 0.05) The desired precision on the Sersic index n in parametric galaxies.  GalSim is significantly faster if it gets a smallish number of Sersic values, so it can cache some of the calculations and use them again the next time it gets a galaxy with the same index.  If ``sersic_prec`` is 0.0, then use the exact value of index n from the catalog.  But if it is >0, then round the index to that precision.
    * ``num`` = *int_value* (default = 0)  If ``input.real_catalog`` is a list, this indicates which number catalog to use.

* 'InclinedExponential'  The 2D projection of a 3D exponential profile: :math:`I(R,z) \sim \mathrm{sech}^2 (z/h_s) * \exp(-R/R_s)` at an arbitrary inclination angle, where :math:`h_s` is the ``scale_height`` and :math:`R_s` is the ``scale_radius``  The base profile is inclined along the y-axis, so if you want a different position angle, you should add a ``rotate`` field.

    * ``inclination`` = *angle_value* (required) The inclination angle, defined such that 0 degrees is face-on and 90 degrees is edge-on.
    * ``half_light_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required) The half-light radius as an alternative to ``scale_radius``.
    * ``scale_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required) The scale_radius, R_s.
    * ``scale_height`` = *float_value* (exactly one of ``scale_height`` or ``scale_h_over_r`` is required) The scale height, h_s.
    * ``scale_h_over_r`` = *float_value* (exactly one of ``scale_height`` or ``scale_h_over_r`` is required) The ratio h_s/R_s as an alternative to ``scale_height``.

* 'InclinedSersic'  Like 'InclinedExponential', but using a `Sersic` profile in the plane of the disc.

    * ``n`` = *float_value* (required)
    * ``half_light_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required) The half-light radius as an alternative to ``scale_radius``.
    * ``scale_radius`` = *float_value* (exactly one of ``half_light_radius`` or ``scale_radius`` is required) The scale radius, R_s.
    * ``trunc`` = *float_value* (optional)  The profile can be truncated to 0 at some radius if desired. The default is no truncation.
    * ``flux_untruncated`` = *bool_value* (default = False)  Set the profile such that the specified flux corresponds to that of the untruncated profile.  Valid only when ``trunc`` > 0; ignored otherwise.
    * ``scale_height`` = *float_value* (exactly one of ``scale_height`` or ``scale_h_over_r`` is required) The scale height, h_s.
    * ``scale_h_over_r`` = *float_value* (exactly one of ``scale_height`` or ``scale_h_over_r`` is required) The ratio h_s/R_s as an alternative to ``scale_height``.

* 'DeltaFunction'  A delta function profile with a specified flux.  This is typically not used for galaxies, but rather for stars in a scene that includes both.  So when convolved by the PSF, the stars will have the profile from the PSF, but the correct flux.
* 'RandomKnots'  A profile made of a sum of a number of delta functions distributed according to either a Gaussian profile or a given specified profile.  This is intended to represent knots of star formation, so it would typically be added to a smooth disk component and have the same size and shape.

    * ``npoints`` = *int_value* (required) How many points to include.
    * ``half_light_radius`` = *float_value* (either ``half_light_radius`` or ``profile`` is required) The expectation of the half light radius, setting the overall scale of the Gaussian profile.  Note: any given realized profile will not necessarily have exactly this half-light radius.
    * ``profile`` = *GSObject* (either ``half_light_radius`` or ``profile`` is required) The profile you want to use for the distribution of knots.


Generic Types
-------------

* 'Gaussian'  A circular Gaussian profile: :math:`I(r) \sim \exp(-r^2 / (2 \sigma^2))`.  This is not all that appropriate for either PSFs or galaxies, but as it is extremely simple, it is often useful for very basic testing, as many measured properties of the profile have analytic values.

    * ``sigma`` = *float_value* (exactly one of ``sigma``, ``fwhm`` or ``half_light_radius`` is required)
    * ``fwhm`` = *float_value* (exactly one of ``sigma``, ``fwhm`` or ``half_light_radius`` is required)
    * ``half_light_radius`` = *float_value* (exactly one of ``sigma``, ``fwhm`` or ``half_light_radius`` is required)

* 'InterpolatedImage'  A profile described simply by a provided image (given in a fits file).

    * ``image`` = *str_value* (required)  The file name from which to read the image.
    * ``x_interpolant`` = *str_value* (default = 'Quintic')  What to use for interpolating between pixel centers.  Options are 'Nearest', 'Linear', 'Cubic', 'Quintic', 'Sinc', or 'LanczosN', where the 'N' after 'Lanczos' should be replaced with the integer order to use for the Lanczos filter.
    * ``k_interpolant`` = *str_value* (default = 'Quintic')  What to use for interpolating between pixel centers in Fourier space, for convolution.  Options are 'Nearest', 'Linear', 'Cubic', 'Quintic', 'Sinc', or 'LanczosN', where the 'N' after 'Lanczos' should be replaced with the integer order to use for the Lanczos filter.  See docstring for this class for caveats about changing this parameter.
    * ``normalization`` = *str_value* (default = 'flux')  What normalization to assume for the input image.  Options are ('flux' or 'f') or ('surface brightness' or 'sb').
    * ``scale`` = *float_value* (default = 'GS_SCALE' entry from the fits header, or 1 if not present)  What pixel scale to use for the image pixels.
    * ``pad_factor`` = *float_value* (default = 4) Amount of zero-padding to use around the image when creating the SBInterpolatedImage.  See docstring for this class for caveats about changing this parameter.
    * ``noise_pad_size`` = *float_value* (optional; required if ``noise_pad`` is provided) If non-zero, then the original image is padded to a larger image of this size using the noise specified in ``noise_pad``.
    * ``noise_pad`` = *str_value* (optional; required if ``noise_pad_size`` is provided) Either a filename to use for padding the ``image`` with noise according to a noise correlation function, or a variance value to pad with Gaussian noise.
    * ``pad_image`` = *str_value* (optional) The name of an image file to use for directly padding the ``image`` (deterministically) rather than padding with noise.
    * ``calculate_stepk`` = *bool_value* (default = True) Recalculate optimal Fourier space separation for convolutions?  Can lead to significant optimization compared to default values.
    * ``calculate_maxk`` = *bool_value* (default = True) Recalculate optimal Fourier space total k range for convolutions?  Can lead to significant optimization compared to default values.
    * ``use_true_center`` = *bool_value* (default = True) Whether to use the true center of the provided image as the nominal center of the profile (True) or round up to the nearest integer value (False).
    * ``hdu`` = *int_value* (default = the primary HDU for uncompressed images, or the first extension for compressed images) Which HDU to use from the input FITS file.

* 'Box'  A rectangular boxcar profile: :math:`I(x,y) \sim H(w/2-|x|) H(h/2-|y|)`,
  where :math:`H` is the Heaviside function, :math:`w` is ``width`` and :math:`h` is ``height``.

    * ``width`` = *float_value* (required)  The full width of the profile.
    * ``height`` = *float_value* (required)  The full height of the profile.

* 'Pixel'  A square boxcar profile: :math:`I(x,y) \sim H(s/2-|x|) H(s/2-|y|)`,
  where :math:`H` is the Heaviside function and :math:`s` is ``scale``.
  This is equivalent to a 'Box' type with ``width`` = ``height`` (called ``scale`` here).  Note however, that the default rendering method already correctly accounts for the pixel response, so normally you will not need to use this as part of the PSF (or galaxy).

    * ``scale`` = *float_value* The pixel scale, which is the width and height of the pixel.

* 'TopHat'  A circular tophat profile: :math:`I(r) \sim H(r-|r|)`, where :math:`H` is the
  Heaviside function and :math:`r` is ``radius``.

    * ``radius`` = *float_value* The radius of the circular tophat profile.

* 'Sum' or 'Add'  Add several profiles together.

    * ``items`` = *list* (required)  A list of profiles to be added.

* 'Convolution' or 'Convolve'  Convolve several profiles together.

    * ``items`` = *list* (required)  A list of profiles to be convolved.

* 'List'  Select profile from a list.

    * ``items`` = *list* (required)  A list of profiles.
    * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)  Which item in the list to select each time.


Other Attributes
----------------

There are a number of transformation attributes that are always allowed for any object type.
Some of these operations do not commute with each other, so the order is important.
The following transformations will be applied in the order given here, which corresponds
roughly to when they occur to the light packet traveling through the universe.

The first few, ``flux``, ``dilate``, ``ellip``, and ``rotate``, are typically used to define the
intrinsic profile of the object.
The next two, ``magnify`` and ``shear``, are typically used to define how the profile is
modified by lensing.
The next one, ``shift``, is used to shift the position of the galaxy relative to its nominal
position on the sky.

* ``flux`` = *float_value* (default = 1.0)  Set the flux of the object in ADU. Note that the component ``items`` in a 'Sum' can also have fluxes specified, which can be used as fractional fluxes (e.g. 0.6 for the disk and 0.4 for the bulge). Then the outer level profile can set the real flux for the whole thing.
* ``dilate`` or ``dilation`` = *float_value* (optional)  Dilate the profile by a given scale, preserving the flux.
* ``ellip`` = *shear_value* (optional)  Shear the profile by a given shear to give the profile some non-round intrinsic shape.
* ``rotate`` or ``rotation`` = *angle_value* (optional)  Rotate the profile by a given angle.
* ``scale_flux`` = *float_value* (optional) Factor by which to scale the flux of the galaxy profile.
* ``magnify`` or ``magnification`` = *float_value* (optional)  Magnify the profile by a given scale, preserving the surface brightness.
* ``shear`` = *shear_value* (optional)  Shear the profile by a given shear.
* ``shift`` = *pos_value* (optional)  Shift the centroid of the profile by a given amount relative to the center of the image on which it will be drawn.
* ``skip`` = *bool_value* (default=False)  Skip this object.
* ``sed`` = *SED* (optional)  If desired, you may set an SED to use for the object.  See `SED Field` below for details.

There are also a few special attributes that are only allowed for the top-level ``gal`` field,
not for objects that are part of an aggregate object like 'Sum', 'Convolution' or 'List'
and not for ``psf``.

* ``resolution`` = *float_value* (optional)  If the base profile allows a ``half_light_radius`` parameter, and the psf is able to calculate a ``half_light_radius``, then it is permissible to specify a resolution: resolution = r_gal / r_psf  (where r_gal and r_psf are the half-light radii) in lieu of specifying the ``half_light_radius`` of the galaxy explicitly. This is especially useful if the PSF size is generated randomly.
* ``signal_to_noise`` = *float_value* (optional)  You may specify a signal-to-noise value rather than a ``flux``. Our definition of the S/N derives from a weighted integral of the flux in the drawn image:
  :math:`S = \sum W(x,y) I(x,y) / \sum W(x,y)` where :math:`W(x,y)` is taken to be a matched filter, so :math:`W(x,y) = I(x,y)`. (Note: This currently requires ``draw_method = 'fft'``.  It is a bit trickier to do this for photon shooting, and we have not enabled that yet.)
* ``redshift`` = *float_value* (optional)  The redshift of the galaxy.  This is required when using 'NFWHaloShear' or 'NFWHaloMagnification'.

Custom Object Types
-------------------

To define your own object type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a function that will be used
to build a GalSim `GSObject`.

The build function should have the following functional form:

.. code-block:: py

    def BuildCustomObject(config, base, ignore, gsparams, logger):
        """Build a custom GSObject of some sort

        Parameters:
            config:     The configuration dict of the object being built
            base:       The base configuration dict.
            ignore:     A list of parameters that might be in the config dict,
                        but which may be ignored.  i.e. it is not an error for
                        these items to be present.
            gsparams:   An optional dict of items used to build a GSParams object
                        (may be None).
            logger:     An optional logger object to log progress (may be None).

        Returns:
            gsobject, safe

        The returned gsobject is the built GSObject instance, and safe is a bool
        value that indicates whether the object is safe to reuse for future stamps
        (e.g. if all the parameters used to build this object are constant and will
        not change for later stamps).
        """
        # If desired, log some output.
        if logger:
            logger.debug("Starting work on building CustomObject")

        # The gsparams are passed around using a dict so they can be easily added to.
        # At this point, we would typically convert them to a regular GSParams
        # instance to use when building the GSObject.
        if gsparams:
            gsparams = galsim.GSParams( **gsparams )

        # If you need a random number generator, this is the one to use.
        rng = base['rng']

        # Build the GSObject
        # Probably something complicated that you want this function to do.
        gsobject = [...]

        safe = False  # typically, but set to True if this object is safe to reuse.
        return gsobject, safe

The ``base`` parameter is the original full configuration dict that is being used for running the
simulation.  The ``config`` parameter is the local portion of the full dict that defines the object
being built, e.g. ``config`` might be ``base['gal']`` or it might be farther down as an item in
the ``items`` attribute of a 'List' or 'Sum' object.

Then, in the Python module, you need to register this function with some type name, which will
be the value of the ``type`` attribute that triggers running this function::

    galsim.config.RegisterObjectType('CustomObject', BuildCustomObject)

.. autofunction:: galsim.config.RegisterObjectType

If the builder will use a particular input type, you should let GalSim know this by specifying
the ``input_type`` when registering.  E.g. if the builder expects to use an input ``dict`` file
to define some properties that will be used, you would register this fact using::

    galsim.config.RegisterObjectType('CustomObject', BuildCustomObject,
                                     input_type='dict')

The input object can be accessed in the build function as e.g.::

    input_dict = galsim.config.GetInputObj('dict', config, base, 'CustomObject')
    ignore = ignore + ['num']

The last argument is just used to help give sensible error messages if there is some problem,
but it should typically be the name of the object type being built.  When you are using an
input object, the 'num' attribute is reserved for indicating which of possibly several input
objects (``dict`` in this case) to use.  You should not also define a num attribute that has
some meaning for this object type.  If you are using the ``ignore`` parameter to check for extra
invalid parameters, you would thus want to add 'num' to the list.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_object.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_object

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_object``,
which will read your file and execute the registration command to add the object to the list
of valid object types.

Then you can use this as a valid object type:

.. code-block:: yaml

    gal:
        type: CustomObject
        ...

For examples of custom objects, see :download:`des_psfex.py <../galsim/des/des_psfex.py>`
and :download:`des_shapelet.py <../galsim/des/des_shapelet.py>`
in the galsim.des module, which define custom object types DES_PSFEx and DES_Shapelet.  These objects are used by :download:`draw_psf.yaml <../examples/des/draw_psf.yaml>`
in the ``GalSim/examples/des`` directory.
It may also be helpful to look at the GalSim implementation of some of the included object builders (click on the ``[source]`` links):

.. autofunction:: galsim.config.gsobject._BuildAdd

.. autofunction:: galsim.config.gsobject._BuildConvolve

.. autofunction:: galsim.config.gsobject._BuildList

.. autofunction:: galsim.config.gsobject._BuildOpticalPSF


SED Field
---------

If you want your object to have a non-trivial wavelength dependence, you can include an
``sed`` parameter to define its `SED`.  Currently, there is only one defined
type to use for the SED, but the code is written in a modular way to allow for
other types, including custom SED types.

* 'FileSED' is the default type here, and you may omit the type name when using it.

    * ``file_name`` = *str_value* (required)  The file to read in.
    * ``wave_type`` = *str_value* (required)  The unit of the wavelengths in the file ('nm' or 'Ang' or variations on these -- cf. `SED`)
    * ``flux_type`` = *str_value* (required)  The type of spectral density or dimensionless normalization used in the file ('flambda', 'fnu', 'fphotons' or '1' -- cf. `SED`)
    * ``redshift`` = *float_value* (optional)  If given, shift the spectrum to the given redshift.  You can also specify the redshift as an object-level parameter if preferred.
    * ``norm_flux_density`` = *float_value* (optional)  Set a normalization value of the flux density at a specific wavelength.  If given, ``norm_wavelength`` is required.
    * ``norm_wavelength`` = *float_value* (optional)  The wavelength to use for the normalization flux density.
    * ``norm_flux`` = *float_value* (optional)  Set a normalization value of the flux over a specific bandpass.  If given, ``norm_bandpass`` is required.
    * ``norm_bandpass`` = *Bandpass* (optional)  The bandpass to use for the normalization flux.

You may also define your own custom SED type in the usual way
with an importable module where you define a custom Builder class and register it with GalSim.
The class should be a subclass of `galsim.config.SEDBuilder`.

.. autoclass:: galsim.config.SEDBuilder
    :members:

Then, as usual, you need to register this type using::

    galsim.config.RegisterSEDType('CustomSED', CustomSEDBuilder())

.. autofunction:: galsim.config.RegisterSEDType

and tell the config parser the name of the module to load at the start of processing.

.. code-block:: yaml

    modules:
        - my_custom_sed

Then you can use this as a valid sed type:

.. code-block:: yaml

    gal:
        ...
        sed:
            type: CustomSED
            ...
