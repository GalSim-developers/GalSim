Changes from v1.0 to v1.1:
--------------------------

Non-backward-compatible API changes:

We believe that these changes will not impose much hardship on the majority
of GalSim users.  The funtions are either expected to be rarely used, or the
change does not affect the most common uses of the function.

* Changed the `xw` and `yw` parameters of the `Pixel` constructor to a
  single `scale` parameter. (Issue #364)
  * `pix = Pixel(xw=scale)` should now be either `pix = Pixel(scale=scale)`
    or simply `pix = Pixel(scale)`.
* Added new `Box` class to take up the functionality that had been `Pixel`
  with unequal values of `xw` and `yw`. (Issue #364)
  * `box = Pixel(xw=width, yw=height)` should now be either
    `box = Box(width=width, height=height)` or `box = Box(width, height)`.
* Changed the handling of the `scale` and `init_value` parameters of the
  `Image` constructor, so that now they have to be named keyword arguments
  rather than positional arguments. (Issue #364)
  * `im = ImageF(nx, ny, scale, init_val)` should now be
    `im = ImageF(nx, ny, scale=scale, init_value=init_val)`.
* Changed `Angle.wrap()` to return the wrapped angle rather than modifying the
  original. (Issue #364)
  * `angle.wrap()` should now be `angle = angle.wrap()`.
* Removed the previously deprecated `Ellipse` and `AtmosphericPSF` classes.
  Also removed `PhotonArray` from the python layer, since it is only used
  by the C++ layer. (Issue #364)
* Changed Bounds methods `addBorder`, `shift`, and `expand` to return new
  Bounds objects rather than changing the original (in the python layer
  only). (Issue #364)
* Stopped importing everything from the `galsim._galsim` namespace into the
  main `galsim` namespace.  Now only the classes and functions which we
  document and use in examples are imported into the `galsim` namespace.
  The rest are considered implementation details, and are not guaranteed
  to maintain backward compatibility of syntax and/or functionality in future
  versions. (Issue #511)
  * e.g. `galsim.SBGaussian` should now be `galsim._galsim.SBGaussian`.
  * Or better, switch to using the documented `galsim.Gaussian` class instead.
  * Similarly for other `SB*` classes along with a few other undocumented
    classes and functions.
* Changed the name of the parameter used to seed various Deviate objects
  from `lseed` to `seed`.  The documentation described it as an unnamed arg,
  rather than a named kwarg, so probably no one was using it by name.
  But if you were, just change `lseed` to `seed`. (Issue #511)
* Added a default value for rng parameter to CorrelatedNoise objects.  The
  consequence of this is that if you were relying on the order of the 
  construction parameters, you will need to rearrange, since rng is no
  longer first (because it is a kwarg now). (Issue #526)
  * `CorrelatedNoise(rng, image)` should now be `CorrelatedNoise(image, 
    rng=rng)` or `CorrelatedNoise(rng=rng, image=image)`.
  * `UncorrelatedNoise(rng, wcs, variance)` should now be `UncorrelatedNoise(
    variance, rng=rng, wcs=wcs)` or `UncorrelatedNoise(rng=rng, wcs=wcs,
    variance=variance)`.
  * `getCOSMOSNoise(rng, file_name)` should now be `getCOSMOSNoise(file_name,
    rng)` or `getCOSMOSNoise(rng=rng, file_name=file_name)`.


Other changes to the API

For these changes, we are currently still allowing the old syntax for ease of
transition, but that syntax is now discouraged.  It is usually marked in the
code as being obsolete.  At some point (probably version 1.2) use of the old
syntax will raise a DeprecationWarning, and with version 2.0, it will be
removed.

* Changed the name of the `dx` parameter in the `draw`, `drawShoot`, `drawK`
  methods of `GSObject` and the constructors of `InterpolatedImage` and
  `CorrelatedNoise` to the name `scale`. (Issue #364)
* Changed the `dx_cosmos` parameter of `getCOSMOSNoise` to `cosmos_scale`.
  (Issue #364)
* Combined the old `Image`, `ImageView` and `ConstImageView` arrays of class
  names into a single python layer `Image` class that automatically constructs
  the appropriate C++ image class as an attribute. (Issue #364)
  * `im = ImageF(...)` and similar is still valid.
  * `im = ImageF(...)` _may_ now be written as `im = Image(...)`.  That is,
    the numpy.float32 type is the default data type if you do not specify
    something else either through the type letter or the `dtype` parameter.
  * `im = ImageViewF(...)` and similar should now be `im = ImageF(...)`
    (preserving the same type letter S, I, F or D).
  * `im = ConstImageViewF(...)` and similar should now be
    `im = ImageF(..., make_const=True)` (again preserving the type letter).
  * `im = Image[type](...)` should now be `Image(..., dtype=type)`
  * `im = ImageView[type](numpy_array.astype(type))` should now be
    `im = Image(numpy_array.astype(type))`.  i.e. the data type inherits
    from the numpy_array argument when appropriate.  If it is already
    the correct type, you do not need the `astype(type)` part.
  * `im = ConstImageView[type](numpy_array.astype(type))` should now be
    `im = Image(numpy_array.astype(type), make_const=True)`
* Changed DES_PSFEx class to take in the original image file to get the correct
  WCS information to convert from image coordinates to world coordinates.  If
  unavailable, then the returned PSF profiles will be in image coordinates.
  The old `scale` parameter in `psfex.getPSF` is obsolete since it is not
  really accurate.  The new behavior accurately converts the PSFEx profile
  between image and world coordinates. (Issue #364)
  * `psfex = galsim.des.DES_PSFEx(psf_file)` `psf = psfex.getPSF(pos, scale)`
    should become `psfex = galsim.des.DES_PSFEx(psf_file, image_file)`
    `psf = psfex.getPSF(pos)`.
* Changed the methods createSheared, createRotated, etc. to more succinct
  names.  The applyShear, applyRotation, etc. methods are also discouraged
  and will eventually be deprecated.  All such usage should be changed to the
  version the returns a new object, rather than modify the object in place.
  (Issue #511)
  * `gal = gal.createSheared(shear)` or `gal.applyShear(shear)` should
    now be `gal = gal.shear(shear)`.
  * `gal = gal.createRotated(theta)` or `gal.applyRotation(theta)` should
    now be `gal = gal.rotate(theta)`.
  * `gal = gal.createDilated(scale)` or `gal.applyDilation(scale)` should
    now be `gal = gal.dilate(scale)`.
  * `gal = gal.createExpanded(scale)` or `gal.applyExpansion(scale)` should
    now be `gal = gal.expand(scale)`.
  * `gal = gal.createMagnified(mu)` or `gal.applyMagnification(mu)` should
    now be `gal = gal.magnify(mu)`.
  * `gal = gal.createShifted(shift)` or `gal.applyShift(shift)` should
    now be `gal = gal.shift(shift)`.
  * `gal = gal.createTransformed(...)` or `gal.applyTransformation(...)` should
    now be `gal = gal.transform(...)`.
  * `gal = gal.createLensed(g1,g2,mu)` or `gal.applyLensing(g1,g2,mu)` should
    now be `gal = gal.lens(g1,g2,mu)`.
* Changed the `setFlux` and `scaleFlux` methods to versions that return new
  objects, rather than change the object in place.  (Issue #511)
  * `gal.setFlux(flux)` should now be `gal = gal.withFlux(flux)`
  * `gal.scaleFlux(flux_ratio)` should now be `gal = gal * flux_ratio`
  * `gal *= flux_ratio` is fine, since python converts it to the above behind
    the scenes.
* Changed the corresponding methods of CorrelatedNoise similarly. (Issue #511)
  * `cn = cn.createSheared(shear)` or `cn.applyShear(shear)` should
    now be `cn = cn.shear(shear)`.
  * `cn = cn.createRotated(theta)` or `cn.applyRotation(theta)` should
    now be `cn = cn.rotate(theta)`.
  * `cn = cn.createDilated(scale)` or `cn.applyDilation(scale)` should
    now be `cn = cn.dilate(scale)`.
  * `cn = cn.createExpanded(scale)` or `cn.applyExpansion(scale)` should
    now be `cn = cn.expand(scale)`.
  * `cn = cn.createMagnified(mu)` or `cn.applyMagnification(mu)` should
    now be `cn = cn.magnify(mu)`.
  * `cn = cn.createShifted(shift)` or `cn.applyShift(shift)` should
    now be `cn = cn.shift(shift)`.
  * `cn = cn.createTransformed(...)` or `cn.applyTransformation(...)` should
    now be `cn = cn.transform(...)`.
  * `cn = cn.createLensed(g1,g2,mu)` or `cn.applyLensing(g1,g2,mu)` should
    now be `cn = cn.lens(g1,g2,mu)`.
* Changed how to set the variance of the various `Noise` methods (including
  `CorrelatedNoise` and all other subclasses of `BaseNoise`). (Issue #511)
  * `n.setVariance(variance)` should now be `n = n.withVariance(variance)`
  * `n.scaleVariance(variance_ratio)` should now be `n = n * variance_ratio`
* Changed the `CorrelatedNoise.convolveWith` method to `convolvedWith`,
  which returns a new object corresponding to the convolvution. (Issue #511)
  * `cn.convolveWith(obj)` should now be `cn = cn.convolvedWith(obj)`.
* Changed the Shapelet.fitImage method to a factory function named
  `FitShapelet` that constructs a new Shapelet object rather than modify
  an existing object in place.
  * `shapelet = galsim.Shapelet(sigma,order); shapelet.fitImage(image)` should
    now be `shapelet = galsim.FitShapelet(sigma, order, image)`
* Changed the name of LVectorSize to ShapeletSize. (Issue #511)
* Changed the `nyquistDx` method to `nyquistScale`.  (Issue #511)
* In general, moved as many classes as possible toward an immutable design.
  The bulk of these changes are the ones listed above, but we also now
  discourage use of setters in various classes that had them.  These methods
  will eventually be deprecated.  The classes are all "light" classes,
  that have trivial constructors, so the preferred syntax is now to create a
  new instance, rather than modifiy an existing instance. (Issue #511)
  * `GaussianNoise.setSigma`
  * `PoissonNoise.setSkyLevel`
  * `CCDNoise`: `setSkyLevel`, `setGain`, `setReadNoise`
  * `BaseDeviate` subclasses: all `set*` methods.
  * `Shear`: `setG1G2`, `setE1E2`, `setEBeta`, `setEta1Eta2`, `setEtaBeta`
  * `Shapelet`: `setSigma`, `setOrder`, `setBVec`, `setNM`, `setPQ`
* Removed `image.getCorrelatedNoise` method.  (Issue #527)
  * `image.getCorrelatedNoise()` should now be `galsim.CorrelatedNoise(image)`.


New WCS classes: (Issue #364)

* Every place in the code that can take a `scale` parameter (e.g. the `Image`
  constructor,  `GSObject.draw()`, `InterpolatedImage`, etc.) can now take a
  `wcs` parameter.  The `scale` parameter is still an option, but now it is
  just shorthand for `wcs = PixelScale(scale)`.
* There are three LocalWCS classes that have a common origin for image and
  world coordinates:
  * `PixelScale` describes a simple scale conversion from pixels to arcsec.
  * `ShearWCS` describes a uniformly sheared coordinate system.
  * `JacobianWCS` describes an arbitrary 2x2 Jacobian matrix.
* There are three non-local UniformWCS classes that have a uniform pixel
  size and shape, but not necessarily the same origin.
  * `OffsetWCS` is a `PixelScale` with the world (0,0) location offset from
    the (0,0) position in image coordinates.
  * `OffsetShearWCS` is a `ShearWCS` with a similar offset.
  * `AffineTransform` is a `JacobianWCS` with an offset.  It is the most
    general possible _uniform_ WCS transformation.  i.e. one where the pixel
    shape is uniform across the image.
* There is one non-uniform EuclideanWCS class that uses Euclidean coordinates
  for the world coordinate system:
  * `UVFunction` is an arbitrary transformation from (x,y) coordinates to
    Euclidean (u,v) coordinates.  It takes arbitrary functions u(x,y) and
    v(x,y) as inputs.  (And optionally x(u,v) and y(u,v) for the inverse
    transformations.)
* There are five CelestialWCS classes that use celestial coordinates for the
  world coordinate system. i.e. the world coordinates are in terms of right
  ascension and declination (RA, Dec).  There is a new CelestialCoord
  class that encapsulates this kind of position on the sphere.
  * `RaDecFunction` takes an arbitrary function radec_func(x,y) that returns
    the RA and Dec.
  * `AstropyWCS` uses the astropy.wcs package to read in a given FITS file.
  * `PyAstWCS` uses the starlink.Ast package to read in a given FITS file.
  * `WcsToolsWCS` uses wcstools commands for a given FITS file.
  * `GSFitsWCS` is GalSim code to read FITS files that use TAN and TPV
    WCS types.  Less flexible than the others, but still useful since
    these are probably the most common WCS types for optical astronomical
    images.  Plus it is quite a bit faster than the others.
* There is a factory function called `FitsWCS` that will try the various
  classes that can read FITS files until it finds one that works.  It will
  revert to `AffineTransform` if it cannot find anything better.
* Another function, `TanWCS`, acts like a WCS class.  It builds a WCS using
  TAN projection and returns a `GSFitsWCS` implementing it.
* When reading in an image from a FITS file, the image will automatically
  try to read the WCS information from the header with the `FitsWCS` function.

See the docstring for `BaseWCS` (the base class for all of these WCS classes)
for information about how to use these classes. Also, check out demo3, demo9,
demo10, and demo11 for example usage.


New `CelestialCoord` class: (Issue #364)

* This class describes a position on the celestial sphere according to
  RightAscension (RA) and Declination (Dec).  These two values are accessible
  as coord.ra and coord.dec.  So it is used by some of the WCS classes for
  the world coordinate positions.
* It has methods to handle a number of spherical trigonometry operations
  that are sometimes required when dealing with celestial coordinates:
  * `coord1.distanceTo(coord2)` returns the great circle distance between two
    coordinates (as a `galsim.Angle`).
  * `coord1.angleBetween(coord2,coord3)` returns the angle between the two
    great circles (coord1-coord2) and (coord1-coord3).
  * `coord1.project(coord2)` applies a tangent plane projection of coord2 with
    respect to the tangent point coord1 using one of 4 possible projection
    schemes specified by the optional keyword `projection`: lambert,
    stereographic, gnomonic, or postel.  See the docstring for this function
    for details.
  * `coord1.deproject(pos)` reverses the projection to go from the position
    on the tangent plane back to celestial coordinates.
  * `coord.precess(from_epoch, to_epoch)` precesses the coordinates to a
    different epoch.
  * `coord.getGalaxyPos()` returns the longitude and latitude in the galactic
    coordinate system as a tuple (el, b).


New chromatic functionality: (Issue2 #467, #520)

* New `Bandpass` class to represent throughput functions, which can either
  represent a complete imaging system, or the individual components thereof,
  such as the atmosphere, mirrors, lenses, filters, detector quantum
  efficiency, etc.
  These can be initialized from 2-column ascii files, python functions,
  `eval()` string expressions, or `galsim.LookupTable`s.
  * Two methods are available to reduce the samples used to evaluate integrals
    over wavelength, and hence reduce the time to draw `ChromaticObject`s:
    * `Bandpass.truncate(blue_limit, red_limit)` returns a truncated Bandpass.
    * `Bandpass.thin(rel_err)` Returns a thinned Bandpass that nonetheless still
      integrates to the unthinned result within `rel_err` relative error.
  * Bandpasses can be multiplied and divided by other Bandpasses.
* New `SED` class to represent stellar and galactic spectra.  These can be
  initialized from 2-column ascii files, python functions, `eval()` string
  expressions or `galsim.LookupTable`s.
  * `sed.withFluxDensity(flux_density, wavelength) returns an SED normalized to
    have `flux_density` (photons/nm) at `wavelength`.
  * `sed.withFlux(flux, bandpass)` returns an SED normalized to have `flux`
    (photons) through `bandpass`.
  * `sed.atRedshift(z)` will return a new SED with redshift set to `z`.
  * SEDs can be added and subtracted from each other, and multiplied and divided
    by constants and functions of wavelength.
* New `ChromaticObject` class to represent wavelength-dependent surface
  brightness profiles:
  * `Chromatic(gsobj, sed)` subclass creates a separable wavelength-dependent
    surface brightness profile: I.e. one that takes the form
    f(x, y, \lambda) = g(x, y) * h(\lambda)
  * `gsobj * sed` is a shortcut for `Chromatic(gsobj, sed)`.
  * Galaxies with color gradients can be created as sums of separable chromatic
    profiles.
    E.g.: `gal = bulge_prof * bulge_sed + disk_prof * disk_sed`.
  * `ChromaticObject.applyDilation()`, `.applyExpansion()`, and `.applyShift()`,
    can take function(s) of wavelength in nanometers as their argument(s), which
    can be used to create a variety of chromatic effects, for instance, a
    wavelength-dependent diffraction limit.
  * `ChromaticAtmosphere(fiducial_PSF, fiducial_wave, zenith_angle)` will modify
    the monochromatic GSObject PSF `fiducial_PSF` defined at wavelength
    `fiducial_wave` to account for atmospheric differential chromatic refraction
    and Kolmogorov chromatic seeing for an observation at `zenith_angle`.
  * `ChromaticObject`s are drawn with almost identical syntax to `GSObjects`.
    The one difference is that `ChromaticObject`s require an additional argument,
    (given first) which is the `Bandpass` throughput function against which to
    integrate over wavelength.
    E.g., `image = chroma_obj.draw(bandpass, ...)`
* Added demo12.py for wavelength dependence examples.

Updates to config options:

Some of these changes are not backwards compatible, but we believe they are
only in rarely used functionality, so we do not expect most users to have to
change their yaml files.

* Changed the previous behavior of the `image.wcs` field to allow several WCS
  types: 'PixelScale', 'Shear', 'Jacobian', 'Affine', 'UVFunction',
  'RaDecFunction', 'Fits', and 'Tan'. (Issue #364)
* Changed the name of `sky_pos` to `world_pos`. (Issue #364)
* Removed `pix` top layer in config structure.  Add `draw_method=no_pixel` to
  do what `pix : None` used to do. (Issue #364)
* Added `draw_method=real_space` to try to use real-space convolution.  This
  had been an option for the psf draw, but not the main draw.  This is only
  possible if there is only one item being convolved with the pixel.
  (Issue #364)
* Added ability to index `Sequence` types by any running index, rather than
  just the default.  i.e. `obj_num`, `image_num`, or `file_num`. (Issue #364)
* Added `Sum` type for value types for which it makes sense: float, int, angle,
  shear, position. (Issue #457)
* Allowed the user to modify or add config parameters from the command line.
  (Issue #479)
* Added a new `image.retry_failures` item that can be set so that if the
  construction of a GSObject fails for any reason, you can ask it to retry.
  An example of this functionality has been added to demo8. (Issue #482)
* Added a new `output.retry_io` item that can be set so that if the output write
  command fails (due to hard drive overloading for example), then it will wait
  a second and try again. (Issue #482)
* Changed the sequence indexing within an image to always start at 0, rather
  than use `obj_num` (which continues increasing through all objects in the run).
  Functionally, this would usually only matter if the number of objects per
  file or image is not a constant.  If the number of objects is constant, the
  automatic looping of the sequencing index essentially did this for you.
  (Issue #487)


Other new features:

* Fixed some bugs in the treatment of correlated noise.  (Issues #526, #528)
* Modify addNoiseSNR() method to return the variance of the noise that was
  added.  (Issue #526)
* Sped up the gzip and bzip2 I/O by using the shell gzip and bzip2 executables
  if they are available on the system. (Issue #344)
* Added some new functions to convert an angle to/from DMS strings.  Sometimes
  handy when dealing with RA or Dec. (Issue #364)
  * `angle.dms()` returns the angle as a string in the form +/-ddmmss.decimal.
  * `angle.hms()` returns the angle as a string in the form +/-hhmmss.decimal.
  * `angle = DMS_Angle(str)` converts from a dms string to a `galsim.Angle`.
  * `angle = HMS_Angle(str)` converts from an hms string to a `galsim.Angle`.
* Added `profile.applyTransformation(dudx, dudy, dvdx, dvdy)` applies a general
  (linear) coordinate transformation to a GSObject profile.  It is a
  generalization of `applyShear`, `applyRotation`, etc.  There is also the
  corresponding `createTransformed` as well. (Issue #364)
* Added `galsim.fits.readFile()` function, which reads a FITS file and returns
  the hdu_list.  Normally, this is equivalent to `pyfits.open(file_name)`, but
  it has a `compression` option that works the same way `compression` works
  for the other `galsim.fits.read*` functions, so it may be convenient
  at times. (Issue #364)
* Added some additional options and changed default interpolant for the lensing
  engine.  (Issue #387)
* Permit users to initialize `OpticalPSF` with a list or array of aberrations,
  as an alternative to specifying each one individually.  (The innards of
  OpticalPSF were also rearranged to use arrays instead of individual values,
  but this is not important for users, just developers.) (Issue #409)
* Added `max_size` optional parameter to OpticalPSF that lets you limit the
  size of the image that it constructs internally. (Issue #478)
* Added option to FitsHeader and FitsWCS to read in SCamp-style text files with
  the header information using the parameter `text_file=True`. (Issue #508)
