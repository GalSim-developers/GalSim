
Config Image Field
==================

The ``image`` field defines some properties about how to draw the images.

Image Field Attributes
----------------------

Some attributes that are allowed for all image types are:

* ``pixel_scale`` = *float_value* (default = 1.0)  The pixel scale, typically taken to be arcsec/pixel. Most size parameters for the profiles are taken to be specified in arcsec. If you would rather specify everything in pixels, just leave off the pixel_scale (or set it to 1.0) and then 1 pixel = 1 arcsec, so everything should work the way you expect.  Or if you want all your units to be degrees or radians or something else, then just set this pixel scale in the same units.
* ``sky_level`` = *float_value* (default = 0.0; only one of ``sky_level`` and ``sky_level_pixel`` is allowed)  The background level of the image in ADU/arcsec^2
* ``sky_level_pixel`` = *float_value* (default = 0.0; only one of ``sky_level`` and ``sky_level_pixel`` is allowed)  The background level of the image in ADU/pixel
* ``index_convention`` = *str_value* (default = 'FITS')  The convention for what to call the lower left pixel of the image.  The standard FITS convention is to call this pixel (1,1).  However, this can be counter-intuitive to people used to C or python indexing.  So if ``index_convention`` is 'C' or 'python' or '0', then the image origin will be considered (0,0) instead.  (While unnecessary to specify explicitly since it is the default, the (1,1) convention may be called 'FITS', 'Fortran' or '1'.)
* ``dtype`` = *str_value* (default = 'np.float32')  The data type to use for the output image.  Allowed values include all the Python and numpy types that are allowed for an `Image`.
* ``random_seed`` = *int_value* or *list* (optional) The random seed to use for random numbers in the simulation.

    * The typical use case is that this is a simple integer value. Then, internally we scramble this value in a deterministic way and use a sequence of seeds based on the scrambled value so that the output is deterministic even when using multiple processes to build each image.
    * If ``random_seed`` is a string, then it is parsed as a integer and treated in the same way as above.  This allows you to do a simple calculation using the ``$`` Eval shorthand to e.g. compute the initial seed from some other kind of ID number you might have in the config file.  It also means you don't have to be careful in your YAML writing to not accidentally write e.g. ``'1234'`` rather than ``1234``.
    * If ``random_seed`` is a dict, it should parse as an *int_value*, but in this case, GalSim will not convert it into a sequence based on a single value.  Rather, it will respect your specification and evaluate the seed for each object as you specifiy.  This let's you do advanced things like use a different cadence for your random numbers (e.g. potentially to repeat the same seed for multiple objects; cf. `Demo 7`).
    * If ``random_seed`` is a list, then multiple random number generators will be available for each object according to the multiple seed specifications.  This is normally used to have one random number behave normally for noise and such, and another one repeat with some cadence (e.g. repeat for each image in an exposure to make sure you generate the same PSFs for multiple CCDs in an exposure).  Whenever you want to use an rng other than the first one, add ``rng_num`` to the field and set it to the number of the rng you want to use in this list.  Each item in the list is parsed as above.

    .. warning::

        Within this ``random_seed`` list, any items that are either an int or str will be
        evaluated once and converted into a sequence in the normal way.  So if you want an item
        to be some complex Eval operation, you should make it a dict with an explicit
        ``type: Eval``, rather than use the ``$`` shorthand notation.

    * The default behavior, if ``random_seed`` is not given, is to get a seed from the system (/dev/urandom if possible, otherwise based on the time).

* ``nproc`` = *int_value*  (default = 1)  Specify the number of processors to use when drawing images. If nproc <= 0, then this means to try to automatically figure out the number of cpus and use that.
* ``wcs`` See `WCS Field` below.
* ``bandpass`` See `Bandpass Field` below.
* ``sensor``  See `Sensor Field` below.

Image Types
-----------

The default image type is 'Single', which means that the image contains just a single
postage stamp.  Other types are possible (and common) that draw more than one postage stamp
on a full image in different ways.  Each type define extra attributes that are either
allowed or required.
The image types defined by GalSim are:

* 'Single' The image contains a single object at the center (unless it has been shifted of course -- see shift attribute above).

    * ``size`` = *int_value* (optional)  If you want square images for each object (common), you just need to set this one value and the images will be ``size`` x ``size``. The default is for GalSim to automatically determine a good size for the image that will encompass most of the flux of the object.
    * ``xsize`` = *int_value* (default = ``size``)  If you want non-square images, you can specify ``xsize`` and ``ysize`` separately instead. It is an error for only one of them to be non-zero.
    * ``ysize`` = *int_value* (default = ``size``)
    * ``world_pos`` = *pos_value* The position of the object in world coordinates.  For the Single image type, this is unconnected to the object rendering on the image, which is always at the center.  However, it may be provided as something that other calculations need to access.  e.g. shear from a PowerSpectrum or NFWHalo.
    * ``image_pos`` = *pos_value* The nominal position on the image at which to center of the object.  For the Single image type, the object is always placed as close as possible to the center of the image (unless an explicit offset is specified), but the bounds will be adjusted so that position is equal to ``image_pos``.

* 'Tiled' The image consists of a tiled array of postage stamps.

    * ``nx_tiles`` = *int_value* (required)
    * ``ny_tiles`` = *int_value* (required)
    * ``stamp_size`` = *int_value* (either ``stamp_size`` or both ``stamp_xsize`` and ``stamp_ysize`` are required)  The size of square stamps on which to draw the objects.
    * ``stamp_xsize`` = *int_value* (either ``stamp_size`` or both ``stamp_xsize`` and ``stamp_ysize`` are required)  The xsize of the stamps on which to draw the objects.
    * ``stamp_ysize`` = *int_value* (either ``stamp_size`` or both ``stamp_xsize`` and ``stamp_ysize`` are required)  The ysize of the stamps on which to draw the objects.
    * ``border`` = *int_value* (default = 0) number of pixels between tiles. Note: the border value may be negative, in which case the tiles will overlap each other.
    * ``xborder`` = *int_value* (default = ``border``) number of pixels between tiles in the x direction
    * ``yborder`` = *int_value* (default = ``border``) number of pixels between tiles in the y direction
    * ``order`` = *str_value* (default = 'row')  Which order to fill the stamps.  'row' means to proceed row by row starting at the bottom row (each row is filled from left to right).  'column' means to fill the columns from left to right (each column is filled from bottom to top).  'random' means to place the tiles in a random order.

* 'Scattered'  The image consists of a large contiguous area on which postage stamps of each object are placed at arbitrary positions, possibly overlapping each other (in which case the fluxes are added together for the final pixel value).

    * ``size`` = *int_value* (either ``size`` or both ``xsize`` and ``ysize`` are required)
    * ``xsize`` = *int_value* (either ``size`` or both ``xsize`` and ``ysize`` are required)
    * ``ysize`` = *int_value* (either ``size`` or both ``xsize`` and ``ysize`` are required)
    * ``nobjects`` = *int_value* (default if using an input catalog and the output type is 'Fits' is the number of entries in the input catalog; otherwise required)
    * ``stamp_size`` = *int_value* (optional)  The ``stamp_size`` attribute works like the ``size`` attribute for 'Single'.
    * ``stamp_xsize`` = *int_value* (default = ``stamp_size``)
    * ``stamp_ysize`` = *int_value* (default = ``stamp_size``)
    * ``world_pos`` = *pos_value* (only one of ``world_pos`` and ``image_pos`` is allowed) The position in world coordinates relative to the center of the image at which to center of the object.
    * ``image_pos`` = *pos_value* (only one of ``world_pos`` and ``image_pos`` is allowed; default if neither is given is to use type 'XY' with ``x`` = 'Random' from 1 .. ``xsize``, ``y`` = 'Random' from 1 .. ``ysize``)  The position on the image at which to center of the object.

Custom Image Types
------------------

To define your own image type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a class that will be used
to build the image.

The class should be a subclass of `galsim.config.ImageBuilder`, which is the class used for
the default 'Single' type.  There are a number of class methods, and you only need to override
the ones for which you want different behavior than that of the 'Single' type.

.. autoclass:: galsim.config.ImageBuilder
    :members:

The ``base`` parameter is the original full configuration dict that is being used for running the
simulation.  The ``config`` parameter is the local portion of the full dict that defines the image
being built, which would typically be ``base['image']``.

Then, in the Python module, you need to register this function with some type name, which will
be the value of the ``type`` attribute that triggers the use of this Builder object::

    galsim.config.RegisterImageType('CustomImage', CustomImageLoader())

.. autofunction:: galsim.config.RegisterImageType

Note that we register an instance of the class, not the class itself.  This opens up the
possibility of having multiple image types use the same class instantiated with different
initialization parameters.  This is not used by the GalSim image types, but there may be use
cases where it would be useful for custom image types.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_image.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_image

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_image``,
which will read your file and execute the registration command to add the builder to the list
of valid image types.

Then you can use this as a valid image type:

.. code-block:: yaml

    image:
        type: CustomImage
        ...

We don't currently have any examples of custom images, but it may be helpful to look at the GalSim
implementation of the included image types (click on the ``[source]`` links):

.. autoclass:: galsim.config.image_scattered.ScatteredImageBuilder
    :show-inheritance:

.. autoclass:: galsim.config.image_tiled.TiledImageBuilder
    :show-inheritance:


Noise
-----

Typically, you will want to add noise to the image.  The noise attribute should be a dict
with a ``type`` attribute to define what kind of noise should be added.  The noise types
that are defined by GalSim are:

* 'Gaussian' is the simplest kind of noise.  Just Gaussian noise across the whole image with a given sigma (or variance).

    * ``sigma`` = *float_value* (either ``sigma`` or ``variance`` is required)  The rms of the noise in ADU.
    * ``variance`` = *float_value* (either ``sigma`` or ``variance`` is required)  The variance of the noise in ADU^2.

* 'Poisson' adds Poisson noise for the flux value in each pixel, with an optional sky background level.  This is the default noise if you don't specify a different noise type.

    * ``sky_level`` = *float_value* (default = 0.0)  The sky level in ADU/arcsec^2 to use for the noise.  If both this and ``image.sky_level`` are provided, then they will be added together for the purpose of the noise, but the background level in the final image will just be ``image.sky_level``.
    * ``sky_level_pixel`` = *float_value* (default = 0.0)  The sky level in ADU/pixel to use for the noise.  If both this and ``image.sky_level_pixel`` are provided, then they will be added together for the purpose of the noise, but the background level in the final image will just be ``image.sky_level_pixel``.

* 'CCDNoise' includes both Poisson noise for the flux value in each pixel (with an optional gain) and an optional Gaussian read noise.

    * ``sky_level`` = *float_value* (default = 0.0)  The sky level in ADU/arcsec^2 to use for the noise.  If both this and ``image.sky_level`` are provided, then they will be added together for the purpose of the noise, but the background level in the final image will just be ``image.sky_level``.
    * ``sky_level_pixel`` = *float_value* (default = 0.0)  The sky level in ADU/pixel to use for the noise.  If both this and ``image.sky_level_pixel`` are provided, then they will be added together for the purpose of the noise, but the background level in the final image will just be ``image.sky_level_pixel``.
    * ``gain`` = *float_value* (default = 1.0)  The CCD gain in e-/ADU.
    * ``read_noise`` = *float_value* (default = 0.0)  The CCD read noise in e-.

* 'COSMOS' provides spatially correlated noise of the sort found in the F814W HST COSMOS science images described by Leauthaud et al (2007).  The point variance (given by the zero distance correlation function value) may be normalized by the user as required, as well as the dimensions of the correlation function.

    * ``file_name`` = *str_value* (optional) The path and filename of the FITS file containing the correlation function data used to generate the COSMOS noise field.  The default is to use the file packaged with GalSim as 'share/acs_I_unrot_sci_20_cf.fits', but this option lets you override this if desired.
    * ``cosmos_scale`` = *float_value* (default = 0.03) The ACS coadd images in COSMOS have a pixel scale of 0.03 arcsec, and so the pixel scale ``cosmos_scale`` adopted in the representation of of the correlation function takes a default value of 0.03.  If you wish to use other units ensure that ``cosmos_scale`` takes the value corresponding to 0.03 arcsec in your chosen system.
    * ``variance`` = *float_value* (default = 0.) Scale the point variance of the noise field to the desired value, equivalent to scaling the correlation function to have this value at zero separation distance.  Choosing the default scaling of 0. uses the variance in the original COSMOS noise fields.

* 'Correlated' is a more general version of COSMOS, allowing any arbitrary correlated noise spectrum to be used.  It requires the user to provide the appropriate file giving an image of the correlation function.  See `BaseCorrelatedNoise.from_file` for details about how this file must look.

    * ``file_name`` = *str_value* (required) The path and filename of the FITS file containing the correlation function data used to generate the Correlated noise field.
    * ``pixel_scale`` = *float_value* (required) The pixel scale of the original image data from which the correlated noise was constructed.
    * ``variance`` = *float_value* (default = 0.) Scale the point variance of the noise field to the desired value, equivalent to scaling the correlation function to have this value at zero separation distance.  Choosing the default scaling of 0. uses the variance in the file without modification.

The ``noise`` field can also take the following attributes, which are relevant when using
object types that have some intrinsic noise already, such as 'RealGalaxy':

* ``whiten`` = *bool_value* (default = False) Whether or not a noise-whitening procedure should be done on the image after it is drawn to make the noise uncorrelated (white noise).  This is only relevant when using the ``gal`` type 'RealGalaxy'.  Note: After the whitening process, there is white Gaussian noise in the image.  We subtract this much noise from the variance of whatever is given in the ``image.noise`` field.  However, unless this is ``type = 'Gaussian'``, the final noise field will not precisely match what you request.  e.g. 'Poisson' noise would have a portion of the variance be Gaussian rather than Poisson.  This probably does not matter in most cases, but if you are whitening, the most coherent noise profile is 'Gaussian', since that works seamlessly.
* ``symmetrize`` = *int_value* (default = None) The order at which to impose N-fold symmetry on the noise in the image after it is drawn, after which there will be correlated Gaussian noise with the desired symmetry in the image (usually much less than must be added to achieve a fully white noise field).  Similar caveats apply to this option as to the ``white`` option.

In addition to the above, you may also define your own custom noise type in the usual way
with an importable module where you define a custom Builder class and register it with GalSim.
The class should be a subclass of `galsim.config.NoiseBuilder`.  This is really an abstract
base class.  At least the first two of these methods need to be overridden:

.. autoclass:: galsim.config.NoiseBuilder
    :members:

Then, as usual, you need to register this type using::

    galsim.config.RegisterNoiseType('CustomNoise', CustomNoiseBuilder())

.. autofunction:: galsim.config.RegisterNoiseType

and tell the config parser the name of the module to load at the start of processing.

.. code-block:: yaml

    modules:
        - my_custom_noise

Then you can use this as a valid noise type:

.. code-block:: yaml

    image:
        noise:
            type: CustomNoise
            ...

We don't currently have any examples of custom noise types, but it may be helpful to look at the GalSim implementation of the various included noise types (click on the ``[source]`` links:

.. autoclass:: galsim.config.GaussianNoiseBuilder
    :show-inheritance:

.. autoclass:: galsim.config.PoissonNoiseBuilder
    :show-inheritance:

.. autoclass:: galsim.config.CCDNoiseBuilder
    :show-inheritance:

.. autoclass:: galsim.config.COSMOSNoiseBuilder
    :show-inheritance:


WCS Field
---------

The ``pixel_scale`` attribute mentioned above is the usual way to define the connection between
pixel coordinates and sky coordinates.  However, one can define a more complicated relationship,
which is known as a World Coordinate System (WCS) if desired.  To do this, use the ``wcs``
attribute instead of the ``pixel_scale`` attribute.  This should be a dict with a ``type``
attribute that defines what kind of WCS to use.  The wcs types that are defined by GalSim are:

* 'PixelScale' implements a regular square pixel grid.  If you do not specify any ``wcs`` item, this is what will be used, and the scale will be the ``image.pixel_scale`` value.

    * ``scale`` = *float_value* (default = ``image.pixel_scale``)  The scale size of the pixels.  The area is ``scale * scale``.

* 'Shear' implements a uniform shear of a regular square pixel grid.  After the shear, the pixel area will still be ``scale * scale``, but they will be parallelograms (rhombi actually) rather than squares.

    * ``scale`` = *float_value* (required)  The pixel scale of the grid before being sheared.
    * ``shear`` = *shear_value* (required)  The shear to apply.

* 'Jacobian' or 'Affine' implements an arbitrary affine transform.  This is the most general WCS that has a uniform pixel shape.  The world (u,v) coordinates are linearly related to the image (i.e. pixel) (x,y) coordinates.

    * ``dudx`` = *float_value* (required) du/dx
    * ``dudy`` = *float_value* (required) du/dy
    * ``dvdx`` = *float_value* (required) dv/dx
    * ``dvdy`` = *float_value* (required) dv/dy

* 'UVFunction' implements an arbitrary transformation from image coordinates (x,y) to world coordinates (u,v) via two functions u(x,y) and v(x,y).  You can also provide the inverse functions x(u,v) and y(u,v).  They are not required, but if they are not given, then positions of objects cannot be given in world coordinates via ``image.world_pos``.

    * ``ufunc`` = *str_value* (required) A string that can be turned into the function u(x,y) via the python command ``eval('lambda x,y : ' + ufunc)``.
    * ``vfunc`` = *str_value* (required) A string that can be turned into the function v(x,y) via the python command ``eval('lambda x,y : ' + vfunc)``.
    * ``xfunc`` = *str_value* (optional) A string that can be turned into the function x(u,v) of the inverse transformation via the python command ``eval('lambda u,v : ' + xfunc)``.
    * ``yfunc`` = *str_value* (optional) A string that can be turned into the function y(u,v) of the inverse transformation via the python command ``eval('lambda u,v : ' + yfunc)``.

* 'RaDecFunction' implements an arbitrary transformation from image coordinates (x,y) to celestial coordinates (ra,dec) via two functions ra(x,y) and dec(x,y).

    * ``ra_func`` = *str_value* (required) A string that can be turned into the function ra(x,y) via the python command ``eval('lambda x,y : ' + rafunc)``.
    * ``dec_func`` = *str_value* (required) A string that can be turned into the function dec(x,y) via the python command ``eval('lambda x,y : ' + decfunc)``.

* 'Fits' reads a WCS from a FITS file.  Most common WCS types are implemented, but if the file uses something a bit unusual, the success of the read may depend on what other python packages you have installed.  See the documentation of `FitsWCS` for more details.

    * ``file_name`` = *str_value* (required) The name of the FITS file.
    * ``dir`` = *str_value* (default = '.')

* 'Tan' implements a tangent-plane projection of the celestial sphere around a given right ascension and declination.  There is an arbitrary Jacobian matrix relating the image coordinates to the coordinates in the tangent plane.

    * ``dudx`` = *float_value* (required) du/dx
    * ``dudy`` = *float_value* (required) du/dy
    * ``dvdx`` = *float_value* (required) dv/dx
    * ``dvdy`` = *float_value* (required) dv/dy
    * ``ra`` = *angle_value* (required) the right ascension of the tangent point
    * ``dec`` = *angle_value* (required) the declination of the tangent point
    * ``unit`` = *str_value* (default = 'arcsec') the units to use for the intermediate (u,v) coordinates.  Options are 'arcsec', 'arcmin', 'deg', 'rad', 'hr'.

In addition, all wcs types can define an origin in either image coordinates, world coordinates, or
both:

* ``origin`` = *pos_value* (default = (0,0))  Optionally set the image coordinates to use as the origin position, if not (x,y) = (0,0).  Special: You can also specify ``origin`` to be 'center', in which case the origin is taken to be the center of the image rather than the corner.
* ``world_origin`` = *pos_value* (default = (0,0))  Optionally set the world coordinates to use as the origin position, if not (u,v) = (0,0).  (Not available for the celestial WCS types: 'RaDecFunction', 'Fits', and 'Tan'.)

In addition to the above, you may also define your own custom WCS type in the usual way
with an importable module where you define a custom Builder class and register it with GalSim.
The class should be a subclass of `galsim.config.WCSBuilder`.

.. autoclass:: galsim.config.WCSBuilder
    :members:

Then, as usual, you need to register this type using::

    galsim.config.RegisterWCSType('CustomWCS', CustomWCSBuilder())

.. autofunction:: galsim.config.RegisterWCSType

If the builder will use a particular input type, you should let GalSim know this by specifying
the ``input_type`` when registering.  e.g. if it uses an input FitsHeader, you would write::

    galsim.config.RegisterWCSType('CustomWCS', CustomWCSBuilder(), input_type='fits_header')

and tell the config parser the name of the module to load at the start of processing.

.. code-block:: yaml

    modules:
        - my_custom_wcs

Then you can use this as a valid wcs type:

.. code-block:: yaml

    image:
        wcs:
            type: CustomWCS
            ...

For examples of custom wcs types, see :download:`des_wcs.py <../examples/des/des_wcs.py>`, which implements ``DES_SlowLocal`` and ``DES_Local``.
The latter is faster because it uses in input field, 'des_wcs', which saves on I/O time by only loading the files once.  DES_Local is used by :download:`meds.yaml <../examples/des/meds.yaml>`.

It may also be helpful to look at the GalSim implementation of the various included wcs types (click on
the ``[source]`` links):

.. autoclass:: galsim.config.SimpleWCSBuilder
    :show-inheritance:

.. autoclass:: galsim.config.OriginWCSBuilder
    :show-inheritance:


.. autoclass:: galsim.config.TanWCSBuilder
    :show-inheritance:

.. autoclass:: galsim.config.ListWCSBuilder
    :show-inheritance:

Bandpass Field
--------------

If you want to draw chromatic objects, then you also need to define the `Bandpass` to use
for the image in a ``image.bandpass`` field.  Currently, there is only one defined
type to use for the Bandpass, but the code is written in a modular way to allow for
other types, including custom Bandpass types.

* 'FileBandpass' is the default type here, and you may omit the type name when using it.

    * ``file_name`` = *str_value* (required)  The file to read in.
    * ``wave_type`` = *str_value* (required)  The unit of the wavelengths in the file ('nm' or 'Ang' or variations on these -- cf. `Bandpass`)
    * ``blue_limit`` = *float_value* (optional)  Hard cut off on the blue side.
    * ``red_limit`` = *float value* (optional)  Hard cut off on the red side.
    * ``zeropoint`` = *float_value* (optional)  The zero-point to use.
    * ``thin`` = *float_value* (optional)  If given, call `Bandpass.thin` on the Bandpass after reading in from the file, using this for the ``rel_err``.

You may also define your own custom Bandpass type in the usual way
with an importable module where you define a custom Builder class and register it with GalSim.
The class should be a subclass of `galsim.config.BandpassBuilder`.

.. autoclass:: galsim.config.BandpassBuilder
    :members:

Then, as usual, you need to register this type using::

    galsim.config.RegisterBandpassType('CustomBandpass', CustomBandpassBuilder())

.. autofunction:: galsim.config.RegisterBandpassType

and tell the config parser the name of the module to load at the start of processing.

.. code-block:: yaml

    modules:
        - my_custom_bandpass

Then you can use this as a valid bandpass type:

.. code-block:: yaml

    image:
        bandpass:
            type: CustomBandpass
            ...

Sensor Field
------------

When drawing with ``method='phot'``, one can use a `Sensor` to model the conversion of photons
to electrons and then the accumulation of these electrons in the pixels.

The sensor types defined by GalSim are:

* 'Simple' is the default, which does nothing but accumulate the photons at their (x,y)
  positions onto the corresponding pixels in the image.  Typically, in this case, you would
  omit the ``sensor`` field entirely.

* 'Silicon' models a silicon-based CCD sensor that converts photons to electrons at a wavelength-
  dependent depth (probabilistically) and drifts them down to the wells, properly taking
  into account the repulsion of previously accumulated electrons (known as the brighter-fatter
  effect).  See `SiliconSensor` for details.

    * ``name`` = *str_value* (default = 'lsst_itl_50_8')  The naem of the sensor model to use.
      See `SiliconSensor` for the other allowed values.
    * ``strength`` = *float_value* (default = 1.0)  The strength of the brighter-fatter effect
      relative to the nominal value in the model.
    * ``diffusion_factor`` = *float_value* (default = 1.0)  The magnitude of the diffusion
      relative to the nominal value in the model.
    * ``qdist`` = *int_value* (default = 3)  The maximum number of pixels away to calculate the
      distortions due to charge accumulation.
    * ``nrecalc`` = *int_value* (default = 10000)  The number of electrons to accumulate before
      recalculating the distortion of the pixel shapes.
    * ``treering_func`` = *table_value* (optional) A `LookupTable` giving the tree ring pattern
      as a radial function f(r).
    * ``treering_center`` = *pos_value* (optional) The position of the center of the tree ring
      pattern in image coordinates (which may be off the edge of the image).
    * ``transpose`` = *bool_value* (default = False) Whether to transpose the meaning of (x,y)
      for the purpose of the brighter-fatter effect.  This changes which direction the BFE
      causes more elongation.

You may also define your own custom `Sensor` type in the usual way
with an importable module where you define a custom Builder class and register it with GalSim.
The class should be a subclass of `galsim.config.SensorBuilder`.

.. autoclass:: galsim.config.SensorBuilder
    :members:

Then, as usual, you need to register this type using::

    galsim.config.RegisterSensorType('CustomSensor', CustomSensorBuilder())

.. autofunction:: galsim.config.RegisterSensorType

and tell the config parser the name of the module to load at the start of processing.

.. code-block:: yaml

    modules:
        - my_custom_sensor

Then you can use this as a valid sensor type:

.. code-block:: yaml

    image:
        sensor:
            type: CustomSensor
            ...
