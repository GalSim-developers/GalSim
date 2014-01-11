Changes from v1.0 to v1.1:
--------------------------

Python layer API changes:

* Changed the name of the `dx` parameter in the `draw`, `drawShoot`, `drawK`
  methods of `GSObject` and the constructors of `InterpolatedImage` and
  `CorrelatedNoise` to the name `scale`. (Issue #364)
* Changed the `xw` and `yw` parameters of the `Pixel` constructor to a
  single `scale` parameter. (Issue #364)
  * `pix = Pixel(xw=scale)` should now be either `pix = Pixel(scale=scale)`
    or simply `pix = Pixel(scale)`.
* Added new `Box` class to take up the functionality that had been `Pixel` 
  with unequal values of `xw` and `yw`. (Issue #364)
  * `box = Pixel(xw=width, yw=height)` should now be either
    `box = Box(width=width, height=height)` or `box = Box(width, height)`.
* Changed the `dx_cosmos` parameter of `getCOSMOSNoise` to `cosmos_scale`.
  (Issue #364)
* Combined the old `Image`, `ImageView` and `ConstImageView` arrays of class 
  names into a single python layer `Image` class that automatically constructs
  the appropriate C++ image class as an attribute. (Issue #364)
  * `im = Image[type](...)` should now be `Image(..., dtype=type)`
  * `im = ImageView[type](numpy_array.astype(type))` should now be 
     `im = Image(numpy_array.astype(type)`.  i.e. The data type inherits
     from the numpy_array argument when appropriate.  If it is already
     the correct type, you do not need the `astype(type)` part.
  * `im = ConstImageView[type](numpy_array.astype(type))` should now be 
    `im = Image(numpy_array.astype(type), make_const=True)`
  * `im = ImageF(...)` and similar is still valid.
  * `im = ImageViewF(...)` and similar should now be `im = ImageF(...)`
    (preserving the same type letter S, I, F or D).
  * `im = ConstImageViewF(...)` and similar should now be 
    `im = ImageF(..., make_const=True)` (again preserving the type letter).
  * `im = ImageF(...)` _may_ now be written as `im = Image(...)`.  That is,
    the numpy.float32 type is the default data type if you do not specify
    something else either through the type letter or the `dtype` parameter.
* Changed the handling of the `scale` and `init_value` parameters of the 
  `Image` constructor, so that now they have to be named keyword arguments
  rather than a positional arguments. (Issue #364)
  * `im = ImageF(nx, ny, scale, init_val)` should now be 
    `im = ImageF(nx, ny, scale=scale, init_value=init_val)`.
* Removed the `im.at(x,y)` syntax.  This had been equivalent to `im(x,y)`, 
  so any such code should now be switched to that. (Issue #364)
* Removed the previously deprecated Ellipse and AtmosphericPSF classes.
  Also removed PhotonArray from the python layer, since it is only used
  by the C++ layer.  (Issue #364)

Updates to config options:

* Changed the name of sky_pos to world_pos. (Issue #364)
* Added a new image.retry_failures item that can be set so that if the 
  construction of a GSObject fails for any reason, you can ask it to retry.
  An example of this functionality has been added to demo8. (Issue #482)
* Added a new output.retry_io item that can be set so that if the output write 
  command fails (due to hard drive overloading for example), then it will wait 
  a second and try again. (Issue #482)
* Changed the sequence indexing within an image to always start at 0, rather 
  than use obj_num (which continues increasing through all objects in the run).
  Functionally, this would usually only matter if the number of objects per
  file or image is not a constant.  If the number of objects is constant, the 
  automatic looping of the sequencing index essentially did this for you.
  (Issue #487)
* Added Sum type for value types for which it makes sense: float, int, angle,
  shear, position. (Issue #457)
* Allowed the user to modify or add config parameters from the command line. 
  (Issue #479)

Other new features:

* New WCS classes.  See the new wcs.py file for details. (Issue #364)
* Permit users to initialize OpticalPSF with a list or array of aberrations,
  as an alternative to specifying each one individually.  (The innards of 
  OpticalPSF were also rearranged to use arrays instead of individual values, 
  but this is not important for users, just developers.) (Issue #409)
