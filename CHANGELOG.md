Changes from v1.0 to v1.1:
--------------------------

API changes:

* Changed the `dx` parameter in the `draw`, `drawShoot`, `drawK` methods of 
  `GSObject` and the constructors of `InterpolatedImage` and
  `CorrelatedNoise` to `scale`.  (Issue #364)
* Changed the `xw` and `yw` parameters of the `Pixel` constructor to 
  single `scale` parameter. (Issue #364)
* Changed the `dx_cosmos` parameter of getCOSMOSNoise to `cosmos_scale`.
  (Issue #364)
* Combined the old Image, ImageView and ConstImageView arrays of class names 
  into a single python layer Image class that automatically constructs the 
  appropriate C++ image class as an attribute.   (Issue #364)
  * `im = Image[type](...)` should now be `Image(..., dtype=type)`
  * `im = ImageView[type](numpy_array.astype(type))` should now be 
     `im = Image(numpy_array.astype(type)`.  i.e. The data type inherits
     from the numpy_array argument when appropriate.  If it is already
     the correct type, you do not need the `astype(type)` part.
  * `im = ConstImageView[type](numpy_array.astype(type))` should now be 
    `im = Image(numpy_array.astype(type), make_const=True)`
  * `im = ImageF(...) and similar is still valid.
  * `im = ImageViewF(...) and similar should now be `im = ImageF(...)`
    (preserving the same type letter S, I, F or D).
  * `im = ConstImageViewF(...) and similar should now be 
    `im = ImageF(..., make_const=True)` (again preserving the type letter).
  * `im = ImageF(...) _may_ now be written as `im = Image(...)`.  That is,
    the numpy.float32 type is the default data type if you do not specify
    something else either through the type letter or the `dtype` parameter.
* Changed the handling of the `scale` and `init_value` parameters of the 
  `Image` constructor, so that now they have to be named keyword arguments
  rather than a positional arguments.
  * `im = ImageF(nx, ny, scale, init_val)` should now be 
    `im = ImageF(nx, ny, scale=scale, init_value=init_val)`.
* Removed the `im.at(x,y)` syntax.  This had been equivalent to `im(x,y)`, 
  so any such code should now be switched to that.

New WCS classes:

We added a new heirarchy of WCS classes that control the conversion between
sky coordinates (in which the PSF and galaxy objects are defined) and chip 
coordinates (in which things are drawn on images).  Previously, the pixel
scale was the only conversion, a constant scale factor in arcsec/pixel.
Now there are more options:

* PixelScale encapsulates the old behavior of using just a pixel scale.
  Most of the time you will not need to explicitly construct this kind of 
  WCS object, since the only notation of `ImageF(..., scale=pixel_scale)`
  or `im.scale = pixel_scale` will set this up automatically for you.
* 

Updates to config options:

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

* Added new `Box` class to take up the functionality that had been `Pixel` 
  with unequal values of `xw` and `yw`. (Issue #364)

