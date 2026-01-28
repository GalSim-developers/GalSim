
Top Level Fields
================

At the top level, there are 6 basic fields:

* ``psf`` defines what kind of PSF profile to use.
* ``gal`` defines what kind of galaxy profile to use.
* ``stamp`` defines parameters related to building the postage stamp image of each object.
* ``image`` defines parameters related to the full images to be drawn.
* ``input`` defines any necessary input files or things that need some kind of initialization.
* ``output`` defines the names and format of the output files.

None of these are technically required, although it is an error to have _neither_ ``psf`` nor
``gal``. (If you don't want to draw anything but noise, you need to let GalSim know that this is intentional by using ``type: None`` for one of these.)  But the most common usage would be to use
``psf``, ``gal``, ``image`` and ``output``.
It is not uncommon for there to be no input files, so you will often omit the ``input`` field.
And sometimes you will omit the ``gal`` field to draw an image with just stars.
Most simulations will use the default ``stamp`` type (called 'Basic'), which involves drawing
a galaxy convolved by a PSF (or just a PSF image if ``gal`` is omitted) on each postage stamp,
so this field will very often be omitted as well.

We will go through each one in turn. As we do, some values will be called *float_value*,
*int_value*, etc. These can either be a value directly (e.g. *float_value* could just be 1.5),
or they can be a dict that describes how the value should be generated each time (e.g. a random
number or a value read from an input catalog).
See `Config Values` for more information about how to specify these values.

In addition each value will have one of (required) or (optional) or (default = _something_) to
indicate whether the item is required or if there is some sensible default value.  The (optional)
tag usually means that the action in question will not be done at all, rather than done using some
default value. Also, sometimes no item is individually required, but one of several is.

**psf**:

The ``psf`` field defines the profile of the point-spread function (PSF). 
Any object type is allowed for
the ``psf`` type, although some types are obviously more appropriate to use as a PSF than others.
For a list of all the available object types, see `Config Objects`.

If this field is omitted, the PSF will effectively be a delta function.  I.e. the ideal 
galaxy surface brightness profiles will be drawn directly on the image without any convolution.

**gal**:

The ``gal`` field defines the profile of the galaxy.
As for the ``psf`` field, any object type is allowed for
the ``gal`` type, although some types are obviously more appropriate to use as a galaxy than others.
For a list of all the available object types, see `Config Objects`.

Technically, the ``gal`` field is not 
fundamental; its usage is defined by the ``stamp`` type.  One could for instance define a 
``stamp`` type that looked for a ``gal_set`` field instead that might give a list of galaxies
to draw onto a single stamp.  However, all of the ``stamp`` types defined natively in GalSim
use the ``gal`` field, so it will be used by most users of the code.

If this field is omitted, the default ``stamp`` type = 'Basic' will draw the PSF surface brightness
profiles directly according to the ``psf`` field.
Other ``stamp`` types may require this field or may require some other field instead.

**stamp**:

The ``stamp`` field defines the relevant properties and parameters of the stamp-building process.
For a list of all the available ``stamp`` types, see `Config Stamp Field`.

This field is often omitted, in which case the 'Basic' ``stamp`` type will be assumed.

**image**:

The ``image`` field defines the relevant properties and parameters of the full image-building
process.
For a list of all the available ``image`` types, see `Config Image Field`.

If this field is omitted, the 'Single' ``image`` type will be assumed.

**input**:

The ``input`` field indicates where to find any files that you want to use in building the images
or how to set up any objects that require initialization.
For a list of all the available ``input`` types, see `Config Input Field`.

This field is only required if you use `object types <Config Objects>`
or `value types <Config Values>` that use an input object.
Such types will indicate this requirement in their descriptions.

**output**:

The ``output`` field indicates where to write the output files and what kind of output format they
should have.
For a list of all the available ``output`` types, see `Config Output Field`.


