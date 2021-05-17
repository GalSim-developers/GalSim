Config Values
=============

There are seven kinds of values that are used within the configuration apparatus, which we
have been designating using the following italic names:

- *float_value* corresponds to a Python ``float`` value.  See `float_value`
- *int_value* corresponds to a Python ``int`` value.  See `int_value`
- *bool_value* corresponds to a Python ``bool`` value.  See `bool_value`
- *str_value* corresponds to a Python ``str`` value.  See `str_value`
- *angle_value* corresponds to a GalSim `Angle` instance.  See `angle_value`
- *shear_value* corresponds to a GalSim `Shear` instance.  See `shear_value`
- *pos_value* corresponds to a GalSim `PositionD` instance.  See `pos_value`
- *sky_value* corresponds to a GalSim `CelestialCoord` instance.  See `sky_value`
- *table_value* corresponds to a GalSim `LookupTable` instance.  See `table_value`

Each of the Python types can be given as a constant value using the normal Python conventions
for how to specify such a value.  The GalSim *angle_value* and *pos_value* also have
direct specification options.  In addition, all of them may be a dict with a ``type`` attribute
defining how to generate the value in question.  These are defined below for each kind of
value.

One special ``type`` that any value may use is 'Eval', which uses the Python ``eval`` function
to evaluate a string.  The `Eval type` is described in its own section.

In addition to all of these, you can also write your own `Custom Value Types`
and register them to be used by the config parser.

float_value
-----------

Options are:

* A normal float value (e.g. 1.8)
* Anything that python can convert into a float (e.g. '1.8')
* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'Catalog'  Read the value from an input catalog. This requires that ``input.catalog`` be specified and uses the following fields:

            * ``col`` = *int_value* for ASCII catalog or *str_value* for FITS catalog (required)
            * ``index`` = *int_value* (default = 'Sequence' from 0 to ``input_cat.nobjects``-1)
            * ``num`` = *int_value* (default = 0)  If ``input.catalog`` is a list, this indicates which number catalog to use.

        * 'Dict'  Read the value from an input dictionary. This requires that ``input.dict`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)  For specifying keys below the first level of the dictionary, the ``key`` string is split using the ``input.dict.key_split`` value (default = '.') into multiple keys.  e.g. ``key : galaxy_constants.redshift`` would be parsed as ``dict['galaxy_constants']['redshift']``.
            * ``num`` = *int_value* (default = 0)  If ``input.dict`` is a list, this indicates which number dictionary to use.

        * 'FitsHeader'  Read the value from an input FITS header. This requires that ``input.fits_header`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)
            * ``num`` = *int_value* (default = 0)  If ``input.fits_header`` is a list, this indicates which number file to use.

        * 'Random'  Generate random values uniformly distributed within a range.

            * ``min`` = *float_value* (required)
            * ``max`` = *float_value* (required)

        * 'RandomGaussian'  Generate random values from a Gaussian deviate.

            * ``sigma`` = *float_value* (required)
            * ``mean`` = *float_value* (default = 0)
            * ``min`` = *float_value* (optional) Clip the distribution at some minimum.
            * ``max`` = *float_value* (optional) Clip the distribution at some maximum.

        * 'RandomPoisson'  Generate random values from a Poisson deviate.

            * ``mean`` = *int_value* (required) The mean value of the Poisson distribution.

        * 'RandomBinomial'  Generate random values from a Binomial deviate.

            * ``N`` = *int_value* (required) The number of "coin flips" for the distribution.
            * ``p`` = *float_value* (default = 0.5) The probability of "heads" for each "coin flip".

        * 'RandomWeibull'  Generate random values from a Weibull deviate.

            * ``a`` = *float_value* (required) (Equivalent to k in the Wikipedia article)
            * ``b`` = *float_value* (required) (Equivalent to lambda in the Wikipedia article)

        * 'RandomGamma'  Generate random values from a Gamma deviate.

            * ``k`` = *float_value* (required) The shape parameter.
            * ``theta`` = *float_value* (required) The scale parameter.

        * 'RandomChi2'  Generate random values from a Chi-square deviate.

            * ``n`` = *float_value* (required) The number of degreed of freedom.

        * 'RandomDistribution'  Generate random values from a given probability distribution.

            * ``function`` = *str_value* (required) A string describing the function of x to use for the probability distribution.  e.g. ``x**2.3``.  Alternatively, it may be a file name from which a tabulated function (with columns of x, p(x)) is read in.
            * ``x_min`` = *float_value* (required unless ``function`` is a file name) The minimum value of x to use for the distribution.  (If ``function`` is a file name, the minimum value read in is taken as ``x_min``.)
            * ``x_max`` = *float_value* (required unless ``function`` is a file name) The maximum value of x to use for the distribution.  (If ``function`` is a file name, the maximum value read in is taken as ``x_max``.)
            * ``npoints`` = *int_value* (default = 256) How many points to use for the cumulative probability distribution (CDF), which is used to map from a uniform deviate to the given distribution.  More points will be more accurate, but slower.
            * ``interpolant`` = *str_value* (default = 'Linear')  What to use for interpolating between tabulated points in the CDF.  Options are 'Nearest', 'Linear', 'Cubic' or 'Quintic'.  (Technically, 'Sinc' and 'LanczosN' are also possible, but they do not make sense here.)

        * 'PowerSpectrumMagnification'  Calculate a magnification from a given power spectrum.  This requires that ``input.power_spectrum`` be specified and uses the following fields:

            * ``max_mu`` = *float_value* (default = 5)  The maximum magnification to allow.  If the power spectrum returns a mu value greater than this or less than 0, then use ``max_mu`` instead.  This is a sign of strong lensing, and other approximations are probably breaking down at this point anyway, so this keeps the object profile from going crazy.
            * ``num`` = *int_value* (default = 0)  If ``input.power_spectrum`` is a list, this indicates which number power spectrum to use.

        * 'NFWHaloMagnification'  Calculate a magnification from an NFW Halo mass.  This requires that ``input.nfw_halo`` be specified and uses the following fields:

            * ``gal.redshift`` = *float_value* (required)  Special: The ``redshift`` item must be in the ``gal`` field, not ``magnification``.
            * ``max_mu`` = *float_value* (default = 5)  The maximum magnification to allow.  If NFWHalo returns a mu value greater than this or less than 0, then use ``max_mu`` instead.  This is a sign of strong lensing, and other approximations are probably breaking down at this point anyway, so this keeps the object profile from going crazy.
            * ``num`` = *int_value* (default = 0)  If ``input.nfw_halo`` is a list, this indicates which number halo to use.

        * 'Sequence'  Generate a sequence of values.

            * ``first`` = *float_value* (default = 0)
            * ``step`` = *float_value* (default = 1) The step size between items.
            * ``repeat`` = *int_value* (default = 1) How many times to repeat the same value before moving on.
            * ``last`` = *float_value* (optional; at most one of ``last`` and ``nitems`` is allowed)

                .. note::

                    If ``last`` is provided, once a value passes ``last``, the sequence will
                    repeat starting with ``first`` again.

            * ``nitems`` = *int_value* (optional; at most one of ``last`` and ``nitems`` is allowed)  The number of items in the sequence before starting over again at ``first``.  The default is to just keep incrementing forever.
            * ``index_key`` = *str_value* (optional; see the option descriptions below for which index is used by default)  Which number to use for indexing in the sequence. Valid options are:

                * 'file_num'  Index according to the running file number being worked on.  This is the default for items in the ``input`` and ``output`` fields.
                * 'image_num'  Index according to the running image number.  This index number does not start back at 0 with each file, but rather keeps incrementing.  This is the default for items in the ``image`` field that apply to the full image (i.e. not including ``random_seed``, ``image_pos``, ``world_pos``, etc.).
                * 'obj_num'  Index according to the running object number. This index number does not start back at 0 with each file or image, but rather keeps incrementing.  This is the default for ``image.random_seed``.
                * 'obj_num_in_file'  Index according to the object number within the current file (i.e. start back at 0 again for each new file).  This is the default for items in ``image`` that apply to the object -- ``image_pos``, ``world_pos``, ``offset``, ``stamp_size`` and related, or ``border`` and related -- and also to items in ``psf`` or ``gal``.  Resetting the count back to zero at the start of each file is generally what you want when the files have different numbers of objects. E.g., when you are reading from input catalogs that contain different numbers of objects, you normally want to start back at 0 for each new catalog.

        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *float_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  This is especially useful if you need to use a value from some other calculation, but the value is a random variate, so you cannot just reproduce it.  You need the actual value returned by the random number generator.

            * ``key`` = *str_value* (required)  The key name of the item to use.  The nested layers in the dictionary should be separated by '.' characters.  e.g. To access the current half-light radius of the galaxy, use 'gal.half_light_radius'.  For list items, use the number in the list as a key (using the normal python 0-based counting convention).  e.g. for the half-light radius of the third item in a galaxy ``List`` type, use 'gal.items.2.half_light_radius'.

        * 'Sum'  The sum of two other *float_value* items.

            * ``items`` = *list* (required)  A list of *float_value* items to be added together.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

int_value
---------

Options are:

* A normal int value (e.g. 8)
* Anything that python can convert into an int (e.g. 8.0, '8')

    .. note::

        float values will silently drop any fractional part, so 8.7 will become 8.

* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'Catalog'  Read the value from an input catalog. This requires that ``input.catalog`` be specified and uses the following fields:

            * ``col`` = *int_value* for ASCII catalog or *str_value* for FITS catalog (required)
            * ``index`` = *int_value* (default = 'Sequence' from 0 to ``input_cat.nobjects``-1)
            * ``num`` = *int_value* (default = 0)  If ``input.catalog`` is a list, this indicates which number catalog to use.

        * 'Dict'  Read the value from an input dictionary. This requires that ``input.dict`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)  For specifying keys below the first level of the dictionary, the ``key`` string is split using the ``input.dict.key_split`` value (default = '.') into multiple keys.  e.g. ``key : galaxy_constants.redshift`` would be parsed as ``dict['galaxy_constants']['redshift']``.
            * ``num`` = *int_value* (default = 0)  If ``input.dict`` is a list, this indicates which number dictionary to use.

        * 'FitsHeader'  Read the value from an input FITS header. This requires that ``input.fits_header`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)
            * ``num`` = *int_value* (default = 0)  If ``input.fits_header`` is a list, this indicates which number file to use.

        * 'Random'  Generate a random value uniformly distributed within a range.

            * ``min`` = *int_value* (required)
            * ``max`` = *int_value* (required)  Note: the range includes both ``min`` and ``max``.

        * 'RandomPoisson'  Generate random values from a Poisson deviate.

            * ``mean`` = *int_value* (required) The mean value of the Poisson distribution.

        * 'RandomBinomial'  Generate random values from a Binomial deviate.

            * ``N`` = *int_value* (required) The number of "coin flips" for the distribution.
            * ``p`` = *float_value* (default = 0.5) The probability of "heads" for each "coin flip".

        * 'Sequence'  Generate a sequence of values.

            * ``first`` = *int_value* (default = 0)
            * ``step`` = *int_value* (default = 1) The step size between items.
            * ``repeat`` = *int_value* (default = 1) How many times to repeat the same value before moving on.
            * ``last`` = *float_value* (optional; at most one of ``last`` and ``nitems`` is allowed)

                .. note::

                    if ``last`` is provided, once a value passes ``last``, the sequence will
                    repeat starting with ``first`` again.

            * ``nitems`` = *int_value* (optional; at most one of ``last`` and ``nitems`` is allowed)  The number of items in the sequence before starting over again at ``first``.  The default is to just keep incrementing forever.
            * ``index_key`` = *str_value* (optional) Which number to use for indexing in the sequence. (See the description of this for *float_value* for more details.)

         * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *int_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Sum'  The sum of two other *int_value* items.

            * ``items`` = *list* (required)  A list of *int_value* items to be added together.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

bool_value
----------

Options are:

* A normal bool value (i.e. True or False)
* Anything that python can convert into a bool (e.g. 1, 0.0)
* Some reasonable (case-insensitive) strings: 'true'/'false', 'yes'/'no', '1'/'0'
* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'Catalog'  Read the value from an input catalog. This requires that ``input.catalog`` be specified and uses the following fields:

            * ``col`` = *int_value* for ASCII catalog or *str_value* for FITS catalog (required)
            * ``index`` = *int_value* (default = 'Sequence' from 0 to ``input_cat.nobjects``-1)
            * ``num`` = *int_value* (default = 0)  If ``input.catalog`` is a list, this indicates which number catalog to use.

        * 'Dict'  Read the value from an input dictionary. This requires that ``input.dict`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)  For specifying keys below the first level of the dictionary, the ``key`` string is split using the ``input.dict.key_split`` value (default = '.') into multiple keys.  e.g. ``key : galaxy_constants.redshift`` would be parsed as ``dict['galaxy_constants']['redshift']``.
            * ``num`` = *int_value* (default = 0)  If ``input.dict`` is a list, this indicates which number dictionary to use.

        * 'FitsHeader'  Read the value from an input FITS header. This requires that ``input.fits_header`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)
            * ``num`` = *int_value* (default = 0)  If ``input.fits_header`` is a list, this indicates which number file to use.

        * 'Random'  Generate a random bool value.

            * ``p`` = *float_value* (default = 0.5)  The probability of getting True.  [New in v1.5]

        * 'RandomBinomial'  Generate random values from a Binomial deviate with N=1.

            .. note::

                The default case with ``p`` = 0.5 is equivalent to the 'Random' type.  So this
                would normally be used for random booleans with a different probability of True.

            * ``p`` = *float_value* (default = 0.5) The probability of True.

        * 'Sequence'  Generate a sequence of values.

            * ``first`` = *bool_value* (default = False)  For bool, the only two values in the sequence are False and True, so ``step`` and ``last`` are not needed.
            * ``repeat`` = *int_value* (default = 1) How many times to repeat the same value before moving on.
            * ``index_key`` = *str_value* (optional) Which number to use for indexing in the sequence. (See the description of this for *float_value* for more details.)

        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *bool_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

str_value
---------

Options are:

* A normal str value (e.g. 'out.fits')
* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'Catalog'  Read the value from an input catalog. This requires that ``input.catalog`` be specified and uses the following fields:

            * ``col`` = *int_value* for ASCII catalog or *str_value* for FITS catalog (required)
            * ``index`` = *int_value* (default = 'Sequence' from 0 to ``input_cat.nobjects``-1)
            * ``num`` = *int_value* (default = 0)  If ``input.catalog`` is a list, this indicates which number catalog to use.

        * 'Dict'  Read the value from an input dictionary. This requires that ``input.dict`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)  For specifying keys below the first level of the dictionary, the ``key`` string is split using the ``input.dict.key_split`` value (default = '.') into multiple keys.  e.g. ``key : galaxy_constants.redshift`` would be parsed as ``dict['galaxy_constants']['redshift']``.
            * ``num`` = *int_value* (default = 0)  If ``input.dict`` is a list, this indicates which number dictionary to use.

        * 'FitsHeader'  Read the value from an input FITS header. This requires that ``input.fits_header`` be specified and uses the following fields:

            * ``key`` = *str_value* (required)
            * ``num`` = *int_value* (default = 0)  If ``input.fits_header`` is a list, this indicates which number file to use.

        * 'NumberedFile'  Build a string that includes a number portion: rootNNNNext.  e.g. file0001.fits, file0002.fits, etc.

            * ``root`` = *str_value* (required)  The part of the string that comes before the number.
            * ``num`` = *int_value* (default = 'Sequence' starting with 0)  The number to use in the string.
            * ``digits`` = *int_value* (default = 0)  How many digits to use (minimum) to write the number.  The number will be left-padded with 0s as needed.
            * ``ext`` = *str_value* (default = '.fits' for ``output.file_name`` and the ``file_name`` entries for sub-items within output -- ``psf``, ``weight``, ``badpix`` --, and '' for all other uses)  An extension to place after the number.

        * 'FormattedStr'  Build a string using a format akin to the normal python %-style formatting or C/C++ printf-style formatting.

            * ``format`` = *str_value* (required)  The formatting string to use.  (e.g. 'image_%f_%d.fits')
            * ``items`` = *list* (required)  A list of items to insert into the corresponding % items in the format string.  The letter after the % indicates what kind of value each item is.  So for the above example, the first item in the string should be a *float_value* to put into the %f spot.  The second should be an *int_value* to put into the %d spot.

        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *str_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

angle_value
-----------

These represent `Angle` values.

Options are:

* A string consisting of a float followed by one of the following angle units: radians, degrees, hours, arcminutes, arcseconds. These may be abbreviated as rad, deg, hr, arcmin, arcsec. (e.g. '45 deg')
* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'Radians' or 'Rad'  Use a *float_value* as an angle in radians.

            * ``theta`` = *float_value* (required)

        * 'Degrees' or 'Deg'  Use a *float_value* as an angle in degrees.

            * ``theta`` = *float_value* (required)

        * 'Random'  Generate a random angle uniformly distributed from 0 to 2pi radians.
        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *angle_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Sum'  The sum of two other *angle_value* items.

            * ``items`` = *list* (required)  A list of *angle_value* items to be added together.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

shear_value
-----------

These represent `Shear` values.

Options are:

* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'E1E2' Specify as a distortion in cartesian coordinates.

            * ``e1`` = *float_value* (required)
            * ``e2`` = *float_value* (required)

        * 'EBeta' Specify as a distortion in polar coordinates.

            * ``e`` = *float_value* (required)
            * ``beta`` = *angle_value* (required)

        * 'G1G2' Specify as a reduced shear in cartesian coordinates.

            * ``g1`` = *float_value* (required)
            * ``g2`` = *float_value* (required)

        * 'GBeta' Specify as a reduced shear in polar coordinates.

            * ``g`` = *float_value* (required)
            * ``beta`` = *angle_value* (required)

        * 'Eta1Eta2' Specify as a conformal shear in cartesian coordinates.

            * ``eta1`` = *float_value* (required)
            * ``eta2`` = *float_value* (required)

        * 'EtaBeta' Specify as a conformal shear in polar coordinates.

            * ``eta`` = *float_value* (required)
            * ``beta`` = *angle_value* (required)

        * 'QBeta' Specify as an axis ratio and position angle.

            * ``q`` = *float_value* (required)
            * ``beta`` = *angle_value* (required)

        * 'PowerSpectrumShear'  Calculate a shear from a given power spectrum.  This requires that ``input.power_spectrum`` be specified and uses the following field:

            * ``num`` = *int_value* (default = 0)  If ``input.power_spectrum`` is a list, this indicates which number power spectrum to use.

        * 'NFWHaloShear'  Calculate a shear from an NFW Halo mass.  This requires that ``input.nfw_halo`` be specified and uses the following fields:

            * ``gal.redshift`` = *float_value* (required)  Special: The ``redshift`` item must be in the ``gal`` field, not ``shear``.
            * ``num`` = *int_value* (default = 0)  If ``input.nfw_halo`` is a list, this indicates which number halo to use.

        * 'List'  Select items from a list.

            * ``items`` = ``list`` (required)  A list of *shear_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Sum'  The sum of two other *shear_value* items.

            .. note::

                Unlike the other kinds of values, shears addition is not commutative.
                ``g_a + g_b`` is not the same as ``g_b + g_a``.  Thus, the order of the elements
                in the ``items`` list matters.  The shear effects are applied from last to first,
                so the effects should be listed in order from closest to the observer to farthest
                along the light path.  This is a *somewhat* standard convention for what
                ``g_a + g_b`` means when applied to a galaxy.  ``g_b`` would be a shear that is
                close to the galaxy, and then ``g_a`` would be another shear closer to the
                observer (perhaps within the telescope).

            * ``items`` = *list* (required)  A list of *shear_value* items to be added together.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

pos_value
---------

These represent `PositionD` values, usually for a location on the image.

Options are:

* A string consisting of two floats separated by a comma and possibly white space. (e.g. '1.7, 3.0')
* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'XY' Specify x and y separately.

            * ``x`` = *float_value* (required)
            * ``y`` = *float_value* (required)

        * 'RTheta' Specify using polar coordinate.

            * ``r`` = *float_value* (required)
            * ``theta`` = *angle_value* (required)

        * 'RandomCircle'  Generate a random value uniformly distributed within a circle of a given radius.

            .. note::

                This is different from 'RTheta' with each one random, since that would
                preferentially pick locations near the center of the circle.

            * ``radius`` = *float_value* (required)  The size of the circle within which to draw a random value.
            * ``inner_radius`` = *float_value* (default = 0)  If desired, an inner circle may be excluded, making this an annulus rather than a full circle.
            * ``center`` = *pos_value* (default = 0,0)  The center of the circle.

        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *pos_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Sum'  The sum of two other *pos_value* items.

            * ``items`` = *list* (required)  A list of *pos_value* items to be added together.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

sky_value
---------

These represent `CelestialCoord` values for a location in the sky.

Options are:

* A dict with:

    * ``type`` = *str* (required)  There is currently only one valid option:

        * 'RaDec' Specify x and y separately.

            * ``ra`` = *angle_value* (required)
            * ``dec`` = *angle_value* (required)

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.

table_value
-----------

These represent `LookupTable` values to provide some kind of unary function, although in some
cases you may be able to provide a regular function instead, e.g. via an Eval type using a lambda
function.

Options are:

* A dict with:

    * ``type`` = *str* (required)  Valid options are:

        * 'File'  Read a `LookupTable` from a file.  cf. `LookupTable.from_file`.

            * ``file_name`` = *str_value* (required)
            * ``interpolant`` = *str_value* (default = 'spline') Which interpolant to use.
            * ``x_log`` = *bool_value* (default = False) Whether to use log(x) for the abscissae.
            * ``f_log`` = *bool_value* (default = False) Whether to use log(f) for the ordinates.
            * ``amplitude`` = *float_value* (default = 1.0) An optional scaling to apply to the
              ordinates.

        * 'List'  Select items from a list.

            * ``items`` = *list* (required)  A list of *table_value* items.
            * ``index`` = *int_value* (default = 'Sequence' from 0 to len(items)-1)

        * 'Current'  Use the current value of some other item in the config file.  (See the description of this for *float_value* for more details.)

            * ``key`` = *str_value* (required)  The key name of the item to use.

        * 'Eval'  Evaluate a string.  See `Eval type`.

* A string that starts with '$' or '@'.  See `Shorthand Notation`.


Eval type
---------

Every kind of value has 'Eval' as one of its allowed types.  This works a little bit differently
than the other types, so we describe it here in its own section.

The only required attribute to go along with an 'Eval' is ``str``, which is the string to be
evaluated using the python ``eval`` function.

For example ``str : '800 * 1.e-9 / 4 * 206265'`` will evaluate to 0.041253.  (This example is taken from demo3.yaml.)  This might either be easier than doing a calculation
yourself or perhaps be clearer as to how the number was formed.  For example, this example
calculates ``lam_over_diam`` using lambda = 800 nm, D = 4 m, converting the result into arcsec.
If you later wanted to change to a 6.5m telescope, it would be very clear what to change,
as opposed to if the value were listed as 0.041253.

Preset variables
^^^^^^^^^^^^^^^^

The 'Eval' type gets even more powerful when you use variables.  The file demo10.yaml has some
examples that use the ``pos`` variable, the position of the galaxy relative to the center of the
image, which GalSim will make available for you for any
'Tiled' or 'Scattered' image.  The PSF ``fwhm`` is given as
``'0.9 + 0.5 * (world_pos.x**2 + world_pos.y**2) / 100**2'``, which calculates the PSF size as a function of
position on the image.

Variables that GalSim will provide for you to use:

* ``world_pos`` = the position of the object in world coordinates relative to the center of the image.

    * Available if image ``type`` is 'Tiled' or 'Scattered'
    * Available if ``image_pos`` or ``world_pos`` is explicitly given in the ``stamp`` field.
    * A `galsim.PositionD` instance

* ``image_pos`` = the position of the object on the image in pixels.

    * Available if image ``type`` is 'Tiled' or 'Scattered'
    * Available if ``image_pos`` or ``world_pos`` is explicitly given in the ``stamp`` field.
    * A `galsim.PositionD` instance

* ``sky_pos`` = the position of the object in sky coordinates (RA, Dec)

    * Available if ``sky_pos`` is explicitly given in the ``stamp`` field.
    * Available if ``world_pos`` is defined (as per above) and the WCS is a CelestialWCS.
    * A `galsim.CelestialCoord` instance

* ``image_center`` = the center of the image in pixels.  This is the position on the image that corresponds to ``world_pos = (0,0)``.

    * A `galsim.PositionD` instance

* ``image_origin`` = the origin of the image in pixels.  This is the position on the image that corresponds to the lower-leftmost pixel

    * A `galsim.PositionI` instance

* ``image_xsize``, ``image_ysize`` = the size of the image in pixels.
* ``image_bounds`` = the bounds of the current image.

    * A `galsim.BoundsI` instance

* ``stamp_xsize``, ``stamp_ysize`` = the size of the postage stamp in pixels if available.

    * Not always available, since the postage stamp is allowed to be automatically sized based on the size of final object profile.

* ``pixel_scale`` = the pixel scale of the current image or postage stamp

    * Only available if the WCS is a simple pixel scale.

* ``wcs`` = the WCS of the current image or postage stamp

    * A `galsim.BaseWCS` instance

* ``bandpass`` = the bandpass of the current image if defined.

    * A `galsim.Bandpass` instance

* ``file_num`` = the number of the file currently being worked on.
* ``image_num`` = the number of the image currently being worked on.

    * If parsing a value in ``input`` or ``output``, this may be set to 0 or the last image number from the previous file (if any).

* ``obj_num`` = the number of the object currently being worked on.

    * If parsing a value in ``input``, ``output``, or ``image``, this may be set to 0 or the last object number from the previous image (if any).

* ``start_obj_num`` = the number of the first object in the current file.
* ``rng`` = the random number generator being used for this object.

    * A `galsim.BaseDeviate` instance
    * You can convert it to whatever deviate you need.  e.g. ``galsim.GaussianDeviate(rng,1.0,0.2)()``

Python modules that GalSim will import for you to use:

* ``math``
* ``numpy`` or ``np``
* ``os``
* ``galsim`` (obviously)
* Anything in the ``modules`` field of your configuration file.


User-defined variables
^^^^^^^^^^^^^^^^^^^^^^

It is also possible to define your own variables to use in your expression simply by
defining more attributes in addition to ``str``.  The first letter of the attribute
declares what type it should be evaluated to.  Then the rest of the attribute name is
the name of your variable.

For example, we do not have a specific type for drawing from a Log-Normal distribution.
If you want the flux, say, to be log-normally distributed, you can write something like
the following:

.. code-block:: yaml

    flux :
        type : Eval
        str : '1.e5 * math.exp(normal)'
        fnormal : { type : RandomGaussian , sigma : 0.2 }

The ``f`` at the start of ``fnormal`` indicates that the variable ``normal`` should be
evaluated as a *float_value*.  In this case using ``type`` = 'RandomGaussian'.

Another example appears in demo10.yaml.  There, we define the magnitude of the ellipticity as:

.. code-block:: yaml

    e:
        type : Eval
        fr : { type : Eval , str : '(world_pos.x**2 + world_pos.y**2)**0.5' }
        str : '0.4 * (r/100)**1.5'

So this declares a float variable ``r`` that evaluates as the radial distance from the center.  Then the ellipticity is defined in terms of ``r`` directly rather than via ``world_pos``.

Initial letters of user-defined variables for 'Eval':

* 'f' = ``float``
* 'i' = ``int``
* 'b' = ``bool``
* 's' = ``str``
* 'a' = `galsim.Angle`
* 'p' = `galsim.PositionD`
* 'g' = `galsim.Shear`
* 't' = `galsim.LookupTable`
* 'd' = ``dict``  (This takes a dict as a literal, rather than evaluating it.)
* 'l' = ``list``  (Similarly, this allows for a literal list in the config file.)

The eval-variables field
^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is useful to have the same variable used by multiple Eval calculations.  For such cases, Eval will look for a top-level field called ``eval_variables``.  If this field is present, then anything defined there will be accessible in all Eval calculations in addition to whatever variables are defined for each specific Eval item.

This is similar to the functionality that YAML provides where a value can be named by putting a variable name with an ``&`` before it before any value.  Then later, you can refer to the value by that name preceded by a ``*``, rather than write the value again.  This can lead to more maintainable config files.  E.g. demo10.yaml uses this functionality for ``num_in_ring``, since the value is needed in two different calculations.

It can be convenient to combine the YAML naming scheme with our ``eval_variables`` setup in the following way:

.. code-block:: yaml

    eval_variables :
        fpixel_scale : &pixel_scale 0.3
        istamp_size : &stamp_size 100
        infiles : &nfiles 30
        [ ... ]

This can be put near the top of the YAML file to put the important values all in one place with appropriate names.  Then in the rest of the file, the variables can be used with the YAML ``*`` notation:

.. code-block:: yaml

    image:
        pixel_scale : *pixel_scale

or as part of an ``Eval`` item:

.. code-block:: yaml

    shift :
        type : RTheta
        r : { type : Eval , str : 'pixel_scale * 0.5' }
        theta : { type : Random }

Shorthand notation
^^^^^^^^^^^^^^^^^^

It can be a bit cumbersome at times to write out a full dict with ``type : Eval`` and the
``str`` item you want to evaluate.  To streamline this, we also allow for a shorthand notation
for both Eval and Current types.

- Any string that starts with '$' is taken to mean an Eval type with the rest of the string
  being used as the ``str`` field.
- Any string that starts with '@' is taken to mean a Current type with the rest of the string
  being used as the ``key`` field.

Furthermore, you may use '@' style Current specifications within an Eval string, where the
text after the '@' up to the next white space is used for the key.
So for the above example of a half pixel shift in some random direction, you could write:

.. code-block:: yaml

    shift:
        type : RTheta
        r : '$0.5 * @image.pixel_scale'
        theta : { type: Random }

In many situations, this shorthand notation aids readability.  However, because there is
no dict, you cannot define any variables with this notation.  If you need to define any
variables, you will need to use the regular dict notation.


Custom Value Types
------------------

To define your own value type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a function that will be used
to generate the value you want from the parameters in the config dict.

The generator function should have the following functional form::

    def GenerateCustomValue(config, base, value_type):
        """Generate some kind of custom value given some configuration parameters

        @param config       The configuration dict of the value being generated.
        @param base         The base configuration dict.
        @param value_type   The desired output type.

        @returns value, safe

        The returned value should be something of type value_type, and safe is a bool
        value that indicates whether the value is safe to reuse for future stamps
        (i.e. it is a constant value that will not change for later stamps).
        """
        # If you need a random number generator, this is the one to use.
        rng = base['rng']

        # Generate the desired value.
        # Probably something complicated that you want this function to do.
        value = [...]

        safe = False  # typically, but set to True if this value is safe to reuse.
        return value, safe

The ``base`` parameter is the original full configuration dict that is being used for running the
simulation.  The ``config`` parameter is the local portion of the full dict that defines the value
being generated, e.g. ``config`` might be ``base['gal']['flux']``.

Normally a generator function can only produce a single kind of output type (float for instance),
in which case you can probably ignore the ``value_type`` parameter.  However, if your generator
can be used for multiple kinds of values (int, float and bool maybe), then you might want
to do something different depending on what ``value_type`` is given.

Then, in the Python module, you need to register this function with some type name, which will
be the value of the ``type`` attribute that triggers running this function.  You also need to
give a list of all valid value types that are allowed for this function::

    galsim.config.RegisterValueType('CustomValue', GenerateCustomValue, [float, int])

.. autofunction:: galsim.config.RegisterValueType

If the generator will use a particular input type, you should let GalSim know this by specifying
the ``input_type`` when registering.  E.g. if the generator expects to use an input ``catalog``
to access some ancillary information for each object, you would register this fact using::

    galsim.config.RegisterValueType('CustomValue', GenerateCustomValue, [float, int],
                                    input_type='catalog')

The input object can be accessed in the build function as e.g.::

    input_cat = galsim.config.GetInputObj('catalog', config, base, 'CustomValue')

The last argument is just used to help give sensible error messages if there is some problem,
but it should typically be the name of the value type being built.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_value.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_value

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_value``,
which will read your file and execute the registration command to add your type to the list
of valid value types.

Then you can use this as a valid value type:

.. code-block:: yaml

    gal:
        flux:
            type: CustomValue
            ...

For examples of custom values, see:

* :download:`log_normal.py <../examples/des/log_normal.py>`
* :download:`hsm_shape_measure.py <../examples/des/hsm_shape_measure.py>`
* :download:`excluded_random.py <../examples/des/excluded_random.py>`
* :download:`great3_reject <../examples/great3/great3_reject.py>`
