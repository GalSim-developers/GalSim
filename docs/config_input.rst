Config Input Field
==================

The ``input`` field indicates where to find any files that you want to use in building the images
or how to set up any objects that require initialization.

Input Types
-----------

The ``input`` fields defined by GalSim are:

* ``catalog`` defines an input catalog that has values for each object.  Connected with 'Catalog' value type described in `Config Values`.

    * ``file_name`` = *str_value* (required)  The name of the file with the input catalog.
    * ``dir`` = *str_value* (default = '.')  The directory the file is in.
    * ``file_type`` = *str_value* (default = automatically determined from extension of ``file_name``) Valid options are:

        * 'ascii' Read from an ASCII file.

            * ``comments`` = *str_value* (default = '#') The character used to indicate the start of a comment in an ASCII catalog.

        * 'fits' Read from a FITS binary table.

            * ``hdu`` = *int_value* (default = 1) Which hdu to use within the FITS file.  Note: 0 means the primary HDU, the first extension is 1.

* ``dict`` defines an input dictionary, such as a YAML or JSON file.  Connected with 'Dict' value type described in `Config Values`.

    * ``file_name`` = *str_value* (required)  The name of the dictionary file.
    * ``dir`` = *str_value* (default = '.')  The directory the file is in.
    * ``file_type`` = *str_value* (default = automatically determined from extension of ``file_name``) Valid options are:

        * 'yaml' Read from a YAML file.
        * 'json' Read from a JSON file.
        * 'pickle' Read from a python pickle file.

    * ``key_split`` = *str_value* (default = '.')  For specifying keys below the first level of the dictionary, use this string to split the key value into multiple strings.  e.g. If ``key_split`` = '.' (the default) then ``key : galaxy_constants.redshift`` would be parsed as ``dict['galaxy_constants']['redshift']``.  Usually '.' is an intuitive choice, but if some of your key names have a '.' in them, then this would not work correctly, so you should pick something else.

* ``fits_header`` lets you read from the header section of a FITS file.  Connected with 'FitsHeader' value type described in `Config Values`.

    * ``file_name`` = *str_value* (required) The name of the FITS file.
    * ``dir`` = *str_value* (default = '.') The directory the file is in.
    * ``hdu`` = *int_value* (optional) Which HDU to read from the input file.  Default is 0 (the primary HDU), unless the compression implies that the first extension should be used.
    * ``compression`` = *str_value* (optional) The kind of compression if any.  The default is to base the compression on the file extension.  e.g. 'blah.fits.fz' implies Rice compression.  But it can also be specified explicitly.  Supported values are 'none', 'rice', 'gzip', 'bzip2', 'gzip_tile', 'hcompress', 'plio'.
    * ``text_file`` = *bool_value* (default = False) Whether the input file is actually a text file, rather than a binary FITS file.  Normally the file is taken to be a FITS file, but if this is True, then it will read it as a text file containing the header information, such as the .head file output from SCamp.

* ``real_catalog`` defines a catalog of real galaxy images.  Connected with 'RealGalaxy' profile described in `Config Objects`.

    * ``file_name`` = *str_value* (optional)  The name of the file with the input catalog.  If omitted, it will try to use the standard catalog in the $PREFIX/share/galsim directory.  You can download the COSMOS catalog to that directory with the executable ``galsim_download_cosmos``.
    * ``sample`` = *str_value* (optional)  A string that can be used to specify the sample to use, either "23.5" or "25.2".  At most one of ``file_name`` and ``sample`` should be specified.
    * ``dir`` = *str_value* (optional)  The directory the file is in (along with related image and noise files).
    * ``preload`` = *bool_value* (default = False)  Whether to preload all the header information from the catalog fits file into memory at the start.  If ``preload=True``, the bulk of the I/O time happens at the start of the processing.  If ``preload=False``, there is approximately the same total I/O time (assuming you eventually use most of the image files referenced in the catalog), but it is spread over the various RealGalaxy objects that get built.

* ``cosmos_catalog`` defines an input catalog that has values for each object.  Connected with 'COSMOSCatalog' profile described in `Config Objects`.

    * ``file_name`` = *str_value* (optional)  The name of the file with the input catalog.  If omitted, it will try to use the standard catalog in the $PREFIX/share/galsim directory.  You can download the COSMOS catalog to that directory with the executable ``galsim_download_cosmos``.
    * ``sample`` = *str_value* (optional)  A string that can be used to specify the sample to use, either "23.5" or "25.2".  At most one of ``file_name`` and ``sample`` should be specified.
    * ``dir`` = *str_value* (optional)  The directory the file is in (along with related image and noise files).
    * ``preload`` = *bool_value* (default = False)  Whether to preload all the header information from the catalog fits file into memory at the start.  If ``preload=True``, the bulk of the I/O time happens at the start of the processing.  If ``preload=False``, there is approximately the same total I/O time (assuming you eventually use most of the image files referenced in the catalog), but it is spread over the various RealGalaxy objects that get built.
    * ``use_real`` = *bool_value* (default = True)  Whether load the RealGalaxy catalog.  If this is True, you can request either real or parametric galaxies from the catalog.  But if you are only going to use parametric galaxies, then you may set this to False, and it will not bother to load the real galaxies.
    * ``exclusion_level`` = *str_value* (default='marginal') Level of additional cuts to make on the galaxies based on the quality of postage stamp definition and/or parametric fit quality [beyond the minimal cuts imposed when making the catalog - see Mandelbaum et al. (2012, MNRAS, 420, 1518) for details].  Valid options are:

        * "none": No cuts.
        * "bad_stamp": Apply cuts to eliminate galaxies that have failures in postage stamp definition.  These cuts may also eliminate a small subset of the good postage stamps as well.
        * "bad_fits": Apply cuts to eliminate galaxies that have failures in the parametric fits.  These cuts may also eliminate a small subset of the good parametric fits as well.
        * "marginal": Apply the above cuts, plus ones that eliminate some more marginal cases.

    * ``min_hlr`` *float_value* (optional) Exclude galaxies whose fitted half-light radius is smaller than this value (in arcsec).
    * ``max_hlr`` *float_value* (optional) Exclude galaxies whose fitted half-light radius is larger than this value (in arcsec).
    * ``min_flux`` *float_value* (optional) Exclude galaxies whose fitted flux is smaller than this value (in arcsec).
    * ``max_flux`` *float_value* (optional) Exclude galaxies whose fitted flux is larger than this value (in arcsec).

* ``nfw_halo`` defines an NFW halo.  Connected with 'NFWHaloShear' and 'NFWHaloMagnification' value types described in `Config Values`.

    * ``mass`` = *float_value* (required)  The mass of the halo in units of (Msolar / h).
    * ``conc`` = *float_value* (required)  The concentration parameter, defined as the virial radius / scale radius.
    * ``redshift`` = *float_value* (required)  The redshift of the halo.
    * ``halo_pos`` = *pos_value* (default = 0,0)  The position of the halo in world coordinates relative to the origin of the world coordinate system. (Typically you would want to to set ``wcs.origin`` to 'center' to get the halo in the center of the image.)
    * ``omega_m`` = *float_value* (default = 1 - ``omega_lam``)
    * ``omega_lam`` = *float_value* (default = 1 - ``omega_m`` or 0.7 if neither is specified)

* ``power_spectrum`` defines a lensing power spectrum.  Connected with 'PowerSpectrumShear' and 'PowerSpectrumMagnification' value types described in `Config Values`.

    * ``e_power_function`` = *str_value* (at least one of ``e_power_function`` and ``b_power_function`` is required)  A string describing the function of k to use for the E-mode power function.  e.g. ``'k**2'``.  Alternatively, it may be a file name from which a tabulated power spectrum is read in.
    * ``b_power_function`` = *str_value* (at least one of ``e_power_function`` and ``b_power_function`` is required)  A string describing the function of k to use for the B-mode power function.  e.g. ``'k**2'``. Alternatively, it may be a file name from which a tabulated power spectrum is read in.
    * ``delta2`` = *bool_value* (default = False)  Whether the function is really Delta^2(k) = k^2 P(k)/2pi rather than P(k).
    * ``units`` = *str_value*  (default = 'arcsec')  The appropriate units for k^-1.  The default is to use our canonical units, arcsec, for all position variables.  However, power spectra are often more appropriately defined in terms of radians, so that can be specified here to let GalSim handle the units conversion.  Other choices are arcmin or degrees.
    * ``grid_spacing`` = *float_value*  (required for 'Scattered' image type, automatic for 'Tiled')  The distance between grid points on which the power spectrum shears are instantiated.
    * ``interpolant`` = *str_value* (default = 'Linear')  What to use for interpolating between pixel centers.  Options are 'Nearest', 'Linear', 'Cubic', 'Quintic', 'Sinc', or 'LanczosN', where the 'N' after 'Lanczos' should be replaced with the integer order to use for the Lanczos filter.
    * ``ngrid`` = *int_value* (optional) The number of grid points to use.  The default is to use image_size * scale / grid_spacing.
    * ``center`` = *pos_value* (optional) The center point of the grid.  The default is to use the image center.
    * ``index`` = *str_value* (optional) If set, the power spectrum will only be computed when this index changed.  E.g. if ``index`` is 'file_num', then it will only update with each new file, not with each image.
    * ``variance`` = *float_value* (optional) If set, rescale the overall variance of the generated shears to this value.

Another feature of the ``input`` field is that it may optionally be a list.  So you can have multiple input catalogs for instance; one for object parameters and one for overall image parameters perhaps.  When using values with the type 'InputCatalog', you would specify which catalog to use with ``num``.  If you only have a single item for one of the inputs, you can omit the ``num`` parameter when you are using it.

Custom Input Types
------------------

To define your own input type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a class that will be used
to load whatever information you want loaded::

    class MyInputData(object):
        """A class that knows how to load some information from a file or maybe build some
        structure in advance that will be used repeatedly in the simulation.
        """
        def __init__(self, ...):
            pass

Next you need to write a Loader class that is a subclass of `galsim.config.InputLoader`,
which the config code will use to build your input object with the right initialization kwargs.

.. autoclass:: galsim.config.InputLoader
    :members:

The main thing you might want to override is the `galsim.config.InputLoader.getKwargs` function
to determine what kwargs you
want to pass to your initialization function or class based on the parameters in the config dict.
The base class, `galsim.config.InputLoader` uses special class attributes, which most GalSim classes
have defined (for precisely this purpose).  If you want to follow that same model, and there is no
special setup to do at the start of each image, then you can just use the ``InputLoader`` itself
rather than defining your own subclass.

In either case, in your Python module, you need to register this function with some name,
which will be the name of the attribute in the ``input`` field that triggers the use of this
Loader object::

    galsim.config.RegisterInputType('CustomInput', CustomInputLoader(MyInputData))

or::

    galsim.config.RegisterInputType('CustomInput', InputLoader(MyInputData))

.. autofunction:: galsim.config.RegisterInputType


In the above functions, the ``base`` parameter is the original full configuration dict that is being
used for running the simulation.  The ``config`` parameter is the local portion of the full dict
that defines the object being built, which would in this case be ``base['input']['CustomInput']``.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_input.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_input

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_input``,
which will read your file and execute the registration command to add the builder to the list
of valid input types.

Then you can use this as a valid value type:

.. code-block:: yaml

    input:
        CustomInput:
            ...

For examples of custom inputs, see :download:`des_wcs.py <../examples/des/des_wcs.py>`,
which is used by :download:`meds.yaml <../examples/des/meds.yaml>`.

Also `The DES Module` uses custom inputs for the `DES_PSFEx` and `DES_Shapelet` object types, which are used by :download:`draw_psf.yaml <../examples/des/draw_psf.yaml>`.

It may also be helpful to look at the GalSim implementation of our various included input types
(click on the ``[source]`` links):

.. autoclass:: galsim.config.input_cosmos.COSMOSLoader

.. autofunction:: galsim.config.input_cosmos._BuildCOSMOSGalaxy

.. autoclass:: galsim.config.input_nfw.NFWLoader

.. autofunction:: galsim.config.input_nfw._GenerateFromNFWHaloShear

.. autofunction:: galsim.config.input_nfw._GenerateFromNFWHaloMagnification

.. autoclass:: galsim.config.input_powerspectrum.PowerSpectrumLoader

.. autofunction:: galsim.config.input_powerspectrum._GenerateFromPowerSpectrumShear

.. autofunction:: galsim.config.input_powerspectrum._GenerateFromPowerSpectrumMagnification

.. autofunction:: galsim.config.input_real._BuildRealGalaxy

.. autofunction:: galsim.config.input_real._BuildRealGalaxyOriginal
