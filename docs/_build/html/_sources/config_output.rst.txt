Config Output Field
===================

The ``output`` field indicates where to write the output files and what kind of output format
they should be.

.. note::

    **Multiprocessing**

    The config processing can use python multiprocessing to split the work among multiple
    processes on a single node.  This can be done either at the file level or the image level.
    If you set output.nproc != 1, then it will parallelize the creation of files, building
    and writing each file in a separate process.  If you instead set image.nproc != 1, then
    the files will be built one at a time, but the work for drawing the objects will be
    parallelized across the processes.

    There are tradeoffs between these two kinds of multiprocessing that the user should be
    aware of.  Python multiprocessing uses pickle to pass information between processes.
    In the image-based multiprocessing, each process builds a postage stamp image for each
    object and sends that stamp back to the main process to assemble into the final image.
    If the objects are all very easy to draw, this communication can end up dominating the
    run time as python will pickle the image data to send back to the main process.

    File-based multiprocessing has much less communication between processes, since each image
    is fully built and written all in a single process.  However, this kind of multiprocessing
    often requires more memory, since each process holds a full image to be written to disk
    as it is building it.  Users should consider this tradeoff carefully when deciding which
    kind of multiprocessing (if either) is appropriate for their use case.

    Finally, one last caveat about multiprocessing.  Galsim turns off OpenMP threading when
    in a multiprocessing context, so you don't, for instance, have 64 processes, each spawning
    64 OpenMP threads at once.  This works for OpenMP, but not some other sources of threading
    that may be initiated by numpy functions.  If you get errors related to being unable to
    create threads, you should install (via pip or conda) the ``threadpoolctl`` package.
    If this package is installed, GalSim will use it to turn off threading for all of the
    possible backends used by numpy.


Output Field Attributes
-----------------------

All output types use the following attributes to specify the location and number
of output files, or aspects of how to build and write the output files.

* ``file_name`` = *str_value* (default = '\<config file root name\>.fits')  You would typically want to specify this explicitly, but if you do not, then if the configuration file is called my_test.yaml, the output file would be my_test.fits.
* ``dir`` = *str_value* (default = '.')  In which directory should the output file be put.
* ``nfiles`` = *int_value* (default = 1)  How many files to build. Note: if ``nfiles`` > 1, then ``file_name`` and/or ``dir`` should not be a simple string. Rather it should be some generated string that provides a different save location for each file. See the section below on setting *str_value*.
* ``nproc`` = *int_value*  (default = 1)  Specify the number of processors to use when building files. If nproc <= 0, then this means to try to automatically figure out the number of cpus and use that. If you are doing many files, it is often more efficient to split up the processes at this level rather than when drawing the postage stamps (which is what ``image.nproc`` means).
* ``timeout`` = *float_value*  (default = 3600)  Specify the number of seconds to allow for each job when multiprocessing before the multiprocessing queue times out.  The default is generally appropriate to prevent jobs from hanging forever from some kind of multiprocessing snafu, but if your jobs are expected to take more than an hour per output file, you might need to increase this.
* ``skip`` = *bool_value* (default = False) Specify files to skip.  This would normally be an evaluated boolean rather than simply True or False of course.  e.g. To only do the fifth file, you could use ``skip : { type : Eval, str : 'ffile_num != 4' }``, which may be useful during debugging if you are trying to diagnose a problem in one particular file.
* ``noclobber`` = *bool_value* (default = False) Specify whether to skip building files that already exist.  This may be useful if you are running close to the memory limit on your machine with multiprocessing.  e.g. You could use ``nproc`` > 1 for a first run using multiprocessing, and then run again with ``nproc`` = 1 and ``noclobber`` = True to clean up any files that failed from insufficient memory during the multiprocessing run.
* ``retry_io`` = *int_value* (default = 0) How many times to retry the write command if there is any kind of failure.  Some systems have trouble with multiple concurrent writes to disk, so if you are doing a big parallel job, this can be helpful.  If this is > 0, then after an OSError exception on the write command, the code will wait an increasing number of seconds (starting with 1 for the first failure), and then try again up to this many times.

Output Types
------------

The default output type is 'Fits', which means to write a FITS file with the constructed
image in the first HDU.  But other types are possible, which are specified as usual with a
``type`` field.  Other types may define additional allowed and/or required fields.
The output types defined by GalSim are:

* 'Fits' A simple fits file.  This is the default if ``type`` is not given.
* 'MultiFits' A multi-extension fits file.

    * ``nimages`` = *int_value* (default if using an input catalog and the image type is 'Single' is the number of entries in the input catalog; otherwise required) The number of hdu extensions on which to draw an image.

* 'DataCube' A fits data cube.

    * ``nimages`` = *int_value* (default if using an input catalog and the image type is 'Single' is the number of entries in the input catalog; otherwise required) The number of images in the data cube (i.e. the third dimension of the cube).

Custom Output Types
-------------------

To define your own output type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a class that will be used
to build the output file.

The class should be a subclass of ``galsim.config.OutputBuilder``, which is the class used for
the default 'Fits' type.  There are a number of class methods, and you only need to override
the ones for which you want different behavior than that of the 'Fits' type.

.. autoclass:: galsim.config.OutputBuilder
    :members:

The ``base`` parameter is the original full configuration dict that is being used for running the
simulation.  The ``config`` parameter is the local portion of the full dict that defines the object
being built, which would typically be ``base['output']``.

Then, in the Python module, you need to register this function with some type name, which will
be the value of the ``type`` attribute that triggers the use of this Builder object::

    galsim.config.RegisterOutputType('CustomOutput', CustomOutputBuilder())

.. autofunction:: galsim.config.RegisterOutputType

Note that we register an instance of the class, not the class itself.  This opens up the
possibility of having multiple output types use the same class instantiated with different
initialization parameters.  This is not used by the GalSim output types, but there may be use
cases where it would be useful for custom output types.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_output.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_output

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_output``,
which will read your file and execute the registration command to add the builder to the list
of valid output types.

Then you can use this as a valid output type:

.. code-block:: yaml

    output:
        type: CustomOutput
        ...

For an example of a custom output type, see `MEDSBuilder` in `The DES Module`,
which is used by :gh-link:`meds.yaml <examples/des/meds.yaml>`.

It may also be helpful to look at the GalSim implementation of the included output types
(click on the ``[source]`` links):

.. autoclass:: galsim.config.output_datacube.DataCubeBuilder
    :show-inheritance:

.. autoclass:: galsim.config.output_multifits.MultiFitsBuilder
    :show-inheritance:

Extra Outputs
-------------

In addition to the fields for defining the main output file(s), there may also be fields
specifying optional "extra" outputs.  Either extra files to be written, or sometimes extra HDUs
to be added to the main FITS files.  These extra output fields are dicts that may have a number
of parameters defining how they should be built or where they should be written.

* ``psf`` will output (typically) noiseless images of the PSF used for each galaxy.

    * ``file_name`` = *str_value* (either ``file_name`` or ``hdu`` is required)  Write the psf image to a different file (in the same directory as the main image).
    * ``hdu`` = *int_value* (either ``file_name`` or ``hdu`` is required) Write the psf image to another hdu in the main file. (This option is only possible if ``type`` == 'Fits')  Note: 0 means the primary HDU, the first extension is 1.  The main image is always written in hdu 0.
    * ``dir`` = *str_value* (default = ``output.dir`` if that is provided, else '.')  (Only relevant if ``file_name`` is provided.)
    * ``draw_method`` = *str_value* (default = 'auto') The same options are available as for the ``image.draw_method`` item, but now applying to the rendering of the psf images.
    * ``shift`` = *pos_value* (optional) A shift to apply to the PSF object.  Special: if this is 'galaxy' then apply the same shift as was applied to the galaxy.
    * ``offset`` = *pos_value* (optional) An offset to apply when drawing the PSF object.  Special: if this is 'galaxy' then apply the same offset as was applied when drawing the galaxy.
    * ``signal_to_noise`` = *float_value* (optional) If provided, noise will be added at the same level as the main image, and the flux will be rescaled to result in the provided signal-to-noise.  The default is to use flux=1 and not add any noise.

* ``weight`` will output the weight image (an inverse variance map of the noise properties).

    * ``file_name`` = *str_value* (either ``file_name`` or ``hdu`` is required)  Write the weight image to a different file (in the same directory as the main image).
    * ``hdu`` = *int_value* (either ``file_name`` or ``hdu`` is required)  Write the weight image to another hdu in the main file. (This option is only possible if ``type`` == 'Fits')  Note: 0 means the primary HDU, the first extension is 1.  The main image is always written in hdu 0.
    * ``dir`` = *str_value* (default = ``output.dir`` if that is provided, else '.')  (Only relevant if ``file_name`` is provided.)
    * ``include_obj_var`` = *bool_value* (default = False)  Normally, the object variance is not included as a component for the inverse variance map.  If you would rather include it, set this to True.

* ``badpix`` will output the bad-pixel mask image.  This will be relevant when we eventually add the ability to add defects to the images.  For now the bad-pixel mask will be all 0s.

    * ``file_name`` = *str_value* (either ``file_name`` or ``hdu`` is required)  Write the bad pixel mask image to a different file (in the same directory as the main image).
    * ``hdu`` = *int_value* (either ``file_name`` or ``hdu`` is required)  Write the bad pixel mask image to another hdu in the main file. (This option is only possible if ``type`` == 'Fits')  Note: 0 means the primary HDU, the first extension is 1.  The main image is always written in hdu 0.
    * ``dir`` = *str_value* (default = ``output.dir`` if that is provided, else '.')  (Only relevant if ``file_name`` is provided.)

* ``truth`` will output a truth catalog.  Note: assuming you are using the ``galsim`` executable to process the config file, the config dict is really read in as an OrderedDict, so the columns in the output catalog will be in the same order as in the YAML file.  If you are doing this manually and just use a regular Python dict for config, then the output columns will be in some arbitrary order.

    * ``file_name`` = *str_value* (either ``file_name`` or ``hdu`` is required)  Write the bad pixel mask image to a different file (in the same directory as the main image).
    * ``hdu`` = *int_value* (either ``file_name`` or ``hdu`` is required)  Write the bad pixel mask image to another hdu in the main file. (This option is only possible if ``type`` == 'Fits')  Note: 0 means the primary HDU, the first extension is 1.  The main image is always written in hdu 0.
    * ``dir`` = *str_value* (default = ``output.dir`` if that is provided, else '.')  (Only relevant if ``file_name`` is provided.)
    * ``columns`` = *dict* (required) A dict connecting the names of the output columns to the values that should be output.  The values can be specified in a few different ways:

        * A string indicating what current value in the config dict to use.  e.g. 'gal.shear.g1' would grab the value of config['gal']['shear']['g1'] that was used for the current object.
        * A dict that should be evaluated in the usual way values are evaluated in the config processing. Caveat: Since we do not have a way to indicate what type the return value should be, this functionality is mostly limited to 'Eval' and 'Current' types, which is normally fine, since it would mostly be useful for just doing some extra processing to some current value.
        * An implicit Eval string starting with '$', typically using '@' values to get Current values.  e.g. to output e1-style shapes for a Shear object that was built with (g1,g2), you could write '$(@gal.ellip).e1' and '$(@gal.ellip).e2'.
        * A straight value.  Not usually very useful, but allowed.  e.g. You might want your truth catalogs to have a consistent format, but some simulations may not define a particular value.  You could just output -999 (or anything) for that column in those cases.

Adding your own Extra Output Type
---------------------------------

You can also add your own extra output type in a similar fashion as the other custom types that
you can define.  (cf. e.g. [Custom Output Types](#custom-output-types))  As usual, you would
write a custom module that can be imported, which should contain a class for building and
writing the extra output, register it with GalSim, and add the module to the ``modules`` field.

The class should be a subclass of ``galsim.config.ExtraOutputBuilder``.  You may override any
of the following methods.

.. autoclass:: galsim.config.ExtraOutputBuilder
    :members:

Then, in the Python module, you need to register this function with some type name, which will
be the value of the attribute in the ``output`` field that triggers the use of this Builder object::

    galsim.config.RegisterExtraOutput('CustomExtraOutput', CustomExtraOutputBuilder())

.. autofunction:: galsim.config.RegisterExtraOutput

Note that we register an instance of the class, not the class itself.  This opens up the
possibility of having multiple output types use the same class instantiated with different
initialization parameters.  This is not used by the GalSim output types, but there may be use
cases where it would be useful for custom output types.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_output.py``, then you would use the following top-level ``modules`` field
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_output

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_output``,
which will read your file and execute the registration command to add the builder to the list
of valid output types.

Then you can use this as a valid extra output directive:

.. code-block:: yaml

    output:
        custom_extra_output:
            ...

For examples of custom extra outputs, see

* :gh-link:`blend.yaml <examples/des/blend.yaml>`
* :gh-link:`blendset.yaml <examples/des/blendset.yaml>`

which use custom extra outputs ``deblend`` and ``deblend_meds`` defined in :gh-link:`blend.py <examples/des/blend.py>`.

Also,

* :gh-link:`cgc.yaml <examples/great3/cgc.yaml>`

which uses custom extra output ``noise_free`` defined in :gh-link:`noise_free.py <examples/great3/noise_free.py>`.

It may also be helpful to look at the GalSim implementation of the included extra output types
(click on the ``[source]`` links):

.. autoclass:: galsim.config.extra_psf.ExtraPSFBuilder
    :show-inheritance:

.. autoclass:: galsim.config.extra_truth.TruthBuilder
    :show-inheritance:

.. autoclass:: galsim.config.extra_weight.WeightBuilder
    :show-inheritance:

.. autoclass:: galsim.config.extra_badpix.BadPixBuilder
    :show-inheritance:
