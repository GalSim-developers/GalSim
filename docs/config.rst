
The Config Module
#################

The basic configuration method is to use a dictionary which can be parsed in python.
Within that structure, each field can either be a value, another dictionary which is then
further parsed, or occasionally a list of items (which can be either values or dictionaries).
The hierarchy can go as deep as necessary. 

Our example config files are all yaml files, which are read using the executable ``galsim``.
This is a nice format for config files, but it is not required. Anything that can represent a
dictionary will do. For example, the executable ``galsim`` also reads in and processes json-style
config files if you prefer.  

If you would like a kind of tutorial that goes through typical uses of the config files, there
are a series of demo config files in the ``GalSim/examples`` directory.
See `Tutorials` for more information.

For a concrete example of what a config file looks like, here is
:download:`demo1.yaml <../examples/demo1.yaml>`
(the first file in the aforementioned tutorial) stripped of most of the comments to make it easier
to see the essence of the structure:

.. code-block:: yaml

    gal :
        type : Gaussian
        sigma : 2  # arcsec
        flux : 1.e5  # total counts in all pixels

    psf :
        type : Gaussian
        sigma : 1  # arcsec

    image :
        pixel_scale : 0.2  # arcsec / pixel
        noise :
            type : Gaussian
            sigma : 30  # standard deviation of the counts in each pixel

    output :
        dir : output_yaml
        file_name : demo1.fits

This file defines a dictionary, which in python would look like::

    config = {
        'gal' : {
            'type' : 'Gaussian',
            'sigma' : 2.,
            'flux' : 1.e5
        },
        'psf' : {
            'type' : 'Gaussian',
            'sigma' : 1.
        },
        'image' : {
            'pixel_scale' : 0.2,
            'noise' : {
                'type' : 'Gaussian',
                'sigma' : 30.
            }
        },
        'output' : {
            'dir' : 'output_yaml',
            'file_name' : 'demo1.fits'
        }
    }

As you can see, there are several top level fields (``gal``, ``psf``, ``image``, and ``output``)
that define various aspects of the simulation.  There are others as well that we will describe
below, but most simulations will want to include at least these four.

Most fields have a ``type`` item that defines what the other items in the field mean.
(The ``image`` and ``output`` fields here have implicit types ``Single`` and ``Fits``,
which are the default, so may be omitted.)
For instance, a Gaussian surface brightness profile is defined by the parameters
``sigma`` and ``flux``.

Most types have some optional items that take reasonable defaults if you omit them.
E.g. the flux is not relevant for a PSF, so it may be omitted in the ``psf`` field, in which
case the default of ``flux=1`` is used.


.. toctree::
    :maxdepth: 1

    config_top
    config_objects
    config_stamp
    config_image
    config_input
    config_output
    config_values
    config_special
    config_galsim
    config_process
