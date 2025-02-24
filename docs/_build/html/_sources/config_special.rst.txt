Special Fields
==============

There are a couple of other top level fields that act more in a support role, rather than being
part of the main processing.

modules
-------

Almost all aspects of the file building can be customized by the user if the existing GalSim
types do not do precisely what you need.  How to do this is described in the pages about
each of the different top-level fields.  In all cases, you need to tell GalSim what Python
modules to load at the start of processing to get the implementations of your custom types.
That is what this field is for.

The ``modules`` field should contain a list of modules that GalSim should import before
processing the rest of the config file.  These modules can be either in the current directory
where you are running the code or installed in your Python distro.  (Or technically, they
need to be located in a directory in ``sys.path``.)

See:

* :gh-link:`meds.yaml <examples/des/meds.yaml>`
* :gh-link:`blendset.yaml <examples/des/blendset.yaml>`
* :gh-link:`cgc.yaml <examples/great3/cgc.yaml>`

for some examples of this field.

eval_variables
--------------

Sometimes, it can be useful to define some configuration parameters right at the top of the
config file that might be used farther down in the file somewhere to highlight them.
Or sometimes, there are calculations that are needed by several different values in the
config file, which you only want to calculate once.

You can put such values in a top-level ``eval_variables`` field.  They work just like
variables that you define for `Eval <Eval Type>`
items, but they can be placed separately from those evaluations.

For examples of this field, see:

* :gh-link:`demo11.yaml <examples/demo11.yaml>`
* :gh-link:`draw_psf.yaml <examples/des/draw_psf.yaml>`
* :gh-link:`cgc.yaml <examples/great3/cgc.yaml>`

template
--------

This feature directs the config processing to first load in some other file (or specific
field with that file) and then possibly modify some components of that dict.

To load in some other config file named ``config.yaml``, you would write::

    template: config.yaml

If you only want to load a particular field from that file, say the ``image`` field, you could
write::

    template: config.yaml:image

The template field may appear anywhere in the config file.  Wherever it appears, the contents
of the other file will be a starting point for that part of the current config dict,
but you can replace
or add values by specifying new values for some of the fields.  Fields that are not at
the top level are specified using a ``.`` to proceed down the levels of the config hierarchy.
e.g. ``image.noise.sky_level`` refers to ``config['image']['noise']['sky_level']``.

For example, if you have a simulation defined in ``my_sim.yaml``, and you want to make another
simulation that is identical, except you want Sersic galaxies instead of Exponential galaxies say,
you could write a new file that looks something like this:

.. code-block:: yaml

    template : my_sim.yaml
    gal:
        type : Sersic
        n : { type : Random, min : 1, max: 4 }
        half_light_radius :
            template : my_sim.yaml:gal.half_light_radius
        flux : 1000
    output.dir : sersic_sim

This will load in the file ``my_sim.yaml`` first, then replace the whole ``config['gal']`` field
as well as ``config['output']['dir']`` (leaving the rest of ``config['output']`` unchanged).
The new ``config['gal']`` field will use the same ``half_light_radius`` specification from
the other file (which might be some complicated random variate that you did not want to
duplicate here).

If the ``template`` field is not at the top level of the config dict, the adjustments should be
made relative to that level of the dictionary:

.. code-block:: yaml

    psf :
        template: cgc.yaml:psf
        index_key : obj_num
        items.0.ellip.e.max : 0.05
        items.1.nstruts : 1
        items.1.strut_angle : { type : Random }

Note that the modifications do not start with ``psf.``, since the template processing is being done
within the ``psf`` field.

Finally, if you want to use a different field from the current config dict as a template, you can
use the colon notation without the file.
E.g. To have a bulge plus disk that have the same kinds of parameters, except that the overall type is a DeVaucouleurs and Exponential respectively, you could do:

.. code-block:: yaml

    gal:
        type: Sum
        items:
            -
                type: DeVaucouleurs
                half_light_radius: { type: Random, min: 0.2, max: 0.8 }
                flux: { type: Random, min: 100, max: 1000 }
                ellip:
                    type: Eta1Eta2
                    eta1: { type: RandomGaussian, sigma: 0.2 }
                    eta2: { type: RandomGaussian, sigma: 0.2 }
            -
                template: :gal.items.0
                type: Exponential

This would generate different values for the size, flux, and shape of each component.  But the way those numbers are drawn would be the same for each.

It is also possible for modules to register a name for a template file, so users can use that name
rather than the actual file location.  For instance, if a module has a template yaml file that is
installed with the python code, it will typically be in an obscure location in a Python
site-packages directory somewhere.  But the installed module would be able to know this location
and register it by name.  E.g. "my_default_sim".  Then users who want to
start with that canonical config file and maybe modify a few things can write:

.. code-block:: yaml

    modules:
        - my_module

    template: my_default_sim

    image.random_seed: 12345
    image.nobjects: 500
    output.file_name: objs_500.fits

To use this feature, the module (i.e. my_module in the example here) should register the name to the correct file name using the `RegisterTemplate` function::

    module_dir = os.path.dirname(__file__)
    default_sim_file = os.path.join(module_dir, 'my_default_sim.yaml')
    galsim.config.RegisterTemplate('my_default_sim', default_sim_file)

.. autofunction:: galsim.config.RegisterTemplate

See:

* :gh-link:`rgc.yaml <examples/great3/rgc.yaml>`
* :gh-link:`cgc_psf.yaml <examples/great3/cgc_psf.yaml>`

for more examples of this feature.

Special Specifications
======================

A few specifications may be used almost anywhere in the config to adjust how the values in those
fields are processed.  They are automatically propagated to lower levels in the dictionary.
For instance, if you set ``index_key : image_num`` in the ``psf`` field, then all values
generated for any aspect of the psf will be constant for a whole image and only change
when the processing goes on to the next image.

index_key
---------

This specifies the cadence on which to generate a new value for each non-constant value.
There are default cadences for each of the major top-level fields, but if you want to specify
a different cadence for some value or field, then you can override it.

Options are:

    * 'file_num'  Update the values for each new file.  This is the default for items in the ``input`` and ``output`` fields.
    * 'image_num'  Update the values for each new image.  This is the default for items in the ``image`` field that apply to the full image (i.e. not including ``random_seed``, ``image_pos``, ``world_pos``, etc.).
    * 'obj_num'  Update the values for each object. This is the default for the other items in ``image``, and also for items in ``stamp``, ``gal``, and ``psf``.
    * 'obj_num_in_file'  For this purpose, equivalent to 'obj_num'.  (For 'Sequence' value types, there is an important distinction between the two.  See its description in `Config Values` for more details.)

It is also possible for a custom module to add additional valid values here by adding to ``galsim.config.valid_index_keys``, which is a list of strings, which are allowed.


rng_index_key
-------------

Each ``index_key`` has its own random number generator to use for generating values that need an rng object.  Normally you want these to match up, but this lets you specify to use the rng for a different key than is used for the actual sequencing.

For instance, if you set ``rng_index_key = 'image_num'`` for a ``gal`` value, then it will use the rng normally used for image_num items, but it will still generate a new value for each obj_num.

rng_num
-------

Normally you specify a single random number seed, which spawns a sequence of rng objects that
update according to the above index keys.  So an rng for each object is stored in ``obj_num_rng``,
one for image_num values is in ``image_num_rng``, etc.

However, you are allowed to specify this seed sequence manually, and in particular, you can
have it be a list of several different sequences which update at different rates, and may
repeat.  For instance, this may be useful to have some galaxy properties repeat for several
exposures, while other properties of the observations are different for each exposure.

You would specify which random number you want to use from such a list using ``rng_num`` in a
field. See the description of ``random_seed`` in  `Image Field Attributes` for more information.
