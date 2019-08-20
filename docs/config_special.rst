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

* :download:`meds.yaml <../examples/des/meds.yaml>`
* :download:`blendset.yaml <../examples/des/blendset.yaml>`
* :download:`cgc.yaml <../examples/great3/cgc.yaml>`

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

* :download:`demo11.yaml <../examples/demo11.yaml>`
* :download:`draw_psf.yaml <../examples/des/draw_psf.yaml>`
* :download:`cgc.yaml <../examples/great3/cgc.yaml>`

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

This would gererate different values for the size, flux, and shape of each component.  But the way those numbers are drawn would be the same for each.

See:

* :download:`rgc.yaml <../examples/great3/rgc.yaml>`
* :download:`cgc_psf.yaml <../examples/great3/cgc_psf.yaml>`

for examples of this feature.

