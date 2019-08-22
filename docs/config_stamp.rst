Config Stamp Field
==================

The ``stamp`` field defines some properties about how to draw the postage-stamp images of each
object.  It is often unneccessary to explicitly include this top-level field.  The default
``stamp`` type, called 'Basic', is often what you want.

Stamp Field Attributes
----------------------

Some attributes that are allowed for all stamp types are:

* ``draw_method`` = *str_value* (default = 'auto')  Valid options are:

    * 'auto'  The default is normally equivalent to 'fft'.  However, if the object being rendered is simple (no convolution) and has hard edges (e.g. a Box or a truncated Moffat or Sersic), then it will switch to 'real_space', since that is often both faster and more accurate in these cases (due to ringing in Fourier space).
    * 'fft'  This method will convolve by the pixel (as well as convolving the galaxy and PSF if you have both) using a fast Fourier transform.
    * 'real_space'  This uses real-space integration to integrate over the pixel. This is only possible if there is only _either_ a PSF or galaxy, not both.  Also, it cannot involve a Convolution internally.  If GalSim is unable to do the real-space integration, this will revert to 'fft'.
    * 'phot'  Use photon shooting, which treats the profile as a probability distribution, draws photons from that distribution, and then "shoots" them at the image.  The flux of each photon is added to whichever pixel it hits.  This automatically handles the integration over the pixel, but it cannot be used with Deconvolutions (including RealGalaxy objects) and the result will necessarily have Poisson noise from the finite number of photons being shot.

        * ``max_extra_noise`` = *float_value* (optional)  If the image is sky noise dominated, then it is efficient to stop shooting photons when the photon noise of the galaxy is much less than the sky noise. This parameter specifies how much extra noise, as a fraction of the sky noise, is permissible in any pixel.
        * ``n_photons`` = *int_value* (optional; default is to assume the object flux is given in photons and use that) Specifies the total number of photons to shoot as a hard, fixed number.  If both ``n_photons`` and ``max_extra_noise`` are specified in the options, ``max_extra_noise`` is ignored and a warning is generated.
        * ``poisson_flux`` = *bool_value* (default = True, unless ``n_photons`` is given, in which case the default is False) Whether to allow the total object flux to vary according to Poisson statistics for the number of photons being shot.

    * 'no_pixel'  This will not integrate the flux over the pixel response.  Rather, it just samples the surface brightness distribution at the pixel centers and multiplies by the pixel area.  This is appropriate if the PSF already has the pixel response included (e.g. from an observed image of a PSF).
    * 'sb'  This is similar to 'no_pixel', except that the image values will simply be the sampled surface brightness, not multiplied by the pixel area.  This does not correspond to any real observing scenario, but it could be useful if you want to view the surface brightness profile of an object directly, without including the pixel integration.

* ``offset`` = *pos_value* (optional) An offset in chip coordinates (i.e. pixel units) to apply when drawing the object on the postage stamp.
* ``gsparams`` = *dict* (optional) A dict of (non-default) GSParams items that you want applied to the constructed object.
* ``retry_failures`` = *int_value* (default = 0) How many times to retry the construction of a GSObject if there is any kind of failure.  For example, you might have a random shear value that technically may come back with :math:`|g| > 1`, but it should be very rare.  So you might set it to retry once or twice in that case.  If this is > 0, then after a failure, the code will wait 1 second (in case the failure was related to memory usage on the machine), and then try again up to this many times.
* ``world_pos`` = *pos_value* or *sky_value* (only one of ``world_pos`` and ``image_pos`` is allowed) The position in world coordinates at which to center the object.  This is often defined in the ``image`` field, but it can be overridden in the ``stamp`` field.
* ``image_pos`` = *pos_value* (only one of ``world_pos`` and ``image_pos`` is allowed) The position on the full image at which to center the object.  This is often defined in the ``image`` field, but it can be overridden in the ``stamp`` field.  Note: the object is always centered as nearly as possible on the postage stamp being drawn (unless an explicit ``offset`` is given), but the ``image_pos`` or ``world_pos`` determines where in the larger image this stamp is placed.
* ``skip`` = *bool_value* (default=False)  Skip this stamp.
* ``quick_skip`` = *bool_value* (default=False)  Skip this stamp before doing any work, even making the rng or calculating the position.  (Usually used by some other part of the processing to precalculate objects that are not worth doing for some reason.)
* ``obj_rng`` = *bool_value* (default=True) Whether to make a fresh random number generator for each object.  If set to False, all objects will use the same rng, which will be the one used for image-level calculations.

Stamp Types
-----------

The default stamp type is 'Basic', which constructs a galaxy object based on the ``gal`` field
(if present) and a PSF object from the ``psf`` field (again, if present), convolves them
together, and draws the object onto a postage stamp.  This is often what you need, but
there is also a 'Ring' type, and you can define your own custom ``stamp`` type if you want
to customize any aspect of the stamp-building process.

* 'Basic' The postage stamp contains a single ``gal`` object convolved by a ``psf`` object, assuming both fields are given.  If only one of the two is given, that one is drawn.

    * ``size`` = *int_value* (optional)  If you want square postage stamps for each object (common), you just need to set this one value and the images will be ``size`` x ``size``. The default is for GalSim to automatically determine a good size for the image that will encompass most of the flux of the object, but note that the ``image`` type may define the stamp size (e.g. 'Tiled'), in which case that will be used.
    * ``xsize`` = *int_value* (default = ``size``)  If you want non-square postage stamps, you can specify ``xsize`` and ``ysize`` separately instead. It is an error for only one of them to be non-zero.
    * ``ysize`` = *int_value* (default = ``size``)
    * ``min_flux_frac`` = *float_value* (optional) If the rendered stamp (before noise is applied) has less than this fraction of the nominal flux of the object, reject it and start over (presumably choosing new random values for size, flux, etc.).  This counts as a "failure" for the purpose of the ``retry_failures`` count.
    * ``min_snr`` = *float_value* (optional) If the measured signal-to-noise ratio (using the optimal matched filter definition of S/N, measured using the signal on the stamp before noise is applied) is less than this, then reject it and start over.  This counts as a "failure" for the purpose of the ``retry_failures`` count.
    * ``max_snr`` = *float_value* (optional) If the measured signal-to-noise ratio is higher than this, then reject it and start over.  This counts as a "failure" for the purpose of the ``retry_failures`` count.
    * ``reject`` = *bool_value* (optional) If this evaluates to true, then reject the current stamp and start over.  Typically, this would be a custom function that would perform some measurement on the pre-noise image.
      See :download:`cgc.yaml <../examples/great3/cgc.yaml>` for an examples of such a custom function.  This counts as a "failure" for the purpose of the ``retry_failures`` count.

* 'Ring' Generate galaxies in a ring for a ring test. (Use num=2 to get pairs of 90 degree rotated galaxies.)

    * ``size``, ``xsize``, ``ysize`` = *int_value* (optional) Same meaning as for 'Basic' type.
    * ``num`` = *int_value* (required)  How many objects to include in the ring.
    * ``full_rotation`` = *angle_value* (default = 180 degrees)  What angle should be spanned by the full rotation?  The default of 180 degrees is appropriate for the typical case of a rotationally symmetric galaxy (e.g. a sheared Exponential), but if the ``first`` profile does not have rotational symmetry, then you probably want to set this to 360 degrees.
    * ``index`` = *int_value* (default = 'Sequence' from 0 to num-1)  Which item in the Ring is this.
    * ``min_flux_frac`` = *float_value* (optional) Equivalent to Basic, but only applies to the first stamp in the ring.
    * ``min_snr`` = *float_value* (optional) Equivalent to Basic, but only applies to the first stamp in the ring.
    * ``max_snr`` = *float_value* (optional) Equivalent to Basic, but only applies to the first stamp in the ring.
    * ``reject`` = *bool_value* (optional) Equivalent to Basic, but only applies to the first stamp in the ring.


Custom Stamp Types
------------------

To define your own stamp type, you will need to write an importable Python module
(typically a file in the current directory where you are running ``galsim``, but it could also
be something you have installed in your Python distro) with a class that will be used
to build the stamp.

The class should be a subclass of `galsim.config.StampBuilder`, which is the class used for
the default 'Basic' type.  There are a number of class methods, and you only need to override
the ones for which you want different behavior than that of the 'Basic' type.

.. autoclass:: galsim.config.StampBuilder
    :members:

The ``base`` parameter is the original full configuration dict that is being used for running the
simulation.  The ``config`` parameter is the local portion of the full dict that defines the stamp
being built, which would typically be ``base['stamp']``.

Then, in the Python module, you need to register this function with some type name, which will
be the value of the ``type`` attribute that triggers the use of this Builder object::

    galsim.config.RegisterStampType('CustomStamp', CustomStampBuilder())

.. autofunction: galsim.config.RegisterStampType

Note that we register an instance of the class, not the class itself.  This opens up the
possibility of having multiple stamp types use the same class instantiated with different
initialization parameters.  This is not used by the GalSim stamp types, but there may be use
cases where it would be useful for custom stamp types.

Finally, to use this custom type in your config file, you need to tell the config parser the
name of the module to load at the start of processing.  e.g. if this function is defined in the
file ``my_custom_stamp.py``, then you would use the following top-level ``modules`` field 
in the config file:

.. code-block:: yaml

    modules:
        - my_custom_stamp

This ``modules`` field is a list, so it can contain more than one module to load if you want.
Then before processing anything, the code will execute the command ``import my_custom_stamp``,
which will read your file and execute the registration command to add the buidler to the list
of valid stamp types.

Then you can use this as a valid stamp type:

.. code-block:: yaml

    stamp:
        type: CustomStamp
        ...

For examples of custom stamps, see

* :download:`blend.yaml <../examples/des/blend.yaml>`
* :download:`blendset.yaml <../examples/des/blendset.yaml>`

which use custom stamp types ``Blend`` and ``BlendSet`` defined in :download:`blend.py <../examples/des/blend.py>`. 

It may also be helpful to look at the GalSim implementation of the include ``Ring`` type:
(click on the ``[source]`` link):

.. autoclass:: galsim.config.stamp_ring.RingBuilder
    :show-inheritance:

