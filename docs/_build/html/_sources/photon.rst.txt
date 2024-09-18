Photon Shooting
===============

Photon shooting was used successfully to generate the simulated images for the GREAT08 and GREAT10
weak lensing challenges. The objects were convolutions of elliptical Sersic-profile galaxies with
Moffat-profile PSFs. GalSim extends this technique to enable photon shooting for nearly all of its
possible objects, except for deconvolutions.

When we "shoot" a `GSObject` or `ChromaticObject`,
:math:`N_\gamma` photons are created with fluxes :math:`f_i` and
positions :math:`x_i`.  The total photon flux within any region has an expectation value of the
integrated surface brightness of the object in that region, and the total photon flux in any
two regions are uncorrelated.  The actual realized flux in each region is distributed according
to Poisson statistics of the number of photons that actually fall in the region.

We allow for non-uniform :math:`f_i` values primarily so that we can represent negative values of
surface brightness. This is necessary to realize interpolation with kernels that have negative
regions (as will any interpolant that approximates band-limited behavior), and to correctly render
interpolated images that have negative pixel values, such as might arise from using empirical,
noisy galaxy images.

The basic way to activate photon shooting is to use ``method='phot'`` when calling the
`GSObject.drawImage` or `ChromaticObject.drawImage` method.
This will switch over to photon shooting, and the resulting
image will have photon shot noise included from the finite number of photons being shot.

.. note::

    This method necessarily accounts for integration over the pixel by summing the photons that
    are incident in each.  This means that if your surface brightness profile already
    includes the pixel convolution, then you will get the wrong answer.  Such profiles should
    normally use ``method='no_pixel'``.  This kind of profile is often the result of PSF estimation
    codes, so some care is required if you intend to use photon shooting with PSFs that come from
    measurements of real data.

There are a number of other parameters that are relevant only when photon shooting that let you
customize the behavior to some extent:

    n_photons
                The total number of photons to shoot is normally calculated from the object's
                flux.  This flux is taken to be given in photons/cm^2/s, so for most simple
                profiles, this times ``area * exptime`` (both of which default to 1) will equal
                the number of photons shot.  (See the discussion in Rowe et al, 2015, for why
                this might be modified for `InterpolatedImage` and related profiles.)  However,
                you can manually set a different number of photons with ``n_photons``.

    rng
                Since photon shooting is a stochastic process, it needs a random number generator.
                This should be a `BaseDeviate` instance.  If none is provided, one will be
                created automatically.

    max_extra_noise
                This allows you to gain some speed by shooting fewer photons with :math:`f_i > 1`
                at the expense of increasing the noise in each pixel above the natural Poisson
                value.  This parameter specifies how much extra noise you are willing to tolerate.
                It is only relevant if you are not setting ``n_photons``, so the number of photons
                is being automatically calculated. The ``max_extra_noise`` parameter specifies
                how much extra noise per pixel is allowed because of this approximation.  A
                typical value might be ``max_extra_noise = sky_level / 100`` where ``sky_level``
                is the flux per pixel due to the sky.

    poisson_flux
                Normally the total flux of the shot photons will itself be a Poisson random
                value with `GSObject.flux` as the expectation value.  However, you can disable
                this effect by setting ``poisson_flux=False`` to have it shoot exactly the
                flux of the `GSObject`.

    sensor
                The default behavior is for the photons to simply accumulate in the pixel where
                they land.  However, more sophisticated behavior is possible by providing a
                `Sensor` object, which can implement e.g. the brighter-fatter effect, charge
                diffusion, and other effects present in real sensors.  See `Sensor Models`
                for more information about the current options.

    photon_ops
                Prior to accumulating on the sensor, one might want to apply one or more
                `Photon Operators` to the photons.  These operators can be used to apply
                a variety of effects to the photons: changing their fluxes or positions,
                assigning wavelengths or incidence angles, etc.  The ``photon_ops`` argument
                should be a list of any such operators you want to apply.

    maxN
                For very bright objects, one might want to limit the number of photons that are
                shot before being accumulated.  Normally all the photons are generated first
                and stored in a `PhotonArray`.  Then the `Photon Operators` (if any) are
                applied.  And finally the photons are accumulated onto the image pixels.
                If you set ``maxN``, then this process will be done in batches of at most this
                many photons at a time.

    save_photons
                This provides the ability to return the `PhotonArray` that was accumulated
                in case you want to do anything else with it.


If you prefer even more fine-grained control over photon shooting, you can use the following
methods:

    `GSObject.drawPhot`
                This is the actual driver function that `GSObject.drawImage` calls after
                performing some basic sanity checks and image setup.  If you are trying to
                optimize your code for low flux objects, you might find it useful to do the
                image setup yourself and then call this directly.

    `GSObject.shoot`
                This is the method that actually shoots the photons for a `GSObject`.  It
                does not apply any photon operators or accumulate onto the `Image`.


.. toctree::
    :maxdepth: 2

    photon_array
    sensor
    photon_ops
