
The Roman Space Telescope Module
################################

The galsim.roman module contains information and functionality that can be used to simulate
images for the Roman Space Telescope.  Some of the functionality is specific to Roman per se, but
some of the routines are more generically simulating aspects of the HgCdTe detectors, which will
be used on Roman.  These routines might therefore be useful for simulating observations from
other telescopes that will use these detectors.

The demo script demo13.py illustrates the use of most of this functionality.

.. note::
    To use this module, you must separately ``import galsim.roman``.  These functions are
    not automatically imported when you ``import galsim``.


Module-level Attributes
=======================

There are a number of attributes of the ``galsim.roman`` module, which define some numerical
parameters related to the Roman geometry.  Some of these parameters relate to the entire
wide-field imager.  Others, especially the return values of the functions to get the
PSF and WCS, are specific to each SCA (Sensor Chip Assembly, the equivalent of a chip for an optical
CCD) and therefore are indexed based on the SCA.  All SCA-related arrays are 1-indexed, i.e., the
entry with index 0 is None and the entries from 1 to n_sca are the relevant ones.  This is
consistent with diagrams and so on provided by the Roman project, which are 1-indexed.

The NIR detectors that will be used for Roman have a different photon detection process from CCDs.
In particular, the photon detection process begins with charge generation.  However, instead of
being read out along columns (as for CCDs), they are read directly from each pixel.  Moreover, the
actual quantity that is measured is technically not charge, but rather voltage.  The charge is
inferred based on the capacitance.  To use a common language with that for CCDs, we will often refer
to quantities measured in units of e-/pixel, but for some detector non-idealities, it is important
to keep in mind that it is voltage that is sensed.

gain
    The gain for all SCAs (sensor chip assemblies) is expected to be the roughly the same,
    and we currently have no information about how different they will be, so this is a
    single value rather than a list of values.  Once the actual detectors exist and have been
    characterized, it might be updated to be a dict with entries for each SCA.

pixel_scale
    The pixel scale in units of arcsec/pixel.  This value is approximate and does not
    include effects like distortion, which are included in the WCS.

diameter
    The telescope diameter in meters.

obscuration
    The linear obscuration of the telescope, expressed as a fraction of the diameter.

collecting_area
    The actual collecting area after accounting for obscuration, struts, etc. in
        units of cm^2.

exptime
    The typical exposure time in units of seconds.  The number that is stored is for a
    single dither.  Each location within the survey will be observed with a total of 5-7
    dithers across 2 epochs.

n_dithers
    The number of dithers per filter (typically 5-7, so this is currently 6 as a
    reasonable effective average).

dark_current
    The dark current in units of e-/pix/s.

nonlinearity_beta
    The coefficient of the (counts)^2 term in the detector nonlinearity
    function.  This will not ordinarily be accessed directly by users; instead,
    it will be accessed by the convenience function in this module that defines
    the nonlinearity function as counts_out = counts_in + beta*counts_in^2.
    Alternatively users can use the `galsim.roman.applyNonlinearity` routine,
    which already knows about the expected form of the nonlinearity in the
    detectors.

reciprocity_alpha
    The normalization factor that determines the effect of reciprocity failure
    of the detectors for a given exposure time.  Alternatively, users can use
    the `galsim.roman.addReciprocityFailure` routine, which knows about this
    normalization factor already, and allows users to choose an exposure time or
    use the default Roman exposure time.

read_noise
    A total of 8.5e-.  This comes from 20 e- per correlated double sampling (CDS) and a
    5 e- floor, so the CDS read noise dominates.  The source of CDS read noise is the
    noise introduced when subtracting a single pair of reads; this can be reduced by
    averaging over multiple reads.  Also, this read_noise value might be reduced
    based on improved behavior of newer detectors which have lower CDS noise.

thermal_backgrounds
    The thermal backgrounds (in units of e-/pix/s) are based on a temperature
    of 282 K, but this plan might change in future.  The thermal backgrounds
    depend on the band, so this is not a single number; instead, it's a
    dictionary that is accessed by the name of the optical band, e.g.,
    ``galsim.roman.thermal_backgrounds['F184']`` (where the names of the
    bandpasses can be obtained using the `getBandpasses` routine described
    below).

pupil_plane_file
    There is actually a separate file for each SCA giving the pupil plane mask
    for the Roman telescope as seen from the center of each SCA.  When building
    the PSF with galsim.roman.getPSF, it will use the correct one for the given
    SCA.  However, for backwards compatibility, if anyone needs a generic image
    of the pupil plane, this file is for SCA 2, near the center of the WFC field.

pupil_plane_scale
    The pixel scale in meters per pixel for the image in pupil_plane_file.

stray_light_fraction
    The fraction of the sky background that is allowed to contribute as stray
    light.  Currently this is required to be <10% of the background due to
    zodiacal light, so its value is set to 0.1 (assuming a worst-case).  This
    could be used to get a total background including stray light.

ipc_kernel
    The 3x3 kernel to be used in simulations of interpixel capacitance (IPC), using
    `galsim.roman.applyIPC`.

persistence_coefficients
    The retention fraction of the previous eight exposures in a simple,
    linear model for persistence.

persistence_fermi_params
    The parameters in the fermi persistence model.

n_sca
    The number of SCAs in the focal plane.

n_pix_tot
    Each SCA has n_pix_tot x n_pix_tot pixels.

n_pix
    The number of pixels that are actively used.  The 4 outer rows and columns will be
    attached internally to capacitors rather than to detector pixels, and used to monitor
    bias voltage drifts.  Thus, images seen by users will be n_pix x n_pix.

jitter_rms
    The worst-case RMS jitter per axis for Roman in the current design (reality
            will likely be much better than this).  Units: arcsec.

charge_diffusion
    The per-axis sigma to use for a Gaussian representing charge diffusion for
    Roman.  Units: pixels.

For example, to get the gain value, use galsim.roman.gain.  Most numbers related to the nature of
the detectors are subject to change as further lab tests are done.

Roman Functions
===============

This module also contains the following routines:

`galsim.roman.getBandpasses`
    A utility to get a dictionary containing galsim.Bandpass objects for each of
    the Roman imaging bandpasses, which by default have AB zeropoints given using
    the GalSim zeropoint convention (see `getBandpasses` docstring for more details).

`galsim.roman.getSkyLevel`
    A utility to find the expected sky level due to zodiacal light at a given
    position, in a given band.

`galsim.roman.applyNonlinearity`
    A routine to apply detector nonlinearity of the type expected for Roman.

`galsim.roman.addReciprocityFailure`
    A routine to include the effects of reciprocity failure in images at
    the level expected for Roman.

`galsim.roman.applyIPC`
    A routine to incorporate the effects of interpixel capacitance in Roman images.

`galsim.roman.applyPersistence`
    A routine to incorporate the effects of persistence - the residual images
    from earlier exposures after resetting.

`galsim.roman.allDetectorEffects`
    A routine to add all sources of noise and all (implemented) detector
    effects to an image containing astronomical objects plus background.  In
    principle, users can simply use this routine instead of separately using
    the various routines like `galsim.roman.applyNonlinearity`.

`galsim.roman.getPSF`
    A routine to get a chromatic representation of the PSF in a single SCA.

`galsim.roman.getWCS`
    A routine to get the WCS for each SCA in the focal plane, for a given target RA, dec,
    and orientation angle.

`galsim.roman.findSCA`
    A routine that can take the WCS from `getWCS` and some sky position, and indicate in
    which SCA that position can be found, optionally including half of the gaps between
    SCAs (to identify positions that are in the focal plane array but in the gap between SCAs).

`galsim.roman.allowedPos`
    A routine to check whether Roman is allowed to look at a given position on a
    given date, given the constraints on orientation with respect to the sun.

`galsim.roman.bestPA`
    A routine to calculate the best observatory orientation angle for Roman when looking
    at a given position on a given date.

Another routine that may be necessary is `galsim.utilities.interleaveImages`.
The Roman PSFs at native Roman pixel scale are undersampled. A Nyquist-sampled PSF image can be
obtained by a two-step process:

    1. Call the `galsim.roman.getPSF` routine and convolve the PSF with the Roman pixel response
       to get the effective PSF.
    2. Draw the effective PSF onto an Image using drawImage routine, with a pixel scale lesser
       than the native pixel scale (using the 'method=no_pixel' option).

However, if pixel-level effects such as nonlinearity and interpixel capacitance must be applied to
the PSF images, then they must drawn at the native pixel scale. A Nyquist-sampled PSF image can be
obtained in such cases by generating multiple images with offsets (a dither sequence) and then
combining them using `galsim.utilities.interleaveImages`.


.. autofunction:: galsim.roman.getBandpasses

.. autofunction:: galsim.roman.getSkyLevel

.. autofunction:: galsim.roman.getPSF

.. autofunction:: galsim.roman.getWCS

.. autofunction:: galsim.roman.findSCA

.. autofunction:: galsim.roman.allowedPos

.. autofunction:: galsim.roman.bestPA

.. autofunction:: galsim.roman.convertCenter

.. autofunction:: galsim.roman.applyNonlinearity

.. autofunction:: galsim.roman.addReciprocityFailure

.. autofunction:: galsim.roman.applyIPC

.. autofunction:: galsim.roman.applyPersistence

.. autofunction:: galsim.roman.allDetectorEffects

