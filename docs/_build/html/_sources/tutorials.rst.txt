Tutorials
=========

The ``GalSim/examples`` directory contains demo files serve as tutorials on how to use the
GalSim code.

There are versions of each demo in both Python (``demo*.py``) and YAML (``demo*.yaml``).
The demos start fairly simple and progress to more sophisticated simulations, adding
a modest number of new features each time with copious documentation about the new features
being introduced.

The YAML files are run using the ``galsim`` executable, which parses the YAML file into a Python
dict and runs this through `The Config Module` in GalSim.  For complicated simulations,
we generally recommend using config files such as these, since they tend to be more quickly
readable than the Python scripts, which makes it easy to see how to modify them to effect
some desired change in the simulation.  For more information about running the ``galsim``
executable, see `The galsim Executable`.

Both versions of each demo produce identical output files.  Internally, this serves as a useful
test of the config parsing code.  But it also serves as a kind of implicit documentation
about how some of the config features are handled by GalSim.

Demo 1
------

:gh-link:`demo1.py <examples/demo1.py>`
:gh-link:`demo1.yaml <examples/demo1.yaml>`

This first demo is about as simple as it gets.  We draw an image of a single galaxy
convolved with a PSF and write it to disk.  We use a circular Gaussian profile for both the
PSF and the galaxy, and add a constant level of Gaussian noise to the image.

**New features in the Python file**:

- obj = galsim.Gaussian(flux, sigma)
- obj = galsim.Convolve([list of objects])
- image = obj.drawImage(scale)
- image.added_flux  (Only present after a drawImage command.)
- noise = galsim.GaussianNoise(sigma)
- image.addNoise(noise)
- image.write(file_name)
- image.FindAdaptiveMom()

**New features in the YAML file**:

- top level fields gal, psf, image, output
- obj type : Gaussian (flux, sigma)
- image : pixel_scale
- image : noise
- noise type : Gaussian (sigma)
- output : dir, file_name

Demo 2
------

:gh-link:`demo2.py <examples/demo2.py>`
:gh-link:`demo2.yaml <examples/demo2.yaml>`

This demo is a bit more sophisticated, but still pretty basic.  We still only make
a single image, but now the galaxy has an exponential radial profile and is sheared.
The PSF is a circular Moffat profile.  The noise is drawn from a Poisson distribution
using the flux from both the object and a background sky level to determine the
variance in each pixel.

**New features in the Python file**:

- obj = galsim.Exponential(flux, scale_radius)
- obj = galsim.Moffat(beta, flux, half_light_radius)
- obj = obj.shear(g1, g2)  -- with explanation of other ways to specify shear
- rng = galsim.BaseDeviate(seed)
- noise = galsim.PoissonNoise(rng, sky_level)
- galsim.hsm.EstimateShear(image, image_epsf)

**New features in the YAML file**:

- obj type : Exponential (flux, scale_radius)
- obj type : Moffat (flux, beta, half_light_radius)
- obj : shear
- shear type : G1G2 (g1, g2)
- noise type : Poisson (sky_level)
- image : random_seed


Demo 3
------

:gh-link:`demo3.py <examples/demo3.py>`
:gh-link:`demo3.yaml <examples/demo3.yaml>`

This demo gets reasonably close to including all the principal features of an image
from a ground-based telescope.  The galaxy is represented as the sum of a bulge and a disk,
where each component is represented by a sheared Sersic profile (with different Sersic
indices).  The PSF has both atmospheric and optical components.  The atmospheric
component is a Kolmogorov turbulent spectrum.  The optical component includes defocus,
coma and astigmatism, as well as obscuration from a secondary mirror.  The noise model
includes both a gain and read noise.  And finally, we include the effect of a slight
telescope distortion.

**New features in the Python file**:

- obj = galsim.Sersic(n, flux, half_light_radius)
- obj = galsim.Sersic(n, flux, scale_radius)
- obj = galsim.Kolmogorov(fwhm)
- obj = galsim.OpticalPSF(lam_over_diam, defocus, coma1, coma2, astig1, astig2, obscuration)
- obj = obj.shear(e, beta)  -- including how to specify an angle in GalSim
- shear = galsim.Shear(q, beta)
- obj = obj.shear(shear)
- obj3 = x1 * obj1 + x2 * obj2
- obj = obj.withFlux(flux)
- image = galsim.ImageF(image_size, image_size)
- image = obj.drawImage(image, wcs)
- image = obj.drawImage(method='sb')
- world_profile = wcs.toWorld(profile)
- shear3 = shear1 + shear2
- noise = galsim.CCDNoise(rng, sky_level, gain, read_noise)

**New features in the YAML file**:

- obj type : Sum (items)
- obj type : Convolve (items)
- obj type : Sersic (flux, n, half_light_radius)
- obj type : Sersic (flux, n, scale_radius)
- obj type : Kolmogorov (fwhm)
- obj type : OpticalPSF (lam_over_diam, defocus, coma1, coma2, astig1, astig2, obscuration)
- obj : ellip
- shear type : QBeta (q, beta) -- including how to specify an angle
- shear type : EBeta (e, beta)
- noise type : CCD (sky_level, gain, read_noise)
- image : size
- image : wcs
- wcs type : Shear
- output : psf

Demo 4
------

:gh-link:`demo4.py <examples/demo4.py>`
:gh-link:`demo4.yaml <examples/demo4.yaml>`

This demo is our first one to create multiple images.  Typically, you would want each object
to have at least some of its attributes vary when you are drawing multiple images (although
not necessarily -- you might just want different noise realization of the same profile).
The easiest way to do this is to read in the properties from a catalog, which is what we
do in this case.  The PSF is a truncated Moffat profile, and the galaxy is bulge plus disk.
Both components get many of their parameters from an input catalog.  We also shift the
profile by a fraction of a pixel in each direction so the effect of pixelization varies
among the images.  Each galaxy has the same applied shear.  The noise is simple Poisson noise.
We write the images out into a multi-extension fits file.

**New features in the Python file**:

- cat = galsim.Catalog(file_name, dir)
- obj = galsim.Moffat(beta, fwhm, trunc)
- obj = galsim.DeVaucouleurs(flux, half_light_radius)
- obj = galsim.RandomKnots(npoints, half_light_radius, flux)
- obj = galsim.Add([list of objects])
- obj = obj.shift(dx,dy)
- galsim.fits.writeMulti([list of images], file_name)

**New features in the YAML file**:

- obj type : Moffat (..., trunc)
- obj type : DeVaucouleurs (flux, half_light_radius)
- obj type : RandomKnots (npoints, half_light_radius, flux)
- value type : Catalog (col)
- obj : shift
- shift type : XY (x, y)
- shear type : E1E2 (e1, e2)
- image : xsize, ysize
- top level field input
- input : catalog (file_name, dir)
- output type : MultiFits (file_name, dir)
- Using both ellip and shear for the same object
- Using variables in a YAML file

Demo 5
------

:gh-link:`demo5.py <examples/demo5.py>`
:gh-link:`demo5.yaml <examples/demo5.yaml>`

This demo is intended to mimic a Great08 (Bridle, et al, 2010) LowNoise image.
We produce a single image made up of tiles of postage stamps for each individual object.
(We only do 10 x 10 postage stamps rather than 100 x 100 as they did in the interest of time.)
Each postage stamp is 40 x 40 pixels.  One image is all stars.  A second image is all galaxies.
The stars are truncated Moffat profiles.  The galaxies are Exponential profiles.
(Great08 mixed pure bulge and pure disk for its LowNoise run.  We just use disks to
make things simpler. However see demo3 for an example of using bulge+disk galaxies.)
The galaxies are oriented randomly, but in 90 degree-rotated pairs to cancel the effect of
shape noise.  The applied shear is the same for each galaxy.

**New features in the Python file**:

- ud = galsim.UniformDeviate(seed)
- gd = galsim.GaussianDeviate(ud, sigma)
- ccdnoise = galsim.CCDNoise(ud)
- image \*= scalar
- bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
- pos = bounds.center
- pos.x, pos.y
- sub_image = image[bounds]
- Build a single large image, and access sub-images within it.
- Set the galaxy size based on the PSF size and a resolution factor.
- Set the object flux according to a target S/N value.
- Use 90 degree-rotated pairs for the intrinsic galaxy shapes.
- Shift by a random (dx, dy) drawn from a unit circle top hat.

**New features in the YAML file**:

- gal : resolution
- gal : signal_to_noise
- stamp type : Ring (first, num)
- value type : RandomGaussian (sigma, min, max)
- angle type : Random
- shift type : RandomCircle (radius)
- image type : Tiled (nx_tiles, ny_tiles, stamp_xsize, stamp_ysize, border)
- output type : Fits (file_name, dir)
- output.psf : shift

Demo 6
------

:gh-link:`demo6.py <examples/demo6.py>`
:gh-link:`demo6.yaml <examples/demo6.yaml>`

This demo uses real galaxy images from COSMOS observations.  The catalog of real galaxy
images distributed with GalSim only includes 100 galaxies, but you can download a much
larger set of images as described in `Downloading the COSMOS Catalog`.

The galaxy images are already convolved with the effective PSF for the original
observations, so GalSim considers the galaxy profile to be the observed image deconvolved
by that PSF (also distributed with the galaxy data).
In this case, we then randomly rotate the galaxies, apply a given gravitational shear as
well as gravitational magnification, and then finally convolve by a double Gaussian PSF.
The final image can of course have any pixel scale, not just that of the original images.
The output for this demo is to a FITS "data cube".  With DS9, this can be viewed with a
slider to quickly move through the different images.

**New features in the Python file**:

- real_cat = galsim.RealGalaxyCatalog(file_name, dir)
- obj = galsim.Gaussian(fwhm, flux)
- obj = galsim.RealGalaxy(real_cat, index, flux)
- obj = obj.rotate(theta)
- obj = obj.magnify(mu)
- image += background
- noise = galsim.PoissonNoise()  # with no sky_level given
- obj.drawImage(..., offset)
- galsim.fits.writeCube([list of images], file_name)

**New features in the YAML file**:

- input : real_catalog (file_name, dir, image_dir)
- obj type : RealGalaxy (index)
- obj : rotate
- obj : magnify
- image : sky_level
- image : offset
- value type : Sequence (first, last, step)
- output type : DataCube (file_name, dir, nimages)
- Using YAML multiple document feature to do more than one thing


Demo 7
------

:gh-link:`demo7.py <examples/demo7.py>`
:gh-link:`demo7.yaml <examples/demo7.yaml>`

This demo introduces drawing profiles with photon shooting rather than doing the
convolution with an FFT.  It makes images using 5 different kinds of PSF and 5 different
kinds of galaxy.  Some of the parameters (flux, size and shape) are random variables, so
each of the 25 pairings is drawn 4 times with different realizations of the random numbers.
The profiles are drawn twice: once with the FFT method, and once with photon shooting.
The two images are drawn side by side into the same larger image so it is easy to
visually compare the results. The 100 total profiles are written to a FITS data cube,
which makes it easy to scroll through the images comparing the two drawing methods.

**New features in the Python file**:

- obj = galsim.Airy(lam_over_diam)
- obj = galsim.Sersic(n, half_light_radius, trunc)
- psf = galsim.OpticalPSF(..., aberrations=aberrations, ...)
- obj = obj.dilate(scale)
- str(obj)
- image.scale = pixel_scale
- obj.drawImage(image, method='fft')
- obj.drawImage(image, method='phot', max_extra_noise, rng)
- dev = galsim.PoissonDeviate(rng, mean)
- noise = galsim.DeviateNoise(dev)
- writeCube(..., compress='gzip')
- gsparams = galsim.GSParams(...)

**New features in the YAML file**:

- obj type : List (items)
- obj type : Airy (lam_over_diam)
- obj type : Sersic (..., trunc)
- obj : dilate
- value type : Sequence (..., repeat, index_key)
- value type : Random (min, max)
- image type : Tiled (..., stamp_size, xborder, yborder)
- stamp : draw_method (fft or phot)
- stamp : gsparams
- output : file_name with .gz, .bz2 or .fz extension automatically uses compression.

Demo 8
------

:gh-link:`demo8.py <examples/demo8.py>`
:gh-link:`demo8.yaml <examples/demo8.yaml>`

In this demo, we show how to run the GalSim config processing using a python dict rather
than using a config file.  The previous demos have shown what Python code corresponds to
the given YAML files.  Now we turn the tables
and show how to use some of the machinery in the GalSim configuration processing
from within Python itself.

This could be useful if you want to use the config machinery to build the images, but then
rather than write the images to disk, you want to keep them in memory and do further
processing with them.  (e.g. Run your shape measurement code on the images from within python.)

**New features in the Python file**:

- galsim.config.Process(config, logger)
- galsim.config.ProcessInput(config, logger)
- galsim.config.BuildFile(config, file_num, logger)
- image = galsim.config.BuildImage(config, image_num, logger)
- galsim.fits.read(file_name)

**New features in the YAML file**:

- stamp : retry_failures
- shear type : Eta1Eta2 (eta1, eta2)
- image : nproc

Demo 9
------

:gh-link:`demo9.py <examples/demo9.py>`
:gh-link:`demo9.yaml <examples/demo9.yaml>`

This script simulates cluster lensing or galaxy-galaxy lensing.  The gravitational shear
applied to each galaxy is calculated for an NFW halo mass profile.  We simulate observations
of galaxies around 20 different clusters -- 5 each of 4 different masses.  Each cluster
has its own file, organized into 4 directories (one for each mass).  For each cluster, we
draw 20 lensed galaxies located at random positions in the image.  The PSF is appropriate for a
space-like simulation.  (Some of the numbers used are the values for HST.)  And we apply
a cubic telescope distortion for the WCS.  Finally, we also output a truth catalog for each
output image that could be used for testing the accuracy of shape or flux measurements.

**New features in the Python file**:

- psf = OpticalPSF(lam, diam, ..., trefoil1, trefoil2, nstruts, strut_thick, strut_angle)
- im = galsim.ImageS(xsize, ysize, wcs)
- pos = galsim.PositionD(x, y)
- nfw = galsim.NFWHalo(mass, conc, z, omega_m, omega_lam)
- g1,g2 = nfw.getShear(pos, z)
- mag = nfw.getMagnification(pos, z)
- distdev = galsim.DistDeviate(rng, function, x_min, x_max)
- pos = bounds.true_center
- wcs = galsim.UVFunction(ufunc, vfunc, xfunc, yfunc, origin)
- wcs.toWorld(profile, image_pos)
- wcs.makeSkyImage(image, sky_level)
- image_pos = wcs.toImage(pos)
- image.invertSelf()
- truth_cat = galsim.OutputCatalog(names, types)
- bounds.isDefined()
- Make multiple output files.
- Place galaxies at random positions on a larger image.
- Write a bad pixel mask and a weight image as the second and third HDUs in each file.
- Use multiple processes to construct each file in parallel.

**New features in the YAML file**:

- obj type : OpticalPSF (lam, diam, ..., trefoil1, trefoil2, nstruts, strut_thick, strut_angle)
- obj type : InclinedExponential (scale_radius, scale_h_over_r, inclination)
- angle type : Radians
- shear type : NFWHaloShear (redshift)
- float type : NFWHaloMagnification (redshift)
- float type : RandomDistribution(function, x_min, x_max)
- input : nfw_halo (mass, conc, redshift)
- shear type : Sum (items)
- image type : Scattered (size, nobjects)
- wcs type : UVFunction (ufunc, vfunc, xfunc, yfunc, origin)
- str type : NumberedFile (root, num, ext, digits)
- str type : FormattedStr (format, items)
- pos type : RandomCircle (..., inner_radius)
- value type : Sequence (..., nitems)
- output : nproc
- output : weight
- output : badpix
- output : truth
- output : skip
- output : noclobber

Demo 10
-------

:gh-link:`demo10.py <examples/demo10.py>`
:gh-link:`demo10.yaml <examples/demo10.yaml>`

This script uses both a variable PSF and variable shear, taken from a power spectrum, along
the lines of a Great10 (Kitching, et al, 2012) image.  The galaxies are placed on a grid
(10 x 10 in this case, rather than 100 x 100 in the interest of time.)  Each postage stamp
is 48 x 48 pixels.  Instead of putting the PSF images on a separate image, we package them
as the second HDU in the file.  For the galaxies, we use a random selection from 5 specific
RealGalaxy objects, selected to be 5 particularly irregular ones. (These are taken from
the same catalog of 100 objects that demo6 used.)  The galaxies are oriented in a ring
test (Nakajima & Bernstein 2007) of 20 each.  And we again output a truth catalog with the
correct applied shear for each object (among other information).

**New features in the Python file**:

- im.wcs = galsim.OffsetWCS(scale, origin)
- rng = galsim.BaseDeviate(seed)
- obj = galsim.RealGalaxy(real_galaxy_catalog, id)
- obj = galsim.Convolve([list], real_space)
- ps = galsim.PowerSpectrum(e_power_function, b_power_function)
- g1,g2 = ps.buildGrid(grid_spacing, ngrid, rng)
- g1,g2 = ps.getShear(pos)
- galsim.random.permute(rng, list1, list2, ...)
- Choosing PSF parameters as a function of (x,y)
- Selecting RealGalaxy by ID rather than index.
- Putting the PSF image in a second HDU in the same file as the main image.
- Using PowerSpectrum for the applied shear.
- Doing a full ring test (i.e. not just 90 degree rotated pairs)

**New features in the YAML file**:

- obj type : Ring (..., full_rotation)
- obj type : RealGalaxy (..., id)
- type : Eval using world_pos variable, user-defined variables and math functions
- type : Current
- shear_value : PowerSpectrumShear
- pos_value : RTheta (r, theta)
- image type : Tiled (..., order)
- input : power_spectrum (e_power_function, b_power_function)
- output.psf : hdu, signal_to_noise, draw_method, offset
- output.truth : hdu
- Evaluated values in output.truth.columns

Demo 11
-------

:gh-link:`demo11.py <examples/demo11.py>`
:gh-link:`demo11.yaml <examples/demo11.yaml>`

This script uses a constant PSF from real data (an image read in from a bzipped FITS file, not a
parametric model) and variable shear and magnification according to some cosmological model for
which we have a tabulated shear power spectrum at specific k values only.  The 288 galaxies in the 0.1 x
0.1 degree field (representing a number density of 8/arcmin^2) are randomly located and
permitted to overlap.  For the galaxies, we use a mix of real and parametric galaxies modeled off
the COSMOS observations with the Hubble Space Telescope.  The real galaxies are similar to those
used in demo10.  The parametric galaxies are based on parametric fits to the same observed galaxies.
The flux and size distribution are thus realistic for an I < 23.5 magnitude limited sample.

**New features in the Python file**:

- coord = galsim.CelestialCoord(ra, dec)
- wcs = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin)
- wcs = galsim.TanWCS(affine, world_origin, units)
- psf = galsim.InterpolatedImage(psf_filename, scale, flux)
- tab = galsim.LookupTable(file)
- cosmos_cat = galsim.COSMOSCatalog(file_name, dir)
- gal = cosmos_cat.makeGalaxy(gal_type, rng, noise_pad_size)
- ps = galsim.PowerSpectrum(..., units)
- gal = gal.lens(g1, g2, mu)
- image.whitenNoise(correlated_noise)
- image.symmetrizeNoise(correlated_noise)
- vn = galsim.VariableGaussianNoise(rng, var_image)
- image.addNoise(cn)
- image.setOrigin(x,y)
- angle.dms(), angle.hms()
- Power spectrum shears and magnifications for non-gridded positions.
- Reading a compressed FITS image (using BZip2 compression).
- Writing a compressed FITS image (using Rice compression).
- Writing WCS information to a FITS header that ds9 reads as RA, Dec

**New features in the YAML file**:

- obj type : InterpolatedImage(image, scale)
- obj type : COSMOSGalaxy
- obj : scale_flus
- image : draw_method (no_pixel)
- input : power_spectrum (e_power_file, delta2, units)
- input : cosmos_catalog (file_name, dir, use_real)
- image : index_convention
- image.noise : whiten
- image.noise : symmetrize
- wcs type : Tan(dudx, dudy, dvdx, dvdy, units, origin, ra, dec)
- top level field eval_variables
- Power spectrum shears and magnifications for non-gridded positions.
- Reading a compressed FITS image (using BZip2 compression).
- Writing a compressed FITS image (using Rice compression).
- Using $ as a shorthand for Eval type.

Demo 12
-------

:gh-link:`demo12.py <examples/demo12.py>`
:gh-link:`demo12.yaml <examples/demo12.yaml>`

This demo introduces wavelength-dependent profiles.  Three kinds of chromatic profiles are
demonstrated:

1. A chromatic object representing a DeVaucouleurs galaxy with an early-type SED at redshift 0.8.
   This galaxy is drawn using the six LSST filters, which demonstrate that the galaxy is a
   g-band dropout.
2. A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs.
3. A wavelength-dependent atmospheric PSF, which includes the effect of differential chromatic
   refraction and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF
   is convolved with a simple Exponential galaxy.

**New features in the Python file**:

- SED = galsim.SED(wave, flambda, wave_type, flux_type)
- SED2 = SED.atRedshift(redshift)
- bandpass = galsim.Bandpass(filename, wave_type)
- bandpass2 = bandpass.truncate(relative_throughput)
- bandpass3 = bandpass2.thin(rel_err)
- gal = GSObject * SED
- obj = galsim.Add([list of ChromaticObjects])
- ChromaticObject.drawImage(bandpass)
- PSF = galsim.ChromaticAtmosphere(GSObject, base_wavelength, zenith_angle)

**New features in the YAML file**:

- sed : file_name, wave_type, flux_type, norm_flux_density, norm_wavelength,
        norm_flux, norm_bandpass
- bandpass : filename, wave_type, thin
- gal : redshift
- psf_type : ChromaticAtmosphere (base_profile, base_wavelength, latitude, HA)


Demo 13
-------

:gh-link:`demo13.py <examples/demo13.py>`
:gh-link:`demo13.yaml <examples/demo13.yaml>`

This script is intended to produce a relatively realistic scene of galaxies and stars as will
be observed by the Roman Space Telescope, including the Roman PSF, WCS, and various NIR detector
effects.

It introduces several non-idealities arising from NIR detectors, in particular those that will
be observed and accounted for in the Roman Space Telescope. Three such non-ideal effects are
demonstrated, in the order in which they are introduced in the detectors:

1. Reciprocity failure: Flux-dependent sensitivity of the detector.
2. Non-linearity: Charge-dependent gain in converting from units of electrons to ADU.  Non-linearity
   in some form is also relevant for CCDs in addition to NIR detectors.
3. Interpixel capacitance: Influence of charge in a pixel on the voltage reading of neighboring
   ones.

It also uses chromatic photon shooting, which is generally a more efficient way to simulate
scenes with many faint galaxies.  The default FFT method for drawing chromatically is fairly
slow, since it needs to integrate the image over the bandpass.  With photon shooting, the
photons are assigned wavelengths according to the SED of the galaxy, and then each photon has
the appropriate application of the chromatic PSF according to the wavelength.

**New features in the Python file**:

- image.quantize()
- obj = galsim.DeltaFunction(flux)
- galsim.roman.addReciprocityFailure(image)
- galsim.roman.applyNonlinearity(image)
- galsim.roman.applyIPC(image)
- galsim.roman.getBandpasses()
- galsim.roman.getPSF()
- galsim.roman.getWCS()
- galsim.roman.allowedPos()
- galsim.roman.getSkyLevel()

**New features in the YAML file**:

- Top-level field modules
- obj type: RomanPSF
- image type: RomanSCA
- draw_method=phot in conjunction with chromatic objects
- Multiple random seeds (particular so one can repeat the same values for multiple images)
- rng_num specification in various fields
- Multiple inputs of the same type.  Use num to specify which item in the list to use each time.


Advanced Simulations
--------------------

Great3 Simulations
^^^^^^^^^^^^^^^^^^

In the directory ``GalSim/examples/great3``,
there are YAML config files that perform essentially the same
simulations that were done for Great3.  The config apparatus had not matured sufficiently by the
time the Great3 sims were run, so these are not what the Great3 team used.  However, the files
in this directory produce essentially equivalent simulations as those used in Great3.

So far there are only config files for the cgc and rgc branches of Great3, but we plan to add
the files for the other branches (Issue #699).

**Significant features in these files**:

- template option to load another config file and then modify a few aspects of it. (e.g.
  :gh-link:`rgc.yaml <examples/great3/rgc.yaml>`)
- template option to load only a particular field from another config file. (e.g.
  :gh-link:`cgc_psf.yaml <examples/great3/cgc_psf.yaml>`)
- stamp.reject
- custom value type (e.g. Great3Reject in :gh-link:`cgc.yaml <examples/great3/cgc.yaml>`)
- custom extra output type (e.g. noise_free in :gh-link:`cgc.yaml <examples/great3/cgc.yaml>`)
- top-level module field
- use of '$' and '@' shorthand in Eval items.

DES Simulations
^^^^^^^^^^^^^^^

In the directory ``examples/des``,
there are YAML config files that showcase some of the classes
defined in the ``galsim.des`` module.  These are mostly gratuitous demos designed to showcase
various features, although :gh-link:`meds.yaml <examples/des/meds.yaml>`
is very close to a real simulation we actually used in DES for testing shear measurements.

**Significant features in these files**:

- top-level module field
- special object types from galsim.des module (e.g. DES_Shapelet and DES_PSFEx in
  :gh-link:`draw_psf.yaml <examples/des/draw_psf.yaml>`)
- special output type from galsim.des module (e.g. MEDS in :gh-link:`meds.yaml <examples/des/meds.yaml>`)
- custom value type (e.g. HSM_Shape_Measure in meds.yaml, LogNormal in :gh-link:`blend.yaml <examples/des/blend.yaml>`)
- custom WCS type (e.g. DES_Local in :gh-link:`meds.yaml <examples/des/meds.yaml>`)
- custom input type (e.g. des_wcs in :gh-link:`meds.yaml <examples/des/meds.yaml>`)
- custom stamp types (e.g. Blend in blend.yaml and BlendSet in :gh-link:`blendset.yaml <examples/des/blendset.yaml>`)
- custom extra output type (e.g. deblend in :gh-link:`blend.yaml <examples/des/blend.yaml>`)
