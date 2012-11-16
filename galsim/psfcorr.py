from . import _galsim

"""@file psfcorr.py 
Routines for adaptive moment estimation and PSF correction.

This file contains the python interface to C++ routines for estimation of second moments of images,
and for carrying out PSF correction using a variety of algorithms.  The algorithms are described in
Hirata & Seljak (2003; MNRAS, 343, 459), and were tested/characterized using real data in Mandelbaum
et al. (2005; MNRAS, 361, 1287).  We also define a python-level container for the outputs of these
codes, HSMShapeData, analogous to the C++-level CppHSMShapeData.

The moments that are estimated are "adaptive moments" (see the first paper cited above for details);
that is, they use an elliptical Gaussian weight that is matched to the image of the object being
measured.  The PSF correction includes several algorithms:

- One from Kaiser, Squires, & Broadhurts (1995), "KSB"

- One from Bernstein & Jarvis (2002), "BJ"

- One that represents a modification by Hirata & Seljak (2003) of methods in Bernstein & Jarvis
(2002), "LINEAR"

- One new method from Hirata & Seljak (2003), "REGAUSS"

These are all based on correction of moments, but with different sets of assumptions.  For more
detailed discussion on all of these algorithms, see the relevant papers above.
"""

class HSMShapeData(object):
    """A class to contain the outputs of using the HSM shape and moments measurement routines.

    At the C++ level, we have a container for the outputs of the HSM shape measurement routines.
    The HSMShapeData class is the analogous object at the python level.  It contains the following
    information about moment measurement (from either EstimateShearHSM or FindAdaptiveMom):

    - image_bounds: a galsim.Bounds object describing the image.

    - moments_status: the status flag resulting from moments measurement; -1 indicates no attempt to
      measure, 0 indicates success.

    - observed_shape: a galsim.Shear object representing the observed shape based on adaptive
      moments.

    - moments_sigma: size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
      not measured.

    - moments_amp: total image intensity for best-fit elliptical Gaussian from adaptive moments.
      This is related to flux via flux = (total image intensity)*(pixel scale)^2.

    - moments_centroid: a galsim.PositionD object representing the centroid based on adaptive
      moments.

    - moments_rho4: the weighted radial fourth moment of the image.

    - moments_n_iter: number of iterations needed to get adaptive moments, or 0 if not measured.

    If EstimateShearHSM was used, then the following fields related to PSF-corrected shape will also
    be populated:

    - correction_status: the status flag resulting from PSF correction; -1 indicates no attempt to
      measure, 0 indicates success.

    - corrected_shape: a galsim.Shear object representing the PSF-corrected shape.

    - corrected_shape_err: shape measurement uncertainty sigma_gamma per component.

    - correction_method: a string indicating the method of PSF correction (will be "None" if
      PSF-correction was not carried out).

    - resolution_factor: Resolution factor R_2;  0 indicates object is consistent with a PSF, 1 indicates
      perfect resolution.

    - error_message: a string containing any error messages from the attempt to carry out
      PSF-correction.

    The HSMShapeData object can be initialized completely empty, or with a CppHSMShapeData that is
    returned from calling the C++ routines for moments measurement and PSF correction.
    """
    def __init__(self, *args):
        # arg checking: require either a CppHSMShapeData, or nothing
        if len(args) > 1:
            raise TypeError("Too many arguments to initialize HSMShapeData!")
        elif len(args) == 1:
            if not isinstance(args[0], _galsim._CppHSMShapeData):
                raise TypeError("Argument to initialize HSMShapeData must be a _CppHSMShapeData!")
            self.image_bounds = args[0].image_bounds
            self.moments_status = args[0].moments_status
            self.observed_shape = galsim.Shear(args[0].observed_shape)
            self.moments_sigma = args[0].moments_sigma
            self.moments_amp = args[0].moments_amp
            self.moments_centroid = args[0].moments_centroid
            self.moments_rho4 = args[0].moments_rho4
            self.moments_n_iter = args[0].moments_n_iter
            self.correction_status = args[0].correction_status
            self.corrected_shape = galsim.Shear(args[0].corrected_shape)
            self.corrected_shape_err = args[0].corrected_shape_err
            self.correction_method = args[0].correction_method
            self.resolution_factor = args[0].resolution_factor
            self.error_message = args[0].error_message
        else:
            self.image_bounds = galsim.BoundsD()
            self.moments_status = -1
            self.observed_shape = galsim.Shear()
            self.moments_sigma = -1.0
            self.moments_amp = -1.0
            self.moments_centroid = galsim.PositionD()
            self.moments_rho4 = -1.0
            self.moments_n_iter = 0
            self.correction_status = -1
            self.corrected_shape = galsim.Shear()
            self.corrected_shape_err = -1.0
            self.correction_method = "None"
            self.resolution_factor = -1.0
            self.error_message = ""

def EstimateShearHSM(gal_image, PSF_image, sky_var = 0.0, shear_est = "REGAUSS", flags = 0xe,
                     guess_sig_gal = 5.0, guess_sig_PSF = 3.0, precision = 1.0e-6,
                     guess_x_centroid = -1000.0, guess_y_centroid = -1000.0, strict = True):
    """Carry out moments-based PSF correction routines.

    Carry out PSF correction using one of the methods of the HSM package (see references in
    docstring for file psfcorr.py) to estimate galaxy shears, correcting for the convolution by the
    PSF.

    This method works from Image inputs rather than galsim.base.GSObject inputs, which provides
    additional flexibility (e.g., it is possible to work from an Image that was read from file and
    corresponds to no particular GSObject), but this also means that users who wish to apply it to
    simple combinations of GSObjects (e.g., a Convolve) must take the additional step of drawing
    their GSObjects into Images.

    Note that the method will fail if the PSF or galaxy are too point-like to easily fit an
    elliptical Gaussian; when running on batches of many galaxies, it may be preferable to set
    `strict=False` and catch errors explicitly, as in the second example below.

    This function has a number of keyword parameters, many of which a typical user will not need to
    change from the default.

    Example usage
    -------------

    Typical application to a single object:

        >>> galaxy = galsim.Gaussian(flux = 1.0, sigma = 1.0)
        >>> galaxy.applyShear(g1=0.05, g2=0.0)  # shears the Gaussian by (0.05, 0) using the 
                                                # |g| = (a - b)/(a + b) definition
        >>> psf = galsim.Kolmogorov(flux = 1.0, fwhm = 0.7)
        >>> pixel = galsim.Pixel(xw = 0.2, yw = 0.2)
        >>> final = galsim.Convolve([galaxy, psf, pixel])
        >>> final_epsf = galsim.Convolve([psf, pixel])
        >>> final_image = final.draw(dx = 0.2)
        >>> final_epsf_image = final_epsf.draw(dx = 0.2)
        >>> result = galsim.EstimateShearHSM(final_image, final_epsf_image)
    
    After running the above code, `result.observed_shape` ["shape" = distortion, the
    (a^2 - b^2)/(a^2 + b^2) definition of ellipticity] is `(0.0876162,1.23478e-17)` and
    `result.corrected_shape` is `(0.0993412,-1.86255e-09)`, compared with the expected
    `(0.09975, 0)` for a perfect PSF correction method.

    The code below gives an example of how one could run this routine on a large batch of galaxies,
    explicitly catching and tracking any failures:

        n_image = 100
        n_fail = 0
        for i=0, range(n_image):
            #...some code defining this_image, this_final_epsf_image...
            result = galsim.EstimateShearHSM(this_image, this_final_epsf_image, strict = False)
            if result.error_message != "":
                n_fail += 1
        print "Number of failures: ", n_fail

    @param gal_image         The Image or ImageView of the galaxy being measured.
    @param PSF_image         The Image or ImageView for the PSF.
    @param sky_var           The variance of the sky level, used for estimating uncertainty on the
                             measured shape; default `sky_var = 0.`.
    @param shear_est         A string indicating the desired method of PSF correction: REGAUSS,
                             LINEAR, BJ, or KSB; default `shear_est = "REGAUSS"`.
    @param flags             A flag determining various aspects of the shape measurement process
                             (only necessary for REGAUSS); default `flags=0`.
    @param guess_sig_gal     Optional argument with an initial guess for the Gaussian sigma of the
                             galaxy, default `guess_sig_gal = 5.` (pixels).
    @param guess_sig_PSF     Optional argument with an initial guess for the Gaussian sigma of the
                             PSF, default `guess_sig_PSF = 3.` (pixels).
    @param precision         The convergence criterion for the moments; default `precision = 1e-6`.
    @param guess_x_centroid  An initial guess for the x component of the object centroid (useful in
                             case it is not located at the center, which is the default
                             assumption).
    @param guess_y_centroid  An initial guess for the y component of the object centroid (useful in
                             case it is not located at the center, which is the default
                             assumption).
    @param strict            If `strict = True` (default), then there will be a `RuntimeError` 
                             exception if shear estimation fails.  If set to `False`, then 
                             information about failures will be silently stored in the output 
                             HSMShapeData object.
    @return                  A HSMShapeData object containing the results of shape measurement.
    """
    gal_image_view = gal_image.view()
    PSF_image_view = PSF_image.view()
    try:
        result = _galsim._EstimateShearHSMView(gal_image_view, PSF_image_view, sky_var = sky_var,
                                               shear_est = shear_est, flags = flags,
                                               guess_sig_gal = guess_sig_gal,
                                               guess_sig_PSF = guess_sig_PSF,
                                               precision = precision, guess_x_centroid =
                                               guess_x_centroid, guess_y_centroid =
                                               guess_y_centroid)
    except RuntimeError as err:
        if (strict == True):
            raise err
        else:
            result = _galsim._CppHSMShapeData()
            result.error_message = err.message
    return galsim.HSMShapeData(result)

def FindAdaptiveMom(object_image, guess_sig = 5.0, precision = 1.0e-6, guess_x_centroid = -1000.0,
                    guess_y_centroid = -1000.0, strict = True):
    """Measure adaptive moments of an object.

    This method estimates the best-fit elliptical Gaussian to the object (see Hirata & Seljak 2003
    for more discussion of adaptive moments).  This elliptical Gaussian is computed iteratively
    by initially guessing a circular Gaussian that is used as a weight function, computing the
    weighted moments, recomputing the moments using the result of the previous step as the weight
    function, and so on until the moments that are measured are the same as those used for the
    weight function.  FindAdaptiveMom can be used either as a free function, or as a method of the
    ImageViewI(), ImageViewD() etc. classes.

    Like EstimateShearHSM, FindAdaptiveMom works on Image inputs, and fails if the object is small
    compared to the pixel scale.  For more details, see galsim.EstimateShearHSM (for doxygen
    documentation, see galsim.psfcorr.EstimateShearHSM).

    Example usage
    -------------

        >>> my_gaussian = galsim.Gaussian(flux = 1.0, sigma = 1.0)
        >>> my_gaussian_image = my_gaussian.draw(dx = 0.2)
        >>> my_moments = galsim.FindAdaptiveMom(my_gaussian_image)

    OR
    
        >>> my_moments = my_gaussian_image.FindAdaptiveMom()

    Assuming a successful measurement, the most relevant pieces of information are
    `my_moments.moments_sigma`, which is `|det(M)|^(1/4)` [=`sigma` for a circular Gaussian] and
    `my_moments.observed_shape`, which is a C++ CppShear.  

    Methods of the CppShear class can be used to get the distortion
    `(e1, e2) = (a^2 - b^2)/(a^2 + b^2)`, e.g. `my_moments.observed_shape.getE1()`, or to get the
    shear `g`, the conformal shear `eta`, and so on.

    @param object_image      The Image or ImageView for the object being measured.
    @param guess_sig         Optional argument with an initial guess for the Gaussian sigma of the
                             object, default `guess_sig = 5.0` (pixels).
    @param precision         The convergence criterion for the moments; default `precision = 1e-6`.
    @param guess_x_centroid  An initial guess for the x component of the object centroid (useful in
                             case it is not located at the center, which is the default
                             assumption).
    @param guess_y_centroid  An initial guess for the y component of the object centroid (useful in
                             case it is not located at the center, which is the default
                             assumption).
    @param strict            If `strict = True` (default), then there will be a `RuntimeError`
                             exception when moment measurement fails.  If set to `False`, then 
                             information about failures will be silently stored in the output 
                             HSMShapeData object.
    @return                  A HSMShapeData object containing the results of moment measurement.
    """
    object_image_view = object_image.view()
    try:
        result = _galsim._FindAdaptiveMomView(object_image_view, guess_sig = guess_sig, precision =
                                              precision, guess_x_centroid = guess_x_centroid,
                                              guess_y_centroid = guess_y_centroid)
    except RuntimeError as err:
        if (strict == True):
            raise err
        else:
            result = _galsim._CppHSMShapeData()
            result.error_message = err.message
    return galsim.HSMShapeData(result)

# make FindAdaptiveMom a method of Image and ImageView classes
for Class in _galsim.ImageView.itervalues():
    Class.FindAdaptiveMom = FindAdaptiveMom

for Class in _galsim.Image.itervalues():
    Class.FindAdaptiveMom = FindAdaptiveMom

del Class # cleanup public namespace
