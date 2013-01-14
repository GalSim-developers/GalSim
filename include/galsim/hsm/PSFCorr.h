/**
 * @file PSFCorr.h 
 *
 * @brief Contains functions for the hsm shape measurement code, which has functions to carry out
 *        PSF correction and measurement of adaptive moments. 
 *
 * All functions are in the hsm namespace.
 */

/****************************************************************
  Copyright 2003, 2004 Christopher Hirata: original code
  2007, 2009 Rachel Mandelbaum: minor modifications

  For a copy of the license, see COPYING; for more information,
  including contact information, see README.

  This file is part of the meas_shape distribution.

  Meas_shape is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  Meas_shape is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with meas_shape.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************/

/* structures and declarations for shape measurement */

/* object data type */

#include "../CppShear.h"
#include "../Image.h"
#include "../Bounds.h"

namespace galsim {
namespace hsm {

    /**
     * @brief A parameter used to optimize convolutions by cutting off galaxy profile.
     *
     * In the first step of the re-Gaussianization method of PSF correction, a Gaussian
     * approximation to the pre-seeing galaxy is calculated.  If re-Gaussianization is called with
     * the flag 0x4 (as is the default), then this approximation is cut off at nsig_rg sigma to save
     * computation time in convolutions.
     */
    const double nsig_rg = 3.0;

    /**
     * @brief A parameter used to optimize convolutions by cutting off PSF residual profile.
     *
     * In the re-Gaussianization method of PSF correction, a "PSF residual" (the difference between
     * the true PSF and its best-fit Gaussian approximation) is constructed. If re-Gaussianization
     * is called with the flag 0x8 (as is the default), then this PSF residual is cut off at
     * nsig_rg2 sigma to save computation time in convolutions.
     */
    const double nsig_rg2 = 3.6;

    /**
     * @brief A parameter for optimizing calculations of adaptive moments by cutting off profiles.
     *
     * This parameter is used to decide how many sigma^2 into the
     * Gaussian adaptive moment to extend the moment calculation, with the weight being defined as 0
     * beyond this point.  i.e., if max_moment_nsig2 is set to 25, then the Gaussian is extended to
     * (r^2/sigma^2)=25, with proper accounting for elliptical geometry.  If this parameter is set
     * to some very large number, then the weight is never set to zero and the exponential function is always
     * called. Note: GalSim script devel/modules/test_mom_timing.py was used to choose a value of 25
     * as being optimal, in that for the cases that were tested, the speedups were typically factors
     * of several, but the results of moments and shear estimation were changed by <10^-5.  Not all
     * possible cases were checked, and so for use of this code for unusual cases, we recommend that
     * users check that this value does not affect accuracy, and/or set it to some large value to
     * completely disable this optimization.
     */
    const double max_moment_nsig2 = 25.0;

    /**
     * @brief A parameter for how strictly the re-Gaussianization code treats small galaxies.
     *
     * If this parameter is 1, then the re-Gaussianization code does not impose a cut on the
     * apparent resolution before trying to measure the PSF-corrected shape of the galaxy; if 0,
     * then it is stricter.  Using the default value of 1 prevents the re-Gaussianization PSF
     * correction from completely failing at the beginning, before trying to do PSF correction, due
     * to the crudest possible PSF correction (Gaussian approximation) suggesting that the galaxy is
     * very small.  This could happen for some usable galaxies particularly when they have very
     * non-Gaussian surface brightness profiles -- for example, if there's a prominent bulge that
     * the adaptive moments attempt to fit, ignoring the more extended disk.  Setting a value of 1
     * is useful for keeping galaxies that would have failed for that reason.  If they later turn
     * out to be too small to really use, this will be reflected in the final estimate of the
     * resolution factor, and they can be rejected after the fact.
     */
    const int regauss_too_small = 1;

    /**
     * @brief The order to which circular adaptive moments should be calculated for KSB method.
     *
     * This parameter only affects calculations using the KSB method of PSF correction.  Warning:
     * deviating from default value of 2 results in code running more slowly, and results have not
     * been significantly tested.
     */
    const int adapt_order = 2;

    /// @brief The maximum number of iterations to use when calculating adaptive moments.
    const long max_mom2_iter = 400;

    /// @brief Number of iterations to report when code fails to converge within max_mom2_iter
    /// iterations.
    const long num_iter_default = -1;

    /// @brief Maximum shift in centroids and sigma between iterations for adaptive moments.
    const double bound_correct_weight = 0.25;

    /// @brief Maximum value for adaptive second moments before throwing exception.
    const double max_amoment = 8000.;

    /// @brief Maximum allowed x / y centroid shift for adaptive moments before throwing exception.
    const double max_ashift = 15.;

    /// @brief Use moments up to ksb_moments_max order for KSB method of PSF correction.
    const int ksb_moments_max = 4;

    /// @brief Value to report for ellipticities and resolution factor if shape measurement fails.
    const double failed_moments = -1000.;

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /**
     * @brief Exception class thrown by the adaptive moment and shape measurement routines in the
     * hsm namespace
     */
    class HSMError : public std::runtime_error {
    public:
        HSMError(const std::string& m="") :
            std::runtime_error("HSM Error: " + m) {}
    };

    //! @endcond

    /**
     * @brief Characterizes the shape and other parameters of objects.
     *
     * Describe the hsm shape-related parameters of some object (usually galaxy) before and after
     * PSF correction.  All ellipticities are defined as (1-q^2)/(1+q^2), with the 1st component
     * aligned with the pixel grid and the 2nd aligned at 45 degrees with respect to it.  There are
     * two choices for measurement type: 'e' = Bernstein & Jarvis (2002) ellipticity (or
     * distortion), 'g' = shear estimator = shear*responsivity.  The sigma is defined based on the
     * observed moments M_xx, M_xy, and M_yy as sigma = (Mxx Myy - M_xy^2)^(1/4) = 
     * [ det(M) ]^(1/4). */
    struct ObjectData 
    {
        double x0; ///< x centroid position within the postage stamp
        double y0; ///< y centroid position within the postage stamp
        double sigma; ///< size parameter
        double flux; ///< total flux
        double e1; ///< first ellipticity component
        double e2; ///< second ellipticity component
        double responsivity; ///< responsivity of ellipticity estimator 
        char meas_type; ///< type of measurement (see function description) 
        double resolution; ///< resolution factor (0=unresolved, 1=resolved) 
    };
  
    /**
     * @brief Struct containing information about the shape of an object.
     *
     * This representation of an object shape contains information about observed shapes and shape
     * estimators after PSF correction.  It also contains information about what PSF correction was
     * used; if no PSF correction was carried out and only the observed moments were measured, the
     * PSF correction method will be 'None'.  Note that observed shapes are bounded to lie in the
     * range |e| < 1 or |g| < 1, so they can be represented using a CppShear object.  In contrast,
     * the PSF-corrected distortions and shears are not bounded at a maximum of 1 since they are
     * shear estimators, and placing such a bound would bias the mean.  Thus, the corrected results
     * are not represented using CppShear objects, since it may not be possible to make a meaningful
     * per-object conversion from distortion to shear (e.g., if |e|>1).
     */
    struct CppHSMShapeData
    {
        /// @brief galsim::Bounds object describing the image of the object
        Bounds<int> image_bounds;

        // Now put information about moments measurement

        /// @brief Status after measuring adaptive moments; -1 indicates no attempt to measure them
        int moments_status;

        /// @brief galsim::CppShear object representing the observed shape
        CppShear observed_shape;

        /// @brief Size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
        /// not measured
        float moments_sigma;

        /// @brief Total image intensity for best-fit elliptical Gaussian from adaptive moments;
        /// note that this is not flux, since flux = (total image intensity)*(pixel scale)^2
        float moments_amp;

        /// @brief Centroid of best-fit elliptical Gaussian
        Position<double> moments_centroid;

        /// @brief The weighted radial fourth moment of the image
        double moments_rho4;

        /// @brief Number of iterations needed to get adaptive moments; 0 if not measured
        int moments_n_iter;

        // Then put information about PSF correction

        /// @brief Status after carrying out PSF correction; -1 indicates no attempt to do so
        int correction_status;

        /// @brief Estimated e1 after correcting for effects of the PSF, for methods that return a
        /// distortion.  Default value -10 if no correction carried out.
        float corrected_e1;

        /// @brief Estimated e2 after correcting for effects of the PSF, for methods that return a
        /// distortion.  Default value -10 if no correction carried out.
        float corrected_e2;

        /// @brief Estimated g1 after correcting for effects of the PSF, for methods that return a
        /// shear.  Default value -10 if no correction carried out.
        float corrected_g1;

        /// @brief Estimated g2 after correcting for effects of the PSF, for methods that return a
        /// shear.  Default value -10 if no correction carried out.
        float corrected_g2;

        /// @brief 'e' for PSF correction methods that return a distortion, 'g' for methods that
        /// return a shear.  "None" if PSF correction was not done.
        std::string meas_type;

        /// @brief Shape measurement uncertainty sigma_gamma (not sigma_e) per component
        float corrected_shape_err;

        /// @brief String indicating PSF-correction method; "None" if PSF correction was not done.
        std::string correction_method;

        /// @brief Resolution factor R_2; 0 indicates object is consistent with a PSF, 1 indicates
        /// perfect resolution; default -1
        float resolution_factor;

        /// @brief A string containing any error messages from the attempted measurements, to
        /// facilitate proper error handling in both C++ and python
        std::string error_message;

        /// @brief Constructor, setting defaults
        CppHSMShapeData() : image_bounds(galsim::Bounds<int>()), moments_status(-1),
            observed_shape(galsim::CppShear()), moments_sigma(-1.), moments_amp(-1.),
            moments_centroid(galsim::Position<double>(0.,0.)), moments_rho4(-1.), moments_n_iter(0),
            correction_status(-1), corrected_e1(-10.), corrected_e2(-10.), corrected_g1(-10.), 
            corrected_g2(-10.), meas_type("None"), corrected_shape_err(-1.),
            correction_method("None"), resolution_factor(-1.), error_message("")
        {}
    };

    /* functions that the user will want to call from outside */

    /**
     * @brief Carry out PSF correction directly using ImageViews.
     *
     * A template function to carry out one of the multiple possible methods of PSF correction using
     * the HSM package, directly accessing the input ImageViews.  The input arguments get repackaged
     * before calling general_shear_estimator, and results for the shape measurement are returned as
     * CppHSMShapeData.  There are two arguments that have default values, namely shear_est (the
     * type of shear estimator) and flags (for the REGAUSS method only).
     *
     * @param[in] gal_image The ImageView for the galaxy being measured
     * @param[in] PSF_image The ImageView for the PSF
     * @param[in] gal_mask_image The ImageView for the mask image to be applied to the galaxy being
     *            measured (integer array, 1=use pixel and 0=do not use pixel).
     * @param[in] sky_var The variance of the sky level, used for estimating uncertainty on the
     *            measured shape; default 0.
     * @param[in] *shear_est A string indicating the desired method of PSF correction: REGAUSS,
     *            LINEAR, BJ, or KSB; default REGAUSS.
     * @param[in] flags A flag determining various aspects of the shape measurement process (only
     *            necessary for REGAUSS); default 0xe.
     * @param[in] guess_sig_gal Optional argument with an initial guess for the Gaussian sigma of
     *            the galaxy, default 5.0 (pixels).
     * @param[in] guess_sig_PSF Optional argument with an initial guess for the Gaussian sigma of
     *            the PSF, default 3.0 (pixels).
     * @param[in] precision The convergence criterion for the moments; default 1e-6.
     * @param[in] guess_x_centroid Optional argument with an initial guess for the x centroid of the
     *            galaxy; if not set, then the code will try the center of the image.
     * @param[in] guess_y_centroid Optional argument with an initial guess for the y centroid of the
     *            galaxy; if not set, then the code will try the center of the image.
     * @return A CppHSMShapeData object containing the results of shape measurement.
     */
    template <typename T, typename U>
        CppHSMShapeData EstimateShearHSMView(const ImageView<T> &gal_image, const ImageView<U> &PSF_image,
                                             const ImageView<int> &gal_mask_image,
                                             float sky_var = 0.0, const char *shear_est = "REGAUSS",
                                             unsigned long flags = 0xe, double guess_sig_gal = 5.0,
                                             double guess_sig_PSF = 3.0, double precision = 1.0e-6,
                                             double guess_x_centroid = -1000.0,
                                             double guess_y_centroid = -1000.0);

    /**
     * @brief Measure the adaptive moments of an object directly using ImageViews.
     *
     * This function repackages the input ImageView in a format that find_ellipmom_2 accepts, in order
     * to iteratively compute the adaptive moments of an object.  The key result is the best-fit
     * elliptical Gaussian to the object, which is computed by initially guessing a circular
     * Gaussian that is used as a weight function, computing the weighted moments, recomputing the
     * moments using the result of the previous step as the weight function, and so on until the
     * moments that are measured are the same as those used for the weight function.  
     *
     * @param[in] object_image The ImageView for the object being measured.
     * @param[in] object_mask_image The ImageView for the mask image to be applied to the object
     *            being measured (integer array, 1=use pixel and 0=do not use pixel).
     * @param[in] guess_sig Optional argument with an initial guess for the Gaussian sigma of
     *            the object, default 5.0 (pixels).
     * @param[in] precision The convergence criterion for the moments; default 1e-6.
     * @param[in] guess_x_centroid Optional argument with an initial guess for the x centroid of the
     *            galaxy; if not set, then the code will try the center of the image.
     * @param[in] guess_y_centroid Optional argument with an initial guess for the y centroid of the
     *            galaxy; if not set, then the code will try the center of the image.
     * @return A CppHSMShapeData object containing the results of moment measurement.
     */
    template <typename T>
        CppHSMShapeData FindAdaptiveMomView(const ImageView<T> &object_image,
                                            const ImageView<int> &object_mask_image,
                                            double guess_sig = 5.0, double precision = 1.0e-6, 
                                            double guess_x_centroid = -1000.0,
                                            double guess_y_centroid = -1000.0);

    /**
     * @brief Carry out PSF correction.
     *
     * Carry out one of the multiple possible methods of PSF correction using the HSM package.
     * Results for the shape measurement are returned by modifying the galaxy data directly.  The
     * flags parameter is only used for the REGAUSS shape measurement method, and is defined as
     * follows: 0x1=recompute galaxy flux by summing unmasked pixels, 0x2=recompute galaxy flux from
     * Gaussian-quartic fit, 0x4=cut off Gaussian approximator at NSIG_RG sigma to save time,
     * 0x8=cut off PSF residual at NSIG_RG2 to save time.    
     * @param[in] gal_image The galaxy Image.
     * @param[in] gal_mask The galaxy mask Image (integers: 1=use pixel, 0=do not use pixel).
     * @param[in] PSF_image The PSF Image.
     * @param[in] PSF_mask The PSF mask Image (integers: 1=use pixel, 0=do not use pixel).
     * @param[in] gal_data The ObjectData object for the galaxy
     * @param[in] PSF_data The ObjectData object for the PSF
     * @param[in] shear_est A string indicating the desired method of PSF correction: REGAUSS,
     *            LINEAR, BJ, or KSB
     * @param[in] flags A parameter for REGAUSS (typical usage is 0xe).
     * @return A status flag that should be zero if the measurement was successful.
     */
    template <typename T, typename U>
    unsigned int general_shear_estimator(
        ConstImageView<T> gal_image, ConstImageView<int> gal_mask, ConstImageView<U> PSF_image, 
        ConstImageView<int> PSF_mask, ObjectData& gal_data, ObjectData& PSF_data, 
        const std::string& shear_est, unsigned long flags);

    /**
     * @brief Measure the adaptive moments of an object.
     *
     * This function iteratively computes the adaptive moments of an image, and tells the user the
     * results plus other diagnostic information.  The key result is the best-fit elliptical
     * Gaussian to the object, which is computed by initially guessing a circular Gaussian that is
     * used as a weight function, computing the weighted moments, recomputing the moments using the
     * result of the previous step as the weight function, and so on until the moments that are
     * measured are the same as those used for the weight function.
     * @param[in] data The Image for the object being measured.
     * @param[in] mask The mask Image for the object being measured  (integers: 1=use pixel, 0=do
     *                 not use pixel).
     * @param[out] A The amplitude of the best-fit elliptical Gaussian (total image intensity for
     *             the Gaussian is 2A). 
     * @param[out] x0 The x centroid of the best-fit elliptical Gaussian.
     * @param[out] y0 The y centroid of the best-fit elliptical Gaussian.
     * @param[out] Mxx The xx component of the moment matrix.
     * @param[out] Mxy The xy component of the moment matrix.
     * @param[out] Myy The yy component of the moment matrix.
     * @param[out] rho4 The weighted radial fourth moment.
     * @param[in] epsilon The required level of accuracy.
     * @param[out] num_iter The number of iterations needed to converge.
     */
    template <typename T>
    void find_ellipmom_2(
        ConstImageView<T> data, ConstImageView<int> mask, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double epsilon, int& num_iter);
  
}}
