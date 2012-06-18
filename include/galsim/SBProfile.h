// -*- c++ -*-
#ifndef SBPROFILE_H
#define SBPROFILE_H

/** 
 * @file SBProfile.h @brief Contains a class definition for two-dimensional Surface Brightness 
 * Profiles.
 *
 * The SBProfiles include common star, galaxy, and PSF shapes.
 */

#include <cmath>
#include <list>
#include <map>
#include <vector>
#include <algorithm>
#include <boost/shared_ptr.hpp>

#include "Std.h"
#include "Shear.h"
#include "FFT.h"
#include "Table.h"
#include "Random.h"
#include "Angle.h"
#include "integ/Int.h"

#include "Image.h"

#include "Laguerre.h"

#include "PhotonArray.h"

// ??? Ask for super-Nyquist sampling factor in draw??
namespace galsim {

    namespace sbp {

        // Magic numbers:

        /// @brief Constant giving minimum FFT size we're willing to do.
        const int minimum_fft_size = 128;

        /// @brief Constant giving maximum FFT size we're willing to do.
        const int maximum_fft_size = 4096;

        /**
         * @brief A threshold parameter used for setting the stepK value for FFTs.
         *
         * The FFT's stepK is set so that at most a fraction alias_threshold
         * of the flux of any profile is aliased.
         */
        const double alias_threshold = 5.e-3;

        /**
         * @brief A threshold parameter used for setting the maxK value for FFTs.
         *
         * The FFT's maxK is set so that the k-values that are excluded off the edge of 
         * the image are less than maxk_threshold.
         */
        const double maxk_threshold = 1.e-3;

        //@{
        /** 
         * @brief The target accuracy for realspace convolution.
         */
        const double realspace_conv_relerr = 1.e-3;
        const double realspace_conv_abserr = 1.e-6;
        //@}

        /**
         * @brief Accuracy of values in k-space.
         *
         * If a k-value is less than kvalue_accuracy, then it may be set to zero.
         * Similarly, if an alternate calculation has errors less than kvalue_accuracy,
         * then it may be used instead of an exact calculation.
         * Note: This does not necessarily imply that all kvalues are this accurate.
         * There may be cases where other choices we have made lead to errors greater 
         * than this.  But whenever we do an explicit calculation about this, this is
         * the value we use.
         *
         * Note that this would typically be more stringent than maxk_threshold.
         */
        const double kvalue_accuracy = 1.e-5;

        /**
         * @brief Accuracy of values in real space.
         *
         * If a value in real space is less than xvalue_accuracy, then it may be set to zero.
         * Similarly, if an alternate calculation has errors less than xvalue_accuracy,
         * then it may be used instead of an exact calculation.
         */
        const double xvalue_accuracy = 1.e-5;

        /**
         * @brief Accuracy of total flux for photon shooting
         *
         * The photon shooting algorithm sometimes needs to sample the radial profile
         * out to some value.  We choose the outer radius such that the integral encloses
         * at least (1-shoot_flux_accuracy) of the flux.
         */
        const double shoot_flux_accuracy = 1.e-5;

        //@{
        /**
         * @brief Target accuracy for other integrations in SBProfile
         *
         * For Sersic and Moffat, we numerically integrate the Hankel transform.
         * These are used for the precision in those integrals.
         */
        const double integration_relerr = kvalue_accuracy;
        const double integration_abserr = kvalue_accuracy * 1.e-2;
        //@}
    }

    /// @brief Exception class thrown by SBProfiles.
    class SBError : public std::runtime_error 
    {
    public:
        SBError(const std::string& m="") : std::runtime_error("SB Error: " + m) {}
    };

    /** 
     * @brief A base class representing all of the 2D surface brightness profiles that 
     * we know how to draw.
     *
     * Every SBProfile knows how to draw an Image<float> of itself in real and k space.  Each also
     * knows what is needed to prevent aliasing or truncation of itself when drawn.
     * **Note** that when you use the SBProfile::draw() routines you will get an image of 
     * **surface brightness** values in each pixel, not the flux that fell into the pixel.  To get
     * flux, you must multiply the image by (dx*dx).
     * drawK() routines are normalized such that I(0,0) is the total flux.
     * Currently we have the following possible implementations of SBProfile:
     * Basic shapes: SBBox, SBGaussian, SBExponential, SBAiry, SBSersic
     * SBLaguerre: Gauss-Laguerre expansion
     * SBDistort: affine transformation of another SBProfile
     * SBRotate: rotated version of another SBProfile
     * SBAdd: sum of SBProfiles
     * SBConvolve: convolution of other SBProfiles
     * SBInterpolatedImage: surface brightness profiles defined by an image and interpolant.
     * SBDeconvolve: deconvolve one SBProfile with another
     *
     * This isn't an abstract base class.  An SBProfile is a concrete object
     * which internally has a pointer to the implementation details (which _is_ an abstract
     * base class).  Furthermore, all SBProfiles are immutable objects.  Any changes
     * are made through modifiers that return a new object.  (e.g. setFlux,
     * shear, shift, etc.)  This means that we can safely make SBProfiles use shallow
     * copies, since that will never be confusing, which in turn means that SBProfiles
     * can be safely returned by value, used in containers (e.g. list<SBProfile>), etc.
     *
     * The only constructor for SBProfile is the copy constructor.  All SBProfiles need
     * to be created as one of the derived types that have real constructors.
     *
     * Well, technically, there is also a default constructor to make it easier to use
     * containers of SBProfiles.  However, it is an error to use an SBProfile that
     * has been default constructed for any purpose. 
     *
     * The assignment operator does a shallow copy, replacing the current contents of
     * the SBProfile with that of the rhs profile.  
     *
     */

    class SBDistort;

    class SBProfile
    {
    public:

        /**
         * @brief Default constructor for convenience only.  Do not use!
         *
         * This constructor is only provided so you can do things like:
         * @code
         * std::list<SBProfile> prof_list;
         * prof_list.push_back(psf);
         * prof_list.push_back(gal);
         * prof_list.push_back(pix);
         * @endcode
         * The default constructor for std::list strangely requires a default
         * constructor for the argument type, even though it isn't ever really used.
         */
        SBProfile() {}

        /// Only legitimate public constructor is a copy constructor.
        SBProfile(const SBProfile& rhs) : _pimpl(rhs._pimpl) {}

        /// operator= replaces the current contents with those of the rhs.
        SBProfile& operator=(const SBProfile& rhs) 
        { _pimpl = rhs._pimpl; return *this; }

        /// Destructor isn't virtual, since derived classes don't have anything to cleanup.
        ~SBProfile() 
        {
            // Not strictly necessary, but it sets the ptr to 0, so if somehow someone
            // manages to use an SBProfile after it was deleted, the assert(_pimpl.get())
            // will trigger an exception.
            _pimpl.reset();
        }

        /** 
         * @brief Return value of SBProfile at a chosen 2D position in real space.
         *
         * Assume all are real-valued.  xValue() may not be implemented for derived classes 
         * (SBConvolve) that require an FFT to determine real-space values.  In this case, an 
         * SBError will be thrown.
         *
         * @param[in] p 2D position in real space.
         */
        double xValue(const Position<double>& p) const
        { 
            assert(_pimpl.get());
            return _pimpl->xValue(p); 
        }

        /**
         * @brief Return value of SBProfile at a chosen 2D position in k space.
         *
         * @param[in] k 2D position in k space.
         */
        std::complex<double> kValue(const Position<double>& k) const
        { 
            assert(_pimpl.get());
            return _pimpl->kValue(k); 
        }

        //@{
        /**
         *  @brief Define the range over which the profile is not trivially zero.
         *
         *  These values are used when a real-space convolution is requested to define
         *  the appropriate range of integration.
         *  The implementation here is +- infinity for both x and y.  
         *  Derived classes may override this if they a have different range.
         */
        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { 
            assert(_pimpl.get());
            _pimpl->getXRange(xmin,xmax,splits); 
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        { 
            assert(_pimpl.get());
            _pimpl->getYRange(ymin,ymax,splits); 
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        { 
            assert(_pimpl.get());
            _pimpl->getYRangeX(x,ymin,ymax,splits); 
        }
        //@}

        /// @brief Value of k beyond which aliasing can be neglected.
        double maxK() const 
        { 
            assert(_pimpl.get());
            return _pimpl->maxK(); 
        }

        /// @brief Image pixel spacing that does not alias maxK.
        double nyquistDx() const { return M_PI / maxK(); }

        /// @brief Sampling in k space necessary to avoid folding too much of image in x space.
        double stepK() const 
        { 
            assert(_pimpl.get());
            return _pimpl->stepK(); 
        }

        /// @brief Characteristic that can affect efficiency of evaluation.
        bool isAxisymmetric() const 
        { 
            assert(_pimpl.get());
            return _pimpl->isAxisymmetric(); 
        }

        /**
         *  @brief The presence of hard edges help determine whether real space 
         *  convolution might be a better choice.
         */
        bool hasHardEdges() const
        {
            assert(_pimpl.get());
            return _pimpl->hasHardEdges();
        }

        /** 
         * @brief Characteristic that can affect efficiency of evaluation.
         *
         * SBProfile is "analytic" in the real domain if values can be determined immediately at 
         * any position through formula or a stored table (no DFT).
         */
        bool isAnalyticX() const 
        { 
            assert(_pimpl.get());
            return _pimpl->isAnalyticX(); 
        }

        /**
         * @brief Characteristic that can affect efficiency of evaluation.
         * 
         * SBProfile is "analytic" in the k domain if values can be determined immediately at any 
         * position through formula or a stored table (no DFT).
         */
        bool isAnalyticK() const 
        { 
            assert(_pimpl.get());
            return _pimpl->isAnalyticK(); 
        }

        /// @brief Returns (X, Y) centroid of SBProfile.
        Position<double> centroid() const 
        { 
            assert(_pimpl.get());
            return _pimpl->centroid(); 
        }

        /// @brief Get the total flux of the SBProfile.
        double getFlux() const 
        { 
            assert(_pimpl.get());
            return _pimpl->getFlux(); 
        }

        // ****Methods implemented in base class****

        // Transformations (all are special cases of affine transformations via SBDistort):

        /**
         * @brief Multiple the flux by fluxRatio
         *
         * This resets the internal pointer to a new SBProfile that wraps the old one
         * with a scaled flux.  This does not change any previous uses of the SBProfile, 
         * so if it had been used in some other context (e.g. in SBAdd or SBConvolve),
         * that object will be unchanged and still valid.
         */
        void scaleFlux(double fluxRatio);

        /**
         * @brief Set the flux to a new value
         *
         * This sets the flux to a new value.  As with scaleFlux, it does not invalidate
         * any previous uses of this object.
         */
        void setFlux(double flux);

        /**
         * @brief Apply a given Ellipse distortion (affine without rotation).
         *
         * This transforms the object by the given transformation.  As with scaleFlux,
         * it does not invalidate any previous uses of this object.
         */
        void applyDistortion(const Ellipse& e);

        /** 
         * @brief Apply a given shear.
         *
         * This shears the object by the given shear.  As with scaleFlux, it does not 
         * invalidate any previous uses of this object.
         */
        void applyShear(double g1, double g2);

        /** 
         * @brief Apply a given rotation.
         *
         * This rotates the object by the given angle.  As with scaleFlux, it does not 
         * invalidate any previous uses of this object.
         */
        void applyRotation(const Angle& theta);

        /**
         * @brief Apply a translation.
         *
         * This shears the object by the given amount.  As with scaleFlux, it does not 
         * invalidate any previous uses of this object.
         */
        void applyShift(double dx, double dy);

        /**
         * @brief Shoot photons through this SBProfile.
         *
         * Returns an array of photon coordinates and fluxes that are drawn from the light
         * distribution of this SBProfile.  Absolute value of each photons' flux should be
         * approximately equal, but some photons can be negative as needed to represent negative
         * regions.  Note that the ray-shooting method is not intended to produce a randomized value
         * of the total object flux, so do not assume that there will be sqrt(N) error on the flux.
         * In fact most implementations will return a PhotonArray with exactly correct flux, with
         * only the *distribution* of flux on the sky that will definitely have sampling noise.
         *
         * The one definitive gaurantee is that, in the limit of large number of photons, the
         * surface brightness distribution of the photons will converge on the SB pattern defined by
         * the object.
         *
         * Objects with regions of negative flux will result in creation of photons with negative
         * flux.  Absolute value of negative photons' flux should be nearly equal to the standard
         * flux of positive photons.  Shot-noise fluctuations between the number of positive and
         * negative photons will produce noise in the total net flux carried by the output
         * [PhotonArray](@ref PhotonArray).
         *
         * The typical implementation will be to take the integral of the absolute value of flux,
         * and divide it nearly equally into N photons.  The photons are then drawn from the
         * distribution of the *absolute value* of flux.  If a photon is drawn from a region of
         * negative flux, then that photon's flux is negated.  Because of cancellation, this means
         * that each photon will carry more than `getFlux()/N` flux if there are negative-flux
         * regions in the object.  It also means that during convolution, addition, or
         * interpolation, positive- and negative-flux photons can be contributing to the same region
         * of the image.  Their cancellation means that the shot noise may be substantially higher
         * than you would expect if you had only positive-flux photons.
         *
         * The photon flux may also vary slightly as a means of speeding up photon-shooting, as an
         * alternative to rejection sampling.  See `OneDimensionalDeviate` documentation.
         *
         * It should be rare to use this method or any `PhotonArray` in user code - the method
         * `drawShoot()` will more typically put the results directly into an image.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const 
        { 
            assert(_pimpl.get());
            return _pimpl->shoot(N,ud); 
        }

        /**
         * @brief Return expectation value of flux in positive photons when shoot() is called
         *
         * Returns expectation value of flux returned in positive-valued photons when 
         * [shoot()](@ref shoot)
         * is called for this object.  Default implementation is to return getFlux(), if it is
         * positive, or 0 otherwise, which will be
         * the case when the SBProfile is constructed entirely from elements of the same sign.
         *
         * It should be generally true that `getPositiveFlux() - getNegativeFlux()` returns the same
         * thing as `getFlux()`.  Small difference may accrue from finite numerical accuracy in
         * cases involving lookup tables, etc.
         *
         * @returns Expected positive-photon flux.
         */
        double getPositiveFlux() const 
        { 
            assert(_pimpl.get());
            return _pimpl->getPositiveFlux(); 
        }

        /**
         * @brief Return expectation value of absolute value of flux in negative photons from 
         * shoot()
         *
         * Returns expectation value of (absolute value of) flux returned in negative-valued photons
         * when shoot() is called for this object.  
         * Default implementation is to return getFlux() if it is negative, 0 otherwise,
         * which will be the case when the SBProfile is constructed entirely from elements that
         * have the same sign.
         *
         * It should be generally true that `getPositiveFlux() - getNegativeFlux()` returns the
         * same thing as `getFlux()`.  Small difference may accrue from finite numerical accuracy 
         * in cases involving lookup tables, etc.
         *
         * @returns Expected absolute value of negative-photon flux.
         */
        double getNegativeFlux() const 
        { 
            assert(_pimpl.get());
            return _pimpl->getNegativeFlux(); 
        }

        // **** Drawing routines ****
        //@{
        /**
         * @brief Draw this SBProfile into Image by shooting photons.
         *
         * The input image must have defined boundaries and pixel scale.  The photons generated by
         * the shoot() method will be binned into the target Image.  See caveats in `shoot()`
         * docstring.  Input `Image` will be cleared before drawing in the photons.  Scale and
         * location of the `Image` pixels will not be altered.  Photons falling outside the `Image`
         * range will be ignored.
         *
         * It is important to remember that the `Image` produced by `drawShoot` represents the
         * `SBProfile` _as convolved with the square Image pixel._ So do not expect an exact match,
         * even in the limit of large photon number, between the outputs of `draw` and `drawShoot`.
         * You should convolve the `SBProfile` with an `SBBox(dx)` in order to match what will be
         * produced by `drawShoot` onto an image with pixel scale `dx`.
         *
         * @param[in] img Image to draw on.
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        template <typename T>
        void drawShoot(ImageView<T> img, double N, UniformDeviate& ud) const;
        template <typename T>
        void drawShoot(Image<T>& img, double N, UniformDeviate& ud) const 
        { drawShoot(img.view(), N, ud); }
        //@}

        /** 
         * @brief Draw an image of the SBProfile in real space.
         *
         * A square image will be
         * drawn which is big enough to avoid "folding."  If drawing is done using FFT,
         * it will be scaled up to a power of 2, or 3x2^n, whichever fits.
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image  may be calculated internally on a larger grid to avoid folding.
         * The default draw() routines decide internally whether image can be drawn directly
         * in real space or needs to be done via FFT from k space.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in] dx    grid on which SBProfile is drawn has pitch `dx`; given `dx=0.` default, 
         *                  routine will choose `dx` to be at least fine enough for Nyquist sampling
         *                  at `maxK()`.  If you specify dx, image will be drawn with this `dx` and
         *                  you will receive an image with the aliased frequencies included.
         * @param[in] wmult specifying `wmult>1` will draw an image that is `wmult` times larger 
         *                  than the default choice, i.e. it will have finer sampling in k space 
         *                  and have less folding.
         * @returns image (as ImageViewF; if another type is preferred, then use the draw 
         *                  method that takes an image as argument)
         */
        ImageView<float> draw(double dx=0., int wmult=1) const;

        //@{
        /** 
         * @brief Draw the SBProfile in real space returning the summed flux.
         *
         * If the input image `img` is an Image (not ImageView) and has null dimension,
         * a square image will be drawn which is big enough to avoid "folding." 
         * If drawing is done using FFT, it will be scaled up to a power of 2, or 3x2^n, 
         * whichever fits.
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image may be calculated internally on a larger grid to avoid folding.
         * The default draw() routines decide internally whether image can be drawn directly
         * in real space or needs to be done via FFT from k space.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   image (any of ImageF, ImageD, ImageS, ImageI)
         * @param[in] dx    grid on which SBProfile is drawn has pitch `dx`; given `dx=0.` default, 
         *                  routine will choose `dx` to be at least fine enough for Nyquist sampling
         *                  at `maxK()`.  If you specify dx, image will be drawn with this `dx` and
         *                  you will receive an image with the aliased frequencies included.
         * @param[in] wmult specifying `wmult>1` will draw an image that is `wmult` times larger 
         *                  than the default choice, i.e. it will have finer sampling in k space 
         *                  and have less folding.
         * @returns summed flux.
         */
        template <typename T>
        double draw(Image<T>& image, double dx=0., int wmult=1) const; 
        template <typename T>
        double draw(ImageView<T>& image, double dx=0., int wmult=1) const; 
        //@}

        //@{
        /** 
         * @brief Draw an image of the SBProfile in real space forcing the use of real methods 
         * where we have a formula for x values.
         *
         * If the input image is an Image and has null dimension, a square image will be
         * drawn which is big enough to avoid "folding." 
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image may be calculated internally on a larger grid to avoid folding.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   image (any of ImageF, ImageD, ImageS, ImageI or views)
         * @param[in] dx    grid on which SBProfile is drawn has pitch `dx`; given `dx=0.` default, 
         *                  routine will choose `dx` to be at least fine enough for Nyquist sampling
         *                  at `maxK()`.  If you specify dx, image will be drawn with this `dx` and
         *                  you will receive an image with the aliased frequencies included.
         * @param[in] wmult specifying `wmult>1` will draw an image that is `wmult` times larger 
         *                  than the default choice, i.e. it will have finer sampling in k space 
         *                  and have less folding.
         * @returns summed flux.
         */
        template <typename T>
        double plainDraw(ImageView<T>& image, double dx=0., int wmult=1) const; 
        template <typename T>
        double plainDraw(Image<T>& image, double dx=0., int wmult=1) const; 
        //@}

        //@{
        /** 
         * @brief Draw an image of the SBProfile in real space forcing the use of Fourier transform
         * from k space.
         *
         * If the input image is an Image and has null dimension, a square image will be
         * drawn which is big enough to avoid "folding."  Drawing is done using FFT,
         * and the image will be scaled up to a power of 2, or 3x2^n, whicher fits.
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image may be calculated internally on a larger grid to avoid folding.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   image (any of ImageF, ImageD, ImageS, ImageI or views)
         * @param[in] dx    grid on which SBProfile is drawn has pitch `dx`; given `dx=0.` default, 
         *                  routine will choose `dx` to be at least fine enough for Nyquist sampling
         *                  at `maxK()`.  If you specify dx, image will be drawn with this `dx` and
         *                  you will receive an image with the aliased frequencies included.
         * @param[in] wmult specifying `wmult>1` will draw an image that is `wmult` times larger 
         *                  than the default choice, i.e. it will have finer sampling in k space 
         *                  and have less folding.
         * @returns summed flux.
         */
        template <typename T>
        double fourierDraw(ImageView<T>& image, double dx=0., int wmult=1) const; 
        template <typename T>
        double fourierDraw(Image<T>& image, double dx=0., int wmult=1) const; 
        //@}

        //@{
        /** 
         * @brief Draw an image of the SBProfile in k space.
         *
         * For drawing in k space: routines are analagous to real space, except 2 images are 
         * needed since the SBProfile is complex.
         * If the input images are Image's and have null dimension, square 
         * images will be drawn which are big enough to avoid "folding."  If drawing is done using 
         * FFT, they will be scaled up to a power of 2, or 3x2^n, whicher fits.
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image may be calculated internally on a larger grid to avoid folding in real space.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   re image of real argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views). 
         * @param[in,out]   im image of imaginary argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views).
         * @param[in] dk    grid on which SBProfile is drawn has pitch `dk`; given `dk=0.` default,
         *                  routine will choose `dk` necessary to avoid folding of image in real 
         *                  space.  If you specify `dk`, image will be drawn with this `dk` and
         *                  you will receive an image with folding artifacts included.
         * @param[in] wmult specifying `wmult>1` will expand the size drawn in k space.
         */
        template <typename T>
        void drawK(ImageView<T>& re, ImageView<T>& im, double dk=0., int wmult=1) const; 
        template <typename T>
        void drawK(Image<T>& re, Image<T>& im, double dk=0., int wmult=1) const; 
        //@}

        //@{
        /** 
         * @brief Draw an image of the SBProfile in k space forcing the use of k space methods 
         * where we have a formula for k values.
         *
         * For drawing in k space: routines are analagous to real space, except 2 images are 
         * needed since the SBProfile is complex.  If the input images are Image's and have
         * null dimension, square images will be drawn which are big enough to 
         * avoid "folding."
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   re image of real argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views).
         * @param[in,out]   im image of imaginary argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views).
         * @param[in] dk    grid on which SBProfile is drawn has pitch `dk`; given `dk=0.` default,
         *                  routine will choose `dk` necessary to avoid folding of image in real 
         *                  space.  If you specify `dk`, image will be drawn with this `dk` and
         *                  you will receive an image with folding artifacts included.
         * @param[in] wmult specifying `wmult>1` will expand the size drawn in k space.
         */
        template <typename T>
        void plainDrawK(ImageView<T>& re, ImageView<T>& im, double dk=0., int wmult=1) const; 
        template <typename T>
        void plainDrawK(Image<T>& re, Image<T>& im, double dk=0., int wmult=1) const; 
        //@}

        //@{
        /**
         * @brief Draw an image of the SBProfile in k space forcing the use of Fourier transform 
         * from real space.
         *
         * For drawing in k space: routines are analagous to real space, except 2 images are 
         * needed since the SBProfile is complex.
         * If the input images are Image's and have null dimension, square 
         * images will be drawn which are big enough to avoid "folding."  Drawing is done using FFT,
         * and the images will be scaled up to a power of 2, or 3x2^n, whicher fits.
         * If input image has finite dimensions then these will be used, although in an FFT the 
         * image may be calculated internally on a larger grid to avoid folding in real space.
         * Note that if you give an input image, its origin may be redefined by the time it comes 
         * back.
         *
         * @param[in,out]   re image of real argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views).
         * @param[in,out]   im image of imaginary argument of SBProfile in k space (any of ImageF,
         *                  ImageD, ImageS, ImageI or views).
         * @param[in] dk    grid on which SBProfile is drawn has pitch `dk`; given `dk=0.` default,
         *                  routine will choose `dk` necessary to avoid folding of image in real 
         *                  space.  If you specify `dk`, image will be drawn with this `dk` and
         *                  you will receive an image with folding artifacts included.
         * @param[in] wmult specifying `wmult>1` will expand the size drawn in k space.
         */
        template <typename T>
        void fourierDrawK(ImageView<T>& re, ImageView<T>& im, double dk=0., int wmult=1) const; 
        template <typename T>
        void fourierDrawK(Image<T>& re, Image<T>& im, double dk=0., int wmult=1) const; 
        //@}

    protected:

    class SBProfileImpl
    {
    public:

        // Constructor doesn't do anything
        SBProfileImpl() {}

        // Virtual destructor
        virtual ~SBProfileImpl() {}

        // Pure virtual functions:
        virtual double xValue(const Position<double>& p) const =0;
        virtual std::complex<double> kValue(const Position<double>& k) const =0; 
        virtual double maxK() const =0; 
        virtual double stepK() const =0;
        virtual bool isAxisymmetric() const =0;
        virtual bool hasHardEdges() const =0;
        virtual bool isAnalyticX() const =0; 
        virtual bool isAnalyticK() const =0; 
        virtual Position<double> centroid() const = 0;
        virtual double getFlux() const =0; 
        virtual PhotonArray shoot(int N, UniformDeviate& ud) const=0;

        // Functions with default implementations:
        virtual void getXRange(double& xmin, double& xmax,
                               std::vector<double>& /*splits*/) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }

        virtual void getYRange(double& ymin, double& ymax,
                               std::vector<double>& /*splits*/) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }

        virtual void getYRangeX(double /*x*/, double& ymin, double& ymax,
                               std::vector<double>& splits) const 
        { getYRange(ymin,ymax,splits); }

        virtual double getPositiveFlux() const { return getFlux()>0. ? getFlux() : 0.; }

        virtual double getNegativeFlux() const { return getFlux()>0. ? 0. : -getFlux(); }

        // Utility for drawing into Image data structures.
        template <typename T>
        double fillXImage(ImageView<T>& image, double dx) const  // return flux integral
        { return doFillXImage(image, dx); }

        // Utility for drawing a k grid into FFT data structures 
        virtual void fillKGrid(KTable& kt) const;

        // Utility for drawing an x grid into FFT data structures 
        virtual void fillXGrid(XTable& xt) const;

        // Virtual functions cannot be templates, so to make fillXImage work like a virtual
        // function, we have it call these, which need to include all the types of Image
        // that we want to use.
        //
        // Then in the derived class, these functions should call a template version of 
        // fillXImage in that derived class that implements the functionality you want.
        virtual double doFillXImage(ImageView<float>& image, double dx) const
        { return doFillXImage2(image,dx); }
        virtual double doFillXImage(ImageView<double>& image, double dx) const
        { return doFillXImage2(image,dx); }

        // Here in the base class, we need yet another name for the version that actually
        // implements this as a template:
        template <typename T>
        double doFillXImage2(ImageView<T>& image, double dx) const;

    private:
        // Copy constructor and op= are undefined.
        SBProfileImpl(const SBProfileImpl& rhs);
        void operator=(const SBProfileImpl& rhs);
    };

        // Classes that need to be able to access _pimpl object of other SBProfiles
        // are made friends.
        friend class SBAdd;
        friend class SBDistort;
        friend class SBConvolve;
        friend class SBDeconvolve;

        // Regular constructor only available to derived classes
        SBProfile(SBProfileImpl* pimpl) : _pimpl(pimpl) {}

        boost::shared_ptr<SBProfileImpl> _pimpl;
    };

    /** 
     * @brief Sums SBProfiles. 
     */
    class SBAdd : public SBProfile 
    {
    public:

        /** 
         * @brief Constructor, 2 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         */
        SBAdd(const SBProfile& s1, const SBProfile& s2) :
            SBProfile(new SBAddImpl(s1,s2)) {}

        /** 
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist list of SBProfiles.
         */
        SBAdd(const std::list<SBProfile>& slist) : 
            SBProfile(new SBAddImpl(slist)) {}

        /// @brief Copy constructor.
        SBAdd(const SBAdd& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBAdd() {}

    protected:

    class SBAddImpl : public SBProfileImpl
    {
    public:
        SBAddImpl(const SBProfile& s1, const SBProfile& s2)
        { add(s1); add(s2); initialize(); }

        SBAddImpl(const std::list<SBProfile>& slist)
        {
            for (ConstIter sptr = slist.begin(); sptr!=slist.end(); ++sptr)
                add(*sptr); 
            initialize();
        }

        ~SBAddImpl() {}

        void add(const SBProfile& rhs);

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const { return _maxMaxK; }
        double stepK() const { return _minStepK; }

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { 
            xmin = integ::MOCK_INF; xmax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double xmin_1, xmax_1;
                pptr->getXRange(xmin_1,xmax_1,splits);
                if (xmin_1 < xmin) xmin = xmin_1;
                if (xmax_1 > xmax) xmax = xmax_1;
            }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = integ::MOCK_INF; ymax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRange(ymin_1,ymax_1,splits);
                if (ymin_1 < ymin) ymin = ymin_1;
                if (ymax_1 > ymax) ymax = ymax_1;
            }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = integ::MOCK_INF; ymax = -integ::MOCK_INF; 
            for (ConstIter pptr = _plist.begin(); pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRangeX(x,ymin_1,ymax_1,splits);
                if (ymin_1 < ymin) ymin = ymin_1;
                if (ymax_1 > ymax) ymax = ymax_1;
            }
        }

        bool isAxisymmetric() const { return _allAxisymmetric; }
        bool hasHardEdges() const { return _anyHardEdges; }
        bool isAnalyticX() const { return _allAnalyticX; }
        bool isAnalyticK() const { return _allAnalyticK; }

        Position<double> centroid() const 
        { return Position<double>(_sumfx / _sumflux, _sumfy / _sumflux); }

        double getFlux() const { return _sumflux; }

        /**
         * @brief Shoot photons through this SBAdd.
         *
         * SBAdd will divide the N photons among its summands with probabilities proportional to
         * their integrated (absolute) fluxes.  Note that the order of photons in output array will
         * not be random as different summands' outputs are simply concatenated.
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        /**
         * @brief Give total positive flux of all summands
         *
         * Note that `getPositiveFlux()` return from SBAdd may not equal the integral of positive
         * regions of the image, because summands could have positive and negative regions
         * cancelling each other.  Rather it will be the sum of the `getPositiveFlux()` of all the
         * images.
         * @returns Total positive flux of all summands
         */
        double getPositiveFlux() const;

        /** @brief Give absolute value of total negative flux of all summands
         *
         * Note that `getNegativeFlux()` return from SBAdd may not equal the integral of negative
         * regions of the image, because summands could have positive and negative regions
         * cancelling each other. Rather it will be the sum of the `getNegativeFlux()` of all the
         * images.
         * @returns Absolute value of total negative flux of all summands
         */
        double getNegativeFlux() const;

        // Overrides for better efficiency
        void fillKGrid(KTable& kt) const;
        void fillXGrid(XTable& xt) const;

        typedef std::list<SBProfile>::iterator Iter;
        typedef std::list<SBProfile>::const_iterator ConstIter;

    private:
        /// @brief The plist content is a pointer to a fresh copy of the summands.
        std::list<SBProfile> _plist; 
        double _sumflux; ///< Keeps track of the cumulated flux of all summands.
        double _sumfx; ///< Keeps track of the cumulated `fx` of all summands.
        double _sumfy; ///< Keeps track of the cumulated `fy` of all summands.
        double _maxMaxK; ///< Keeps track of the cumulated `maxK()` of all summands.
        double _minStepK; ///< Keeps track of the cumulated `minStepK()` of all summands.
        double _minMinX; ///< Keeps track of the cumulated `minX()` of all summands.
        double _maxMaxX; ///< Keeps track of the cumulated `maxX()` of all summands.
        double _minMinY; ///< Keeps track of the cumulated `minY()` of all summands.
        double _maxMaxY; ///< Keeps track of the cumulated `maxY()` of all summands.

        /// @brief Keeps track of the cumulated `isAxisymmetric()` properties of all summands.
        bool _allAxisymmetric;

        /// @brief Keeps track of whether any summands have hard edges.
        bool _anyHardEdges;

        /// @brief Keeps track of the cumulated `isAnalyticX()` property of all summands. 
        bool _allAnalyticX; 

        /// @brief Keeps track of the cumulated `isAnalyticK()` properties of all summands.
        bool _allAnalyticK; 

        void initialize();  ///< Sets all private book-keeping variables to starting state.

        // Copy constructor and op= are undefined.
        SBAddImpl(const SBAddImpl& rhs);
        void operator=(const SBAddImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBAdd& rhs);
    };

    /**
     * @brief An affine transformation of another SBProfile.
     *
     * Origin of original shape will now appear at `_cen`.
     * Flux is NOT conserved in transformation - surface brightness is preserved.
     * We keep track of all distortions in a 2x2 matrix `M = [(A B), (C D)]` = [row1, row2] 
     * plus a 2-element Positon object `cen` for the shift, and a flux scaling,
     * in addition to the scaling implicit in the matrix M = abs(det(M)).
     */
    class SBDistort : public SBProfile 
    {
    public:
        /** 
         * @brief General constructor.
         *
         * @param[in] sbin SBProfile being distorted
         * @param[in] mA A element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mB B element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mC C element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mD D element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] cen 2-element (x, y) Position for the translational shift.
         * @param[in] fluxScaling Amount by which the flux should be multiplied.
         */
        SBDistort(const SBProfile& sbin,
                  double mA, double mB, double mC, double mD, 
                  const Position<double>& cen=Position<double>(0.,0.), double fluxScaling=1.) :
            SBProfile(new SBDistortImpl(sbin,mA,mB,mC,mD,cen,fluxScaling)) {}

        /** 
         * @brief Construct from an input Ellipse 
         *
         * @param[in] sbin SBProfile being distorted.
         * @param[in] e  Ellipse.
         * @param[in] fluxScaling Amount by which the flux should be multiplied.
         */
        SBDistort(const SBProfile& sbin, const Ellipse& e=Ellipse(), double fluxScaling=1.) : 
            SBProfile(new SBDistortImpl(sbin,e,fluxScaling)) {}

        /// @brief Copy constructor
        SBDistort(const SBDistort& rhs) : SBProfile(rhs) {}

        /// @brief Destructor
        ~SBDistort() {}

    protected:

    class SBDistortImpl : public SBProfileImpl
    {
    public:

        SBDistortImpl(const SBProfile& sbin, double mA, double mB, double mC, double mD,
                      const Position<double>& cen, double fluxScaling);

        SBDistortImpl(const SBProfile& sbin, const Ellipse& e, double fluxScaling);

        ~SBDistortImpl() {}

        double xValue(const Position<double>& p) const 
        { return _adaptee.xValue(inv(p-_cen)) * _fluxScaling; }

        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return _stillIsAxisymmetric; }
        bool hasHardEdges() const { return _adaptee.hasHardEdges(); }
        bool isAnalyticX() const { return _adaptee.isAnalyticX(); }
        bool isAnalyticK() const { return _adaptee.isAnalyticK(); }

        double maxK() const { return _adaptee.maxK() / _minor; }
        double stepK() const { return _adaptee.stepK() / _major; }

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const;

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const;

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const;

        Position<double> centroid() const { return _cen+fwd(_adaptee.centroid()); }

        double getFlux() const { return _adaptee.getFlux()*_absdet; }

        double getPositiveFlux() const 
        { return _adaptee.getPositiveFlux()*_absdet; }
        double getNegativeFlux() const 
        { return _adaptee.getNegativeFlux()*_absdet; }

        /**
         * @brief Shoot photons through this SBDistort.
         *
         * SBDistort will simply apply the affine distortion to coordinates of photons
         * generated by its adaptee, and rescale the flux by the determinant of the distortion
         * matrix.
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        // Override for better efficiency:
        void fillKGrid(KTable& kt) const; 

    private:
        SBProfile _adaptee; ///< SBProfile being adapted/distorted

        double _mA; ///< A element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mB; ///< B element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mC; ///< C element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        double _mD; ///< D element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
        Position<double> _cen;  ///< Centroid position.

        // Calculate and save these:
        double _absdet;  ///< Determinant (flux magnification) of `M` matrix * fluxScaling
        double _fluxScaling;  ///< Amount to multiply flux by.
        double _invdet;  ///< Inverse determinant of `M` matrix.
        double _major; ///< Major axis of ellipse produced from unit circle.
        double _minor; ///< Minor axis of ellipse produced from unit circle.
        bool _stillIsAxisymmetric; ///< Is output SBProfile shape still circular?
        double _xmin, _xmax, _ymin, _ymax; ///< Ranges propagated from adaptee
        double _coeff_b, _coeff_c, _coeff_c2; ///< Values used in getYRangeX(x,ymin,ymax);
        std::vector<double> _xsplits, _ysplits; ///< Good split points for the intetegrals

        void initialize();

        /** 
         * @brief Forward coordinate transform with `M` matrix.
         *
         * @param[in] p input position.
         * @returns transformed position.
         */
        Position<double> fwd(const Position<double>& p) const 
        { return _fwd(_mA,_mB,_mC,_mD,p.x,p.y,_invdet); }

        /// @brief Forward coordinate transform with transpose of `M` matrix.
        Position<double> fwdT(const Position<double>& p) const 
        { return _fwd(_mA,_mC,_mB,_mD,p.x,p.y,_invdet); }

        /// @brief Inverse coordinate transform with `M` matrix.
        Position<double> inv(const Position<double>& p) const 
        { return _inv(_mA,_mB,_mC,_mD,p.x,p.y,_invdet); }

        /// @brief Returns the the k value (no phase).
        std::complex<double> kValueNoPhase(const Position<double>& k) const;

        std::complex<double> (*_kValue)(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& k, const Position<double>& cen);
        std::complex<double> (*_kValueNoPhase)(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& , const Position<double>& );

        Position<double> (*_fwd)(
            double mA, double mB, double mC, double mD, double x, double y, double );
        Position<double> (*_inv)(
            double mA, double mB, double mC, double mD, double x, double y, double invdet);

        // Copy constructor and op= are undefined.
        SBDistortImpl(const SBDistortImpl& rhs);
        void operator=(const SBDistortImpl& rhs);
    };

        static std::complex<double> _kValueNoPhaseNoDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueNoPhaseWithDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueWithPhase(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& k, const Position<double>& cen);

        static Position<double> _fwd_normal(
            double mA, double mB, double mC, double mD, double x, double y, double )
        { return Position<double>(mA*x + mB*y, mC*x + mD*y); }
        static Position<double> _inv_normal(
            double mA, double mB, double mC, double mD, double x, double y, double invdet)
        { return Position<double>(invdet*(mD*x - mB*y), invdet*(-mC*x + mA*y)); }
        static Position<double> _ident(
            double , double , double , double , double x, double y, double )
        { return Position<double>(x,y); }

    private:
        // op= is undefined
        void operator=(const SBDistort& rhs);
    };

    // Defined in RealSpaceConvolve.cpp
    double RealSpaceConvolve(
        const SBProfile& p1, const SBProfile& p2, const Position<double>& pos, double flux);

    /**
     * @brief Convolve SBProfiles.
     *
     * Convolve one, two, three or more SBProfiles together.
     *
     * The profiles to be convolved may be provided either as the first 1, 2, or 3
     * parameters in the constructor, or as a std::list<SBProfile*>.
     *
     * The convolution will normally be done using discrete Fourier transforms of 
     * each of the component profiles, multiplying them together, and then transforming
     * back to real space.
     *
     * The stepK used for the k-space image will be (Sum 1/stepK()^2)^(-1/2)
     * where the sum is over all teh components being convolved.  Since the size of 
     * the convolved image scales roughly as the quadrature sum of the components,
     * this should be close to Pi/Rmax where Rmax is the radius that encloses
     * all but (1-alias_threshold) of the flux in the final convolved image..
     *
     * The maxK used for the k-space image will be the minimum of the maxK() calculated for
     * each component.  Since the k-space images are multiplied, if one of them is 
     * essentially zero beyond some k value, then that will be true of the final image
     * as well.
     *
     * There is also an option to do the convolution as integrals in real space.
     * Each constructor has an optional boolean parmeter, real_space, that comes
     * immediately after the list of profiles to convolve.  Currently, the real-space
     * integration is only enabled for 2 profiles.  (Aside from the trivial implementaion
     * for 1 profile.)  If you try to use it for more than 2 profiles, an exception will
     * be thrown.  
     *
     * The real-space convolution is normally slower than the DFT convolution.
     * The exception is if both component profiles have hard edges.  e.g. a truncated
     * Moffat with a Box.  In that case, the maxK for each component is quite large
     * since the ringing dies off fairly slowly.  So it can be quicker to use 
     * real-space convolution instead.
     *
     * Finally, there is another optional parameter in the constructors which can set
     * an overall flux scale for the final image.  The nominal flux is normally just
     * the product of the fluxes of each of the component profiles.  However, if you
     * set f to something other than 1, then the final flux will be f times the 
     * product of the component fluxes.
     */
    class SBConvolve : public SBProfile 
    {
    public:
        /**
         * @brief Constructor, 2 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const SBProfile& s1, const SBProfile& s2, bool real_space=false) :
            SBProfile(new SBConvolveImpl(s1,s2,real_space)) {}

        /**
         * @brief Constructor, 3 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         * @param[in] s3 third SBProfile.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const SBProfile& s1, const SBProfile& s2, const SBProfile& s3,
                   bool real_space=false) :
            SBProfile(new SBConvolveImpl(s1,s2,s3,real_space)) {}

        /**
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist Input: list of SBProfiles.
         * @param[in] real_space  Do convolution in real space? (default `real_space = false`).
         */
        SBConvolve(const std::list<SBProfile>& slist, bool real_space=false) :
            SBProfile(new SBConvolveImpl(slist,real_space)) {}

        /// @brief Copy constructor.
        SBConvolve(const SBConvolve& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBConvolve() {}

    protected:

    class SBConvolveImpl: public SBProfileImpl
    {
    public:

        SBConvolveImpl(const SBProfile& s1, const SBProfile& s2, bool real_space) : 
            _real_space(real_space)
        { add(s1);  add(s2); initialize(); }

        SBConvolveImpl(const SBProfile& s1, const SBProfile& s2, const SBProfile& s3,
                       bool real_space) :
            _real_space(real_space)
        { add(s1);  add(s2);  add(s3); initialize(); }

        SBConvolveImpl(const std::list<SBProfile>& slist, bool real_space) :
            _real_space(real_space)
        {
            for (ConstIter sptr = slist.begin(); sptr!=slist.end(); ++sptr) 
                add(*sptr); 
            initialize(); 
        }

        ~SBConvolveImpl() {}

        void add(const SBProfile& rhs); 

        // Do the real-space convolution to calculate this.
        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return _isStillAxisymmetric; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return _real_space; }
        bool isAnalyticK() const { return !_real_space; }    // convolvees must all meet this
        double maxK() const { return _minMaxK; }
        double stepK() const { return _netStepK; }

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { 
            // Getting the splits correct would require a bit of work.
            // So if we ever do real-space convolutions where one of the elements 
            // is (or includes) another convolution, we might want to rework this a 
            // bit.  But I don't think this is really every going to be used, so
            // I didn't try to get that right.  (Note: ignoring the splits won't be
            // wrong -- just not optimal.)
            std::vector<double> splits0;
            ConstIter pptr = _plist.begin();
            pptr->getXRange(xmin,xmax,splits0);
            for (++pptr; pptr!=_plist.end(); ++pptr) {
                double xmin_1, xmax_1;
                pptr->getXRange(xmin_1,xmax_1,splits0);
                xmin += xmin_1;
                xmax += xmax_1;
            }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            std::vector<double> splits0;
            ConstIter pptr = _plist.begin();
            pptr->getYRange(ymin,ymax,splits0);
            for (++pptr; pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRange(ymin_1,ymax_1,splits0);
                ymin += ymin_1;
                ymax += ymax_1;
            }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            std::vector<double> splits0;
            ConstIter pptr = _plist.begin();
            pptr->getYRangeX(x,ymin,ymax,splits0);
            for (++pptr; pptr!=_plist.end(); ++pptr) {
                double ymin_1, ymax_1;
                pptr->getYRangeX(x,ymin_1,ymax_1,splits0);
                ymin += ymin_1;
                ymax += ymax_1;
            }
        }

        Position<double> centroid() const 
        { return Position<double>(_x0, _y0); }

        double getFlux() const { return _fluxProduct; }

        double getPositiveFlux() const;
        double getNegativeFlux() const;
        /**
         * @brief Shoot photons through this SBConvolve.
         *
         * SBConvolve will add the displacements of photons generated by each convolved component.
         * Their fluxes are multiplied (modulo factor of N).
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        void fillKGrid(KTable& kt) const;

    private:
        typedef std::list<SBProfile>::iterator Iter;
        typedef std::list<SBProfile>::const_iterator ConstIter;

        std::list<SBProfile> _plist; ///< list of profiles to convolve
        double _x0; ///< Centroid position in x.
        double _y0; ///< Centroid position in y.
        bool _isStillAxisymmetric; ///< Is output SBProfile shape still circular?
        double _minMaxK; ///< Minimum maxK() of the convolved SBProfiles.
        double _netStepK; ///< Minimum stepK() of the convolved SBProfiles.
        double _sumMinX; ///< sum of minX() of the convolved SBProfiles.
        double _sumMaxX; ///< sum of maxX() of the convolved SBProfiles.
        double _sumMinY; ///< sum of minY() of the convolved SBProfiles.
        double _sumMaxY; ///< sum of maxY() of the convolved SBProfiles.
        double _fluxProduct; ///< Flux of the product.
        bool _real_space; ///< Whether to do convolution as an integral in real space.

        void initialize();

        // Copy constructor and op= are undefined.
        SBConvolveImpl(const SBConvolveImpl& rhs);
        void operator=(const SBConvolveImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBConvolve& rhs);
    };

    /////////////////////////////////////////////////////////////////////////////////////
    // Below here are the concrete "atomic" SBProfile types
    /////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Gaussian Surface Brightness Profile
     *
     * The Gaussian Surface Brightness Profile is characterized by two properties, its `flux`
     * and the characteristic size `sigma` where the radial profile of the circular Gaussian
     * drops off as `exp[-r^2 / (2. * sigma^2)]`.
     * The maxK() and stepK() are for the SBGaussian are chosen to extend to 4 sigma in both 
     * real and k domains, or more if needed to reach the `alias_threshold` spec.
     */
    class SBGaussian : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] sigma  characteristic size, surface brightness scales as 
         *                   `exp[-r^2 / (2. * sigma^2)]`.
         * @param[in] flux   flux of the Surface Brightness Profile (default `flux = 1.`).
         */
        SBGaussian(double sigma, double flux=1.) : 
            SBProfile(new SBGaussianImpl(sigma, flux)) {}

        /// @brief Copy constructor.
        SBGaussian(const SBGaussian& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBGaussian() {}

        /// @brief Returns the characteristic size sigma of the Gaussian profile.
        double getSigma() const 
        { 
            assert(dynamic_cast<const SBGaussianImpl*>(_pimpl.get()));
            return dynamic_cast<const SBGaussianImpl&>(*_pimpl).getSigma(); 
        }

    protected:
    class SBGaussianImpl : public SBProfileImpl
    {
    public:
      SBGaussianImpl(double sigma, double flux);

        ~SBGaussianImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        /**
         * @brief Shoot photons through this SBGaussian.
         *
         * SBGaussian shoots photons by analytic transformation of the unit disk.  Slightly more
         * than 2 uniform deviates are drawn per photon, with some analytic function calls (sqrt,
         * etc.)
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        double getSigma() const { return _sigma; }

    private:
        double _flux; ///< Flux of the Surface Brightness Profile.

        /// Characteristic size, surface brightness scales as `exp[-r^2 / (2. * sigma^2)]`.
        double _sigma;
        double _sigma_sq; ///< Calculated value: sigma*sigma
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _norm; ///< flux / sigma^2 / 2pi

        // Copy constructor and op= are undefined.
        SBGaussianImpl(const SBGaussianImpl& rhs);
        void operator=(const SBGaussianImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBGaussian& rhs);
    };

    /**
     * @brief Sersic Surface Brightness Profile.
     *
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic 
     * index `n`, its `flux` and the half-light radius `re`.
     */
    class SBSersic : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] n     Sersic index.
         * @param[in] re    half-light radius.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBSersic(double n, double re, double flux=1.) : 
            SBProfile(new SBSersicImpl(n, re, flux)) {}

        /// @brief Copy constructor.
        SBSersic(const SBSersic& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBSersic() {}

        /// @brief Returns the Sersic index `n` of the profile.
        double getN() const
        { 
            assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
            return dynamic_cast<const SBSersicImpl&>(*_pimpl).getN(); 
        }

        /// @brief Returns the half light radius of the Sersic profile.
        double getHalfLightRadius() const 
        {
            assert(dynamic_cast<const SBSersicImpl*>(_pimpl.get()));
            return dynamic_cast<const SBSersicImpl&>(*_pimpl).getHalfLightRadius(); 
        }

    protected:
        class SersicInfo;

    class SBSersicImpl : public SBProfileImpl
    {
    public:
        SBSersicImpl(double n, double re, double flux);

        ~SBSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; 
            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        /// @brief Sersic photon shooting done by rescaling photons from appropriate `SersicInfo`
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        double getN() const { return _n; }
        double getHalfLightRadius() const { return _re; }

    private:
        double _n; ///< Sersic index.
        double _flux; ///< Flux.
        double _re;   ///< Half-light radius.
        double _re_sq; ///< Calculated value: _re*_re
        double _norm; ///< Calculated value: _flux/_re_sq
        double _ksq_max; ///< The ksq_max value from info rescaled with this re value.

        const SersicInfo* _info; ///< Points to info structure for this n.

        // Copy constructor and op= are undefined.
        SBSersicImpl(const SBSersicImpl& rhs);
        void operator=(const SBSersicImpl& rhs);
    };

        /** 
         * @brief Subclass of `SBSersic` which provides the un-normalized radial function.
         *
         * Serves as interface to `OneDimensionalDeviate` used for sampling from this 
         * distribution.
         */
        class SersicRadialFunction: public FluxDensity 
        {
        public:
            /**
             * @brief Constructor
             *
             * @param[in] n  Sersic index
             * @param[in] b  Factor which makes radius argument enclose half the flux.
             */
            SersicRadialFunction(double n, double b): _invn(1./n), _b(b) {}
            /**
             * @brief The un-normalized Sersic function
             * @param[in] r radius, in units of half-light radius.
             * @returns Sersic function, normalized to unity at origin
             */
            double operator()(double r) const { return std::exp(-_b*std::pow(r,_invn)); } 
        private:
            double _invn; ///> 1/n
            double _b;  /// radial normalization constant
        };

        /// @brief A private class that caches the needed parameters for each Sersic index `n`.
        class SersicInfo 
        {
        public:
            /** 
             * @brief Constructor
             * @param[in] n Sersic index
             */
            SersicInfo(double n); 

            /// @brief Destructor: deletes photon-shooting classes if necessary
            ~SersicInfo() {}

            /** 
             * @brief Returns the real space value of the Sersic function,
             * normalized to unit flux (see private attributes).
             * @param[in] xsq The *square* of the radius, in units of half-light radius.
             * Avoids taking sqrt in most user code.
             * @returns Value of Sersic function, normalized to unit flux.
             */
            double xValue(double xsq) const;

            /// @brief Looks up the k value for the SBProfile from a lookup table.
            double kValue(double ksq) const;

            double maxK() const { return _maxK; }
            double stepK() const { return _stepK; }

            double getKsqMax() const { return _ksq_max; }

            /**
             * @brief Shoot photons through unit-size, unnormalized profile
             * Sersic profiles are sampled with a numerical method, using class
             * `OneDimensionalDeviate`.
             *
             * @param[in] N Total number of photons to produce.
             * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
             * @returns PhotonArray containing all the photons' info.
             */
            PhotonArray shoot(int N, UniformDeviate& ud) const;

        private:
            SersicInfo(const SersicInfo& rhs); ///< Hides the copy constructor.
            void operator=(const SersicInfo& rhs); ///<Hide assignment operator.

            double _n; ///< Sersic index.

            /** 
             * @brief Scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`,
             * calculated from Sersic index `n` and half-light radius `re`.
             */
            double _b; 

            double _inv2n;   ///< `1 / (2 * n)`
            double _maxK;    ///< Value of k beyond which aliasing can be neglected.
            double _stepK;   ///< Sampling in k space necessary to avoid folding 

            double _norm; ///< Amplitude scaling in Sersic profile `exp(-b*pow(xsq,inv2n))`.
            double _kderiv2; ///< Quadratic dependence near k=0.
            double _kderiv4; ///< Quartic dependence near k=0.
            Table<double,double> _ft;  ///< Lookup table for Fourier transform of Moffat.
            double _ksq_min; ///< Minimum ksq to use lookup table.
            double _ksq_max; ///< Maximum ksq to use lookup table.

            /// Function class used for photon shooting
            boost::shared_ptr<SersicRadialFunction> _radial;  

            /// Class that does numerical photon shooting
            boost::shared_ptr<OneDimensionalDeviate> _sampler;   

            double findMaxR(double missing_flux_fraction, double gamma2n);
        };

        /** 
         * @brief A map to hold one copy of the SersicInfo for each `n` ever used during the 
         * program run.  Make one static copy of this map.  
         * *Be careful of this when multithreading:*
         * Should build one `SBSersic` with each `n` value before dispatching multiple threads.
         */
        class InfoBarn : public std::map<double, boost::shared_ptr<SersicInfo> > 
        {
        public:

            /**
             * @brief Get the SersicInfo table for a specified `n`.
             *
             * @param[in] n Sersic index for which the information table is required.
             */
            const SersicInfo* get(double n) 
            {
                /** 
                 * @brief The currently hardwired max number of Sersic `n` info tables that can be 
                 * stored.  Should be plenty.
                 */
                const int MAX_SERSIC_TABLES = 100; 

                MapIter it = _map.find(n);
                if (it == _map.end()) {
                    boost::shared_ptr<SersicInfo> info(new SersicInfo(n));
                    _map[n] = info;
                    if (int(_map.size()) > MAX_SERSIC_TABLES)
                        throw SBError("Storing Sersic info for too many n values");
                    return info.get();
                } else {
                    return it->second.get();
                }
            }

        private:
            typedef std::map<double, boost::shared_ptr<SersicInfo> >::iterator MapIter;
            std::map<double, boost::shared_ptr<SersicInfo> > _map;
        };

        /// One static map of all `SersicInfo` structures for whole program.
        static InfoBarn nmap; 

    private:
        // op= is undefined
        void operator=(const SBSersic& rhs);
    };

    /** 
     * @brief Exponential Surface Brightness Profile.  
     *
     * This is a special case of the Sersic profile, but is given a separate class since the 
     * Fourier transform has closed form and can be generated without lookup tables.
     */
    class SBExponential : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in 
         * SBSersic.
         *
         * @param[in] r0    scale length for the profile that scales as `exp[-(r / r0)]`, NOT the 
         *                  half-light radius `re`.
         * @param[in] flux  flux (default `flux = 1.`).
         */
         SBExponential(double r0, double flux=1.) :
             SBProfile(new SBExponentialImpl(r0, flux)) {}

        /// @brief Copy constructor.
        SBExponential(const SBExponential& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBExponential() {}

        /// @brief Returns the scale radius of the Exponential profile.
        double getScaleRadius() const 
        { 
            assert(dynamic_cast<const SBExponentialImpl*>(_pimpl.get()));
            return dynamic_cast<const SBExponentialImpl&>(*_pimpl).getScaleRadius(); 
        }

    protected:
    class SBExponentialImpl : public SBProfileImpl
    {
    public:

        SBExponentialImpl(double r0, double flux);

        ~SBExponentialImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        { 
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; 
            if (std::abs(x/_r0) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getScaleRadius() const { return _r0; }

        PhotonArray shoot(int N, UniformDeviate& ud) const;

    private:
        double _flux; ///< Flux.
        double _r0;   ///< Characteristic size of profile `exp[-(r / r0)]`.
        double _r0_sq; ///< Calculated value: r0*r0
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _norm; ///< flux / r0^2 / 2pi

        // Copy constructor and op= are undefined.
        SBExponentialImpl(const SBExponentialImpl& rhs);
        void operator=(const SBExponentialImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBExponential& rhs);
    };

    /** 
     * @brief Surface Brightness Profile for the Airy disk (perfect diffraction-limited PSF for a 
     * circular aperture), with central obscuration.
     *
     * maxK() is set at the hard limit for Airy disks, stepK() makes transforms go to at least 
     * 5 lam/D or EE>(1-alias_threshold).  Schroeder (10.1.18) gives limit of EE at large radius.
     * This stepK could probably be relaxed, it makes overly accurate FFTs.
     * Note x & y are in units of lambda/D here.  Integral over area will give unity in this 
     * normalization.
     */
    class SBAiry : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam_over_D   `lam_over_D` = (lambda * focal length) / (telescope diam) if 
         *                         arg is focal plane position, else `lam_over_D` = 
         *                         lambda / (telescope diam) if arg is in radians of field angle.
         * @param[in] obscuration  linear dimension of central obscuration as fraction of pupil
         *                         dimension (default `obscuration = 0.`).
         * @param[in] flux         flux (default `flux = 1.`).
         */
        SBAiry(double lam_over_D, double obscuration=0., double flux=1.) :
            SBProfile(new SBAiryImpl(lam_over_D, obscuration, flux)) {}

        /// @brief Copy constructor
        SBAiry(const SBAiry& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBAiry() {}

        /// @brief Returns lam_over_D param of the SBAiry.
        double getLamOverD() const 
        {
            assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
            return dynamic_cast<const SBAiryImpl&>(*_pimpl).getLamOverD(); 
        }

        /// @brief Returns obscuration param of the SBAiry.
        double getObscuration() const 
        {
            assert(dynamic_cast<const SBAiryImpl*>(_pimpl.get()));
            return dynamic_cast<const SBAiryImpl&>(*_pimpl).getObscuration(); 
        }

    protected:

        /**
         * @brief Subclass is a scale-free version of the Airy radial function.
         *
         * Serves as interface to numerical photon-shooting class `OneDimensionalDeviate`.
         *
         * Input radius is in units of lambda/D.  Output normalized
         * to integrate to unity over input units.
         */
        class AiryRadialFunction: public FluxDensity 
        {
        public:
            /**
             * @brief Constructor
             * @param[in] obscuration Fractional linear size of central obscuration of pupil.
             */
            AiryRadialFunction(double obscuration, double obssq) : 
                _obscuration(obscuration), _obssq(obssq),
                _norm(M_PI / (4.*(1.-_obssq))) {}

            /**
             * @brief Return the Airy function
             * @param[in] radius Radius in units of (lambda / D)
             * @returns Airy function, normalized to integrate to unity.
             */
            double operator()(double radius) const;

        private:
            double _obscuration; ///< Central obstruction size
            double _obssq; ///< _obscuration*_obscuration
            double _norm; ///< Calculated value M_PI / (4.*(1-obs^2))
        };

    class SBAiryImpl : public SBProfileImpl 
    {
    public:
        SBAiryImpl(double lam_over_D, double obs, double flux);

        ~SBAiryImpl() { flushSampler(); }

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getLamOverD() const { return _lam_over_D; }
        double getObscuration() const { return _obscuration; }

        /**
         * @brief Airy photon-shooting is done numerically with `OneDimensionalDeviate` class.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

    private:
        
        double _lam_over_D;  ///< inverse of _D (see below), harmonise inputs with other GSObjects
        /** 
         * `_D` = (telescope diam) / (lambda * focal length) if arg is focal plane position, 
         *  else `_D` = (telescope diam) / lambda if arg is in radians of field angle.
         */
        double _D;
        double _obscuration; ///< Radius ratio of central obscuration.
        double _flux; ///< Flux.

        double _Dsq; ///< Calculated value: D*D
        double _obssq; ///< Calculated value: _obscuration * _obscuration
        double _norm; ///< Calculated value: flux*D*D

        ///< Class that can sample radial distribution
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler; 

        AiryRadialFunction _radial;  ///< Class that embodies the radial Airy function.

        /// Circle chord length at `h < r`.
        double chord(double r, double h, double rsq, double hsq) const; 

        /// @brief Area inside intersection of 2 circles radii `r` & `s`, seperated by `t`.
        double circle_intersection(double r, double s, double rsq, double ssq, double tsq) const; 
        double circle_intersection(double r, double rsq, double tsq) const; 

        /// @brief Area of two intersecting identical annuli.
        double annuli_intersect(double r1, double r2, double r1sq, double r2sq, double tsq) const; 

        /** 
         * @brief Beam pattern of annular aperture, in k space, which is just the autocorrelation 
         * of two annuli.  Normalized to unity at `k=0` for now.
         */
        double annuli_autocorrelation(double ksq) const; 

        void checkSampler() const; ///< Check if `OneDimensionalDeviate` is configured.
        void flushSampler() const; ///< Discard the photon-shooting sampler class.

        // Copy constructor and op= are undefined.
        SBAiryImpl(const SBAiryImpl& rhs);
        void operator=(const SBAiryImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBAiry& rhs);
    };

    /** 
     * @brief Surface Brightness Profile for the Boxcar function.
     *
     * Convolution with a Boxcar function of dimensions `xw` x `yw` and sampling at pixel centres
     * is equivalent to pixelation (i.e. Surface Brightness integration) across rectangular pixels
     * of the same dimensions.  This class is therefore useful for pixelating SBProfiles.
     */ 
    class SBBox : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] xw    width of Boxcar function along x.
         * @param[in] yw    width of Boxcar function along y.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBBox(double xw, double yw=0., double flux=1.) :
            SBProfile(new SBBoxImpl(xw,yw,flux)) {}

        /// @brief Copy constructor.
        SBBox(const SBBox& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBBox() {}

        /// @brief Returns the x dimension width of the Boxcar.
        double getXWidth() const 
        {
            assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
            return dynamic_cast<const SBBoxImpl&>(*_pimpl).getXWidth(); 
        }

        /// @brief Returns the y dimension width of the Boxcar.
        double getYWidth() const 
        {
            assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
            return dynamic_cast<const SBBoxImpl&>(*_pimpl).getYWidth(); 
        }

    protected:
    class SBBoxImpl : public SBProfileImpl 
    {
    public:
        SBBoxImpl(double xw, double yw, double flux) :
            _xw(xw), _yw(yw), _flux(flux)
        {
            if (_yw==0.) _yw=_xw; 
            _norm = _flux / (_xw * _yw);
        }

        ~SBBoxImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return false; } 
        bool hasHardEdges() const { return true; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -0.5*_xw;  xmax = 0.5*_xw; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -0.5*_yw;  ymax = 0.5*_yw; }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        double getXWidth() const { return _xw; }
        double getYWidth() const { return _yw; }

        /// @brief Boxcar is trivially sampled by drawing 2 uniform deviates.
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        // Override for better efficiency:
        void fillKGrid(KTable& kt) const;
        // Override to put in fractional edge values:
        void fillXGrid(XTable& xt) const;

        template <typename T>
        double fillXImage(ImageView<T>& I, double dx) const;

        double doFillXImage(ImageView<float>& I, double dx) const
        { return fillXImage(I,dx); }
        double doFillXImage(ImageView<double>& I, double dx) const
        { return fillXImage(I,dx); }
        double doFillXImage(ImageView<short>& I, double dx) const
        { return fillXImage(I,dx); }
        double doFillXImage(ImageView<int>& I, double dx) const
        { return fillXImage(I,dx); }

    private:
        double _xw;   ///< Boxcar function is `xw` x `yw` across.
        double _yw;   ///< Boxcar function is `xw` x `yw` across.
        double _flux; ///< Flux.
        double _norm; ///< Calculated value: flux / (xw*yw)

        // Sinc function used to describe Boxcar in k space. 
        double sinc(double u) const; 

        // Copy constructor and op= are undefined.
        SBBoxImpl(const SBBoxImpl& rhs);
        void operator=(const SBBoxImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBBox& rhs);
    };

    /// @brief Class for describing Gauss-Laguerre polynomial Surface Brightness Profiles.
    class SBLaguerre : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] bvec   `bvec[n,n]` contains flux information for the `(n, n)` basis function.
         * @param[in] sigma  scale size of Gauss-Laguerre basis set (default `sigma = 1.`).
         */
        SBLaguerre(LVector bvec=LVector(), double sigma=1.) : 
            SBProfile(new SBLaguerreImpl(bvec,sigma)) {}

        /// @brief Copy Constructor. 
        SBLaguerre(const SBLaguerre& rhs) : SBProfile(rhs) {}

        /// @brief Destructor. 
        ~SBLaguerre() {}

    protected:
    class SBLaguerreImpl : public SBProfileImpl 
    {
    public:
        SBLaguerreImpl(const LVector& bvec, double sigma) : 
            _bvec(bvec.duplicate()), _sigma(sigma) {}

        ~SBLaguerreImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const 
        { throw SBError("SBLaguerre::centroid calculations not yet implemented"); }

        double getFlux() const;

        /// @brief Photon-shooting is not implemented for SBLaguerre, will throw an exception.
        PhotonArray shoot(int N, UniformDeviate& ud) const 
        { throw SBError("SBLaguerre::shoot() is not implemented"); }

    private:
        /// `bvec[n,n]` contains flux information for the `(n, n)` basis function.
        LVector _bvec;  

        double _sigma;  ///< Scale size of Gauss-Laguerre basis set.

        // Copy constructor and op= are undefined.
        SBLaguerreImpl(const SBLaguerreImpl& rhs);
        void operator=(const SBLaguerreImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBLaguerre& rhs);
    };

    /**
     * @brief Surface Brightness for the Moffat Profile (an approximate description of ground-based
     * PSFs).
     */
    class SBMoffat : public SBProfile 
    {
    public:
        enum  RadiusType
        {
            FWHM,
            HALF_LIGHT_RADIUS,
            SCALE_RADIUS
        };

        /** @brief Constructor.
         *
         * @param[in] beta           Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
         * @param[in] size           Size specification.
         * @param[in] rType          Kind of size being specified (one of FWHM, HALF_LIGHT_RADIUS,
         *                           SCALE_RADIUS).
         * @param[in] trunc          Outer truncation radius in same physical units as size,
         *                           trunc = 0. for no truncation (default `trunc = 0.`). 
         * @param[in] flux           Flux (default `flux = 1.`).
         */
        SBMoffat(double beta, double size, RadiusType rType, double trunc=0., 
                 double flux=1.) :
            SBProfile(new SBMoffatImpl(beta, size, rType, trunc, flux)) {}


        /// @brief Copy constructor.
        SBMoffat(const SBMoffat& rhs) : SBProfile(rhs) {}

        /// @brief Destructor.
        ~SBMoffat() {}

        /// @brief Returns beta of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getBeta() const 
        {
            assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
            return dynamic_cast<const SBMoffatImpl&>(*_pimpl).getBeta(); 
        }
        /// @brief Returns the FWHM of the Moffat profile.
        double getFWHM() const 
        {
            assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
            return dynamic_cast<const SBMoffatImpl&>(*_pimpl).getFWHM(); 
        }
        /// @brief Returns the scale radius rD of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getScaleRadius() const 
        {
            assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
            return dynamic_cast<const SBMoffatImpl&>(*_pimpl).getScaleRadius();
        }
        /// @brief Returns the half light radius of the Moffat profile.
        double getHalfLightRadius() const 
        {
            assert(dynamic_cast<const SBMoffatImpl*>(_pimpl.get()));
            return dynamic_cast<const SBMoffatImpl&>(*_pimpl).getHalfLightRadius();
        }

    protected:
    class SBMoffatImpl : public SBProfileImpl 
    {
    public:
        SBMoffatImpl(double beta, double size, RadiusType rType, double trunc, double flux);

        ~SBMoffatImpl() {}

        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const; 

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return (1.-_fluxFactor) > sbp::maxk_threshold; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_maxR; xmax = _maxR; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_maxR; ymax = _maxR; }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& ) const 
        {
            ymax = sqrt(_maxR_sq - x*x);
            ymin = -ymax;
        }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }


        double getFlux() const { return _flux; }

        /**
         * @brief Moffat photon shooting is done by analytic inversion of cumulative flux 
         * distribution.
         *
         * Will require 2 uniform deviates per photon, plus analytic function (pow and sqrt)
         */
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        double getBeta() const { return _beta; }
        double getScaleRadius() const { return _rD; }
        double getFWHM() const { return _FWHM; }
        double getHalfLightRadius() const;

    private:
        double _beta; ///< Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
        double _flux; ///< Flux.
        double _norm; ///< Normalization. (Including the flux)
        double _rD;   ///< Scale radius for profile `[1 + (r / rD)^2]^beta`.
        double _maxR; ///< Maximum `r`
        double _FWHM;  ///< Full Width at Half Maximum.
        double _trunc;  ///< Outer truncation radius in same physical units as `_rD`
        double _fluxFactor; ///< Integral of total flux in terms of 'rD' units.
        double _rD_sq; ///< Calculated value: rD*rD;
        double _maxR_sq; ///< Calculated value: maxR * maxR
        double _maxK; ///< Maximum k with kValue > 1.e-3

        Table<double,double> _ft;  ///< Lookup table for Fourier transform of Moffat.

        mutable double _re; ///< Stores the half light radius if set or calculated post-setting.

        double (*pow_beta)(double x, double beta);

        /// Setup the FT Table.
        void setupFT();

        // Copy constructor and op= are undefined.
        SBMoffatImpl(const SBMoffatImpl& rhs);
        void operator=(const SBMoffatImpl& rhs);
    };

        static double pow_1(double x, double ) { return x; }
        static double pow_2(double x, double ) { return x*x; }
        static double pow_3(double x, double ) { return x*x*x; }
        static double pow_4(double x, double ) { return x*x*x*x; }
        static double pow_int(double x, double beta) { return std::pow(x,int(beta)); }
        static double pow_gen(double x, double beta) { return std::pow(x,beta); }

    private:
        // op= is undefined
        void operator=(const SBMoffat& rhs);
    };

    /// @brief This class is for backwards compatibility; prefer rotate() method.
    class SBRotate : public SBDistort 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] s     SBProfile being rotated.
         * @param[in] theta Rotation angle in radians anticlockwise.
         */
        SBRotate(const SBProfile& s, Angle theta) :
            SBDistort(s, 
                      std::cos(theta.rad()), -std::sin(theta.rad()),
                      std::sin(theta.rad()), std::cos(theta.rad())) {}
    };

    /**
     * @brief Surface Brightness for the de Vaucouleurs Profile, a special case of the Sersic with 
     * `n = 4`.
     */
    class SBDeVaucouleurs : public SBSersic 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] r0    Half-light radius.
         * @param[in] flux  flux (default `flux = 1.`).
         */
      SBDeVaucouleurs(double r0, double flux=1.) : SBSersic(4., r0, flux) {}
    };


}

#endif // SBPROFILE_H

