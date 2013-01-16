// -*- c++ -*-
#ifndef CORRELATEDNOISE_H
#define CORRELATEDNOISE_H

//#define DEBUGLOGGING

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
/*
 * There are three levels of verbosity which can be helpful when debugging, which are written as
 * dbg, xdbg, xxdbg (all defined in Std.h).
 * It's Mike's way to have debug statements in the code that are really easy to turn on and off.
 *
 * If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value of 
 * verbose_level.
 * dbg requires verbose_level >= 1
 * xdbg requires verbose_level >= 2
 * xxdbg requires verbose_level >= 3
 * If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
 * so the compiler parses the statement fine, but trivially optimizes the code away, so there is no
 * efficiency hit from leaving them in the code.
 */
#endif

/**
 * @file CorrelatedNoise.h @brief Contains a class definition for handling the correlation
 * properties of noise in Images.
 */

#include <complex>
#include "TMV_Sym.h"
#include "SBInterpolatedImage.h"
#include "Random.h"
#include "SBAdd.h"

namespace galsim {

    /**
     * @brief Class for storing 2D correlation functions represented by interpolation over a data
     * table/image.
     *
     * This class inherits much (privately) from SBInterpolatedImage to store the 2D correlation 
     * function.  The CorrelationFunction and SBInterpolatedImage classes represent a profile 
     * (supplied as an image), including rules for how to interpolate the profile between the 
     * supplied pixel values.  Many of the SBProfile methods are, however, not enabled.
     *
     * The input data table is assumed to be odd integer-sized in both dimensions (e.g. 31 x 31 or
     * 27 x 45) and centred on the central pixel.
     *
     * CorrelationFunction also imposes two-fold rotational symmetry: all pixels in the positive
     * region of the input image above the line y = 0 will be ignored.  The reason we use the 
     * y-negative part of the input image is due to the way that the SBInterpolated image defines 
     * the center of the distribution.  However, in order for this simple prescription to work
     * requires input data tables with odd numbered dimensions, as discussed here
     * https://github.com/GalSim-developers/GalSim/pull/329#discussion-diff-2381280  
     *
     * The CorrelationFunction therefore checks to see whether the input data table is odd
     * dimensioned, and will throw an ImageError if not.  In GalSim, the CorrelationFunction is used
     * by the CorrFunc class to store correlation functions based on both even and odd sized images:
     * within the `__init__` function this class copies symmetric lookup table data to ensure that
     * all the inputs to the CorrelationFunction are odd integer in each dimension.
     *
     * It is also assumed that the input image oversamples the correlation function profile they 
     * represent.  maxK() is set at the Nyquist frequency of the input image, although it should be
     * noted that interpolants other than the ideal sinc function may make the max frequency higher
     * than this.  The output is required to be periodic on a scale > original image extent + kernel
     * footprint, and stepK() is set accordingly. 
     *
     * The normal way to make an SBInterpolatedImage is to provide the image to interpolate
     * and the interpolation scheme.  See Interpolant.h for more about the different 
     * kind of interpolation.  
     *
     * You can provide different interpolation schemes for real and fourier space
     * (passed as xInterp and kInterp respectively).  If either one is omitted, the 
     * defaults are:
     *
     * xInterp = Lanczos(5, fluxConserve=true, tol=kvalue_accuracy)
     *
     * kInterp = Quintic(tol=kvalue_accuracy)
     *
     * The ideal k-space interpolant is a sinc function; however, the quintic interpolant is the
     * default, based on detailed investigations on the tradeoffs between accuracy and speed.  Note
     * that, as in Bernstein & Gruen (2012), the accuracy achieved by this interpolant is dependent
     * on our choice of 4x pad factor.  Users who do not wish to pad the arrays to this degree may
     * need to use a higher-order Lanczos interpolant instead, but this is not the recommended
     * usage.
     *
     * There are also optional arguments for the pixel size (default is to get it from
     * the image), and a factor by which to pad the image (default = 4).
     */
    class CorrelationFunction : private SBInterpolatedImage
    {
    public:
        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image of the Correlation Function.
         *
         * @param[in] image     Input Image (any of ImageF, ImageD, ImageS, ImageI), must be odd
         *                      integer in size along each dimension, and be centred on the central
         *                      pixel.
         * @param[in] xInterp   Interpolation scheme to adopt between pixels 
         * @param[in] kInterp   Interpolation scheme to adopt in k-space
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] pad_factor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `pad_factor = 4`)
         */
        template <typename T>
        CorrelationFunction(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant2d> xInterp = sbp::defaultXInterpolant2d,
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d,
           double dx=0., double pad_factor=0.);

        /// @brief Copy Constructor.
        CorrelationFunction(const CorrelationFunction& rhs);

        /// @brief Destructor
        ~CorrelationFunction();

        // CorrelationFunction privately inherits, so state methods we need from the base class
        //using SBInterpolatedImage::draw;
        //using SBInterpolatedImage::drawShoot;
        //using SBInterpolatedImage::drawK;
        //using SBInterpolatedImage::applyTransformation;
        //using SBInterpolatedImage::applyShear;
        //using SBInterpolatedImage::applyRotation;

        /** 
         * @brief Return value of correlation function at a chosen 2D position in real space.
         *
         * Reflects two-fold rotational symmetry of the correlation function, so that
         *
         *     xValue(p) = xValue(-p)
         *
         * Assume all are real-valued.  xValue() may not be implemented for derived classes 
         * (SBConvolve) that require an FFT to determine real-space values.  In this case, an 
         * SBError will be thrown.
         *
         * @param[in] p 2D position in real space.
         */
        double xValue(const Position<double>& p) const
        {
            return SBInterpolatedImage::xValue(p);
        }

        /**
         * @brief Return value of correlation function at a chosen 2D position in k space.
         *
         * Reflects two-fold rotational symmetry of the correlation function, so that
         *
         *     kValue(k) = kValue(-k)
         *
         * @param[in] k 2D position in k space.
         */
        std::complex<double> kValue(const Position<double>& k) const
        {
            return SBInterpolatedImage::kValue(k);
        }

        // Doxygen automatically takes relevant documentation from include/galsim/SBProfile.h
        void applyTransformation(const CppEllipse& e)
        {
            return SBInterpolatedImage::applyTransformation(e);
        }

        // Doxygen automatically takes relevant documentation from include/galsim/SBProfile.h
        void applyShear(CppShear s)
        {
            return SBInterpolatedImage::applyShear(s);
        }

        // Doxygen automatically takes relevant documentation from include/galsim/SBProfile.h
        void applyRotation(const Angle& theta)
        {
            return SBInterpolatedImage::applyRotation(theta);
        }

        /**
         * @brief Multiply the overall variance of the correlation function by varianceRatio.
         *
         * This resets the internal pointer to a new CorrelationFunction that wraps the old one
         * with a scaled flux.  This does not change any previous uses (also see documentation to
         * the SBProfile.scaleFlux() member function which this function reimplements).
         */
        void scaleVariance(double varianceRatio)
        {
            return SBInterpolatedImage::scaleFlux(varianceRatio);
        }

        /**
         * @brief Return, as a square Image, a noise covariance matrix between every element in an 
         * input Image with pixel scale dx.
         *
         * The matrix is symmetric, and therefore only the upper triangular elements are actually
         * written into.  The rest are initialized and remain as zero.
         *
         * Currently, this actually copies elements from an internal calculation of the covariance
         * matrix (using Mike Jarvis' TMV library).  It could, therefore, be calculated more 
         * efficiently by direct assignment.  However, as this public member function is foreseen
         * as being mainly for visualization/checking purposes, we go via the TMV intermediary to 
         * avoid code duplication.  If, in future, it becomes critical to speed up this function
         * this can be revisited.
         */
        template <typename T>
        Image<double> calculateCovarianceMatrix(ImageView<T> image, double dx) const;

        // Doxygen would automatically takes documentation from include/galsim/SBProfile.h for the
        // draw methods below, but we don't want that as they reference SBProfiles...

        /**
         * Draw the CorrelationFunction in real space returning the total sum.
         * 
         * See documentation for the SBProfile::draw() member function for more information.
         */
        template <typename T>
        double draw(ImageView<T> image, double gain=1., double wmult=1.) const
        {
            return SBInterpolatedImage::draw(image, gain, wmult);
        }

        /**
         * @brief Draw an image of the CorrelationFunction in k space.
         *
         * See documentation for the SBProfile::drawK() member function for more information.
         */
        template <typename T>
        void drawK(ImageView<T> re, ImageView<T> im, double gain=1., double wmult=1.) const
        {
            return SBInterpolatedImage::drawK(re, im, gain, wmult);
        }

        /** 
         * @brief Draw this CorrelationFunction into an Image by shooting photons.
         * 
         * See documentation for the SBProfile::drawShoot() member function for more information.
         */
        template <typename T>
        double drawShoot(ImageView<T> image, double N, UniformDeviate ud,
                         double gain=1., double max_extra_noise=0., bool poisson_flux=true) const
        {
            return SBInterpolatedImage::drawShoot(
                image, N, ud, gain, max_extra_noise, poisson_flux
            );
        }

    protected:

        class CorrelationFunctionImpl;

         /**
         * @brief Return, as a TMV SymMatrix, a noise covariance matrix between every element in 
         * an input Image with pixel scale dx.
         *
         * The TMV SymMatrix uses FortranStyle indexing (to match the FITS-compliant usage in Image)
         * along with ColumnMajor ordering (the default), and Upper triangle storage.
         */
        template <typename T>
        tmv::SymMatrix<double, tmv::FortranStyle|tmv::Upper> calculateCovarianceSymMatrix(
            ImageView<T> image, double dx) const;

    };

    /**
     * @brief Sums CorrelationFunctions.
     *
     * The AddCorrelationFunction class can be used to add arbitrary numbers of CorrelationFunctions
     * together.  It borrows heavily from the SBAdd class.
     */
    class AddCorrelationFunction : private SBAdd, public CorrelationFunction
    {
    public:

        /** 
         * @brief Constructor, 2 inputs.
         *
         * @param[in] c1 first CorrelationFunction.
         * @param[in] c2 second CorrelationFunction.
         */
        AddCorrelationFunction(const CorrelationFunction& c1, const CorrelationFunction& c2);

        /** 
         * @brief Constructor, list of inputs.
         *
         * @param[in] clist list of CorrelationFunctions.
         */
        AddCorrelationFunction(const std::list<CorrelationFunction>& clist);

        /// @brief Copy constructor.
        AddCorrelationFunction(const AddCorrelationFunction& rhs);

        /// @brief Destructor.
        ~AddCorrelationFunction();

    protected:

        class AddCorrelationFunctionImpl;

    private:
        // op= is undefined
        void operator=(const AddCorrelationFunction& rhs);
    };

}
#endif
