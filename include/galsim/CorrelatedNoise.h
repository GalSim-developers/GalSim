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
#include "SBInterpolatedImageImpl.h"
#include "SBInterpolatedImage.h"

namespace galsim {

    /**
     * @brief Class for storing 2D correlation functions represented by interpolation over a data
     * table/image.
     *
     * This class inherits much from SBInterpolatedImage to store the 2D correlation function.
     * The NoiseCorrFunc and SBInterpolatedImage classes represent a profile (supplied as an image
     * an image), including rules for how to interpolate the profile between the supplied pixel
     * values.  Many of the SBProfile methods are, however, disabled.
     *
     * NoiseCorrFunc also imposes two-fold rotational symmetry: any pixels in the negative region 
     * of the input image below the line y = 0 will be ignored.
     *
     * It is assumed that the input image oversamples the correlation function profile they 
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
    class NoiseCorrFunc: public SBInterpolatedImage
    {
    public:
        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image of the Correlation function.
         *
         * @param[in] image     Input Image (any of ImageF, ImageD, ImageS, ImageI).
         * @param[in] xInterp   Interpolation scheme to adopt between pixels 
         * @param[in] kInterp   Interpolation scheme to adopt in k-space
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] pad_factor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `pad_factor = 4`)
         */
        template <typename T> 
        NoiseCorrFunc(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant2d> xInterp = sbp::defaultXInterpolant2d,
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d,
            double dx=0., double pad_factor=0.);

        /// @brief Copy Constructor.
        NoiseCorrFunc(const NoiseCorrFunc& rhs);

        /// @brief Destructor
        ~NoiseCorrFunc();

        ///
        double xValue(const Position<double>& p) const;

    protected:

        class NoiseCorrFuncImpl: public SBInterpolatedImage::SBInterpolatedImageImpl
        {
        public:
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
	    double xValue(const Position<double>& p) const;

            /**
             * @brief Return value of SBProfile at a chosen 2D position in k space.
             *
             * Reflects two-fold rotational symmetry of the correlation function, so that
             *
             *     kValue(k) = kValue(-k)
             *
             * @param[in] k 2D position in k space.
             */
	    std::complex<double> kValue(const Position<double>& p) const;
	}

    private:
        // op= is undefined
        void operator=(const SBInterpolatedImage& rhs);

        // Most of the SBProfile methods are not going to be available eventually...
        
    };
}

