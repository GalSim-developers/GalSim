// -*- c++ -*-
#ifndef SBINTERPOLATED_IMAGE_H
#define SBINTERPOLATED_IMAGE_H
/** 
 * @file SBInterpolatedImage.h @brief SBProfile that interpolates a given image.
 */

#include "SBProfile.h"
#include "Interpolant.h"
#include "FFT.h"

namespace galsim {

    namespace sbp {

        // Magic numbers:

        /// @brief FT must be at least this much larger than input
        const double oversample_x = 4.;

        /// @brief The default k-space interpolator
        const boost::shared_ptr<Quintic> defaultKInterpolant1d(new Quintic(sbp::kvalue_accuracy));
        const boost::shared_ptr<InterpolantXY> defaultKInterpolant2d(
            new InterpolantXY(defaultKInterpolant1d));

        /// @brief The default real-space interpolator
        const boost::shared_ptr<Lanczos> defaultXInterpolant1d(new Lanczos(5,true,kvalue_accuracy));
        const boost::shared_ptr<InterpolantXY> defaultXInterpolant2d(
            new InterpolantXY(defaultXInterpolant1d));
    }

    /**
     * @brief A Helper class that stores multiple images and their fourier transforms
     *
     * One of the ways to create an SBInterpolatedImage is to build it from a 
     * weighted sum of several component images.  The idea is that the component
     * images would be constant, but the weights might vary across the field of view.
     * (E.g. they could be principal components of the PSF).
     *
     * This class stores those images along with helpful derived information
     * (most notably, the fourier transforms), so that each SBInterpolatedImage
     * doesn't have to recalculate everything from scratch.
     */
    class MultipleImageHelper
    {
    public:
        /** 
         * @brief Construct from a std::vector of images.
         *
         * @param[in] images    List of images to use
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] pad_factor Multiple by which to increase the image size when zero-padding for 
         *                       the Fourier transform (default `pad_factor = 4`)
         * @param[in] pad_image Image to use for padding the SBInterpolatedImage, if `pad_factor` is
         *                      not equal to 1.
         *
         */
        template <typename T>
        MultipleImageHelper(const std::vector<boost::shared_ptr<BaseImage<T> > >& images,
                            double dx=0., double pad_factor=0.,
                            boost::shared_ptr<ImageView<T> > pad_image = Image<T>().view());

        /** 
         * @brief Convenience constructor that only takes a single image.
         *
         * @param[in] image     Single input image
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] pad_factor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `pad_factor = 4`)
         * @param[in] pad_image Image to use for padding the SBInterpolatedImage, if `pad_factor` is
         *                      not equal to 1.
         */
        template <typename T>
        MultipleImageHelper(const BaseImage<T>& image,
                            double dx=0., double pad_factor=0.,
                            boost::shared_ptr<ImageView<T> > pad_image = ImageView<T>().view());

        /// @brief Copies are shallow, so can pass by value without any copying.
        MultipleImageHelper(const MultipleImageHelper& rhs) : _pimpl(rhs._pimpl) {}

        /// @brief Replace the current contents with the contents of rhs.
        MultipleImageHelper& operator=(const MultipleImageHelper& rhs)
        {
            if (this != &rhs) _pimpl = rhs._pimpl;
            return *this;
        }

        ~MultipleImageHelper() {}

        /// @brief How many images are being stored.
        size_t size() const { return _pimpl->vx.size(); }

        /// @brief Get the XTable for the i-th image.
        boost::shared_ptr<XTable> getXTable(int i) const { return _pimpl->vx[i]; }

        /// @brief Get the KTable for the i-th image.
        boost::shared_ptr<KTable> getKTable(int i) const;

        /// @brief Get the flux of the i-th image.
        double getFlux(int i) const { return _pimpl->flux[i]; }

        /// @brief Get the x-weighted flux of the i-th image.
        double getXFlux(int i) const { return _pimpl->xflux[i]; }

        /// @brief Get the y-weighted flux of the i-th image.
        double getYFlux(int i) const { return _pimpl->yflux[i]; }

        /// @brief Get the initial (unpadded) size of the images.
        int getNin() const { return _pimpl->Ninitial; }

        /// @brief Get the size of the images in k-space.
        int getNft() const { return _pimpl->Nk; }

        /// @brief Get the scale size being used for the images.
        double getScale() const { return _pimpl->dx; }

    private:
        // Note: I'm not bothering to make this a real class with setters and getters and all.
        // A struct is good enough for what we need.
        // Just want it to be easy to make shallow copies.
        struct MultipleImageHelperImpl
        {
            int Ninitial; ///< maximum size of input images
            int Nk;  ///< Size of the padded grids and Discrete Fourier transform table.
            double dx;  ///< Input pixel scales.

            /// @brief input images converted into XTables.
            std::vector<boost::shared_ptr<XTable> > vx;

            /// @brief fourier transforms of the images
            std::vector<boost::shared_ptr<KTable> > vk;

            /// @brief Vector of fluxes for each image plane of a multiple image.
            std::vector<double> flux;

            /// @brief Vector x weighted fluxes for each image plane of a multiple image.
            std::vector<double> xflux;

            /// @brief Vector of y weighted fluxes for each image plane of a multiple image.
            std::vector<double> yflux;
        };

        boost::shared_ptr<MultipleImageHelperImpl> _pimpl;
    };

    /** 
     * @brief Surface Brightness Profile represented by interpolation over one or more data 
     * tables/images.
     *
     * The SBInterpolatedImage class represents an arbitrary surface brightness profile (supplied as
     * an image), including rules for how to interpolate the profile between the supplied pixel
     * values.
     *
     * It is assumed that input images oversample the profiles they represent.  maxK() is set at 
     * the Nyquist frequency of the input image, although it should be noted that interpolants 
     * other than the ideal sinc function may make the max frequency higher than this.  The output
     * is required to be periodic on a scale > original image extent + kernel footprint, and 
     * stepK() is set accordingly. 
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
     *
     * You can also make an SBInterpolatedImage as a weighted sum of several images
     * using MultipleImageHelper.  This helper object holds the images and their fourier
     * transforms, so it is efficient to make many SBInterpolatedImages with different
     * weight vectors.  This version does not take the `dx` or `pad_factor` parameters,
     * since these are set in the MultipleImageHelper constructor.
     */
    class SBInterpolatedImage : public SBProfile 
    {
    public:
        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image.
         *
         * @param[in] image     Input Image (any of ImageF, ImageD, ImageS, ImageI).
         * @param[in] xInterp   Interpolation scheme to adopt between pixels 
         * @param[in] kInterp   Interpolation scheme to adopt in k-space
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] pad_factor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `pad_factor = 4`)
         * @param[in] pad_image Image to use for padding the SBInterpolatedImage, if `pad_factor` is
         *                      not equal to 1.
         *
         */
        template <typename T> 
        SBInterpolatedImage(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant2d> xInterp = sbp::defaultXInterpolant2d,
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d,
            double dx=0., double pad_factor=0.,
            boost::shared_ptr<ImageView<T> > pad_image = Image<T>().view());

        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image.
         *
         * @param[in] multi     MultipleImageHelper object which stores the information about
         *                      the component images and their fourier transforms.
         * @param[in] weights   The weights to use for each component image.
         * @param[in] xInterp   Interpolation scheme to adopt between pixels 
         * @param[in] kInterp   Interpolation scheme to adopt in k-space
         */
        SBInterpolatedImage(
            const MultipleImageHelper& multi,
            const std::vector<double>& weights,
            boost::shared_ptr<Interpolant2d> xInterp = sbp::defaultXInterpolant2d,
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d);

        /// @brief Copy Constructor.
        SBInterpolatedImage(const SBInterpolatedImage& rhs);

        /// @brief Destructor
        ~SBInterpolatedImage();

        /// @brief Refine the value of stepK if the input image was larger than necessary.
        void calculateStepK() const;

        /// @brief Refine the value of stepK if the input image had a smaller scale than necessary.
        void calculateMaxK() const;

    protected:

        class SBInterpolatedImageImpl;

        // Regular SBProfile pimpl constructor so as to be available to derived classes
        SBInterpolatedImage(SBProfileImpl* pimpl) : SBProfile(pimpl) {}

    private:
        // op= is undefined
        void operator=(const SBInterpolatedImage& rhs);
    };
}

#endif // SBINTERPOLATED_IMAGE_H
