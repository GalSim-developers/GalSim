// -*- c++ -*-
#ifndef SBINTERPOLATED_IMAGE_H
#define SBINTERPOLATED_IMAGE_H
/// @file SBInterpolatedImage.h 
/// @brief Contains the class definition for SBInterpolatedImage objects.

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"
#include "ProbabilityTree.h"

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
         * @param[in] padFactor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `padFactor = 4`)
         */
        template <typename T>
        MultipleImageHelper(const std::vector<boost::shared_ptr<BaseImage<T> > >& images,
                            double dx=0., double padFactor=0.);

        /** 
         * @brief Convenience constructor that only takes a single image.
         *
         * @param[in] image     Single input image
         * @param[in] dx        Stepsize between pixels in image data table (default value of 
         *                      `dx = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] padFactor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `padFactor = 4`)
         */
        template <typename T>
        MultipleImageHelper(const BaseImage<T>& image,
                            double dx=0., double padFactor=0.);

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
     * xInterp = Lanczos(5, fluxConserve=true, tol=kvalue_accuracy)
     * kInterp = Quintic(tol=kvalue_accuracy)
     *
     * There are also optional arguments for the pixel size (default is to get it from
     * the image), and a factor by which to pad the image (default = 4).
     *
     * You can also make an SBInterpolatedImage as a weighted sum of several images
     * using MultipleImageHelper.  This helper object holds the images and their fourier
     * transforms, so it is efficient to make many SBInterpolatedImages with different
     * weight vectors.  This version does not take the `dx` or `padFactor` parameters,
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
         * @param[in] padFactor Multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `padFactor = 4`)
         */
        template <typename T> 
        SBInterpolatedImage(
            const BaseImage<T>& image,
            boost::shared_ptr<Interpolant2d> xInterp = sbp::defaultXInterpolant2d,
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d,
            double dx=0., double padFactor=0.) :
            SBProfile(new SBInterpolatedImageImpl(image,xInterp,kInterp,dx,padFactor)) {}

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
            boost::shared_ptr<Interpolant2d> kInterp = sbp::defaultKInterpolant2d) :
            SBProfile(new SBInterpolatedImageImpl(multi,weights,xInterp,kInterp)) {}

        /// @brief Copy Constructor.
        SBInterpolatedImage(const SBInterpolatedImage& rhs) : SBProfile(rhs) {}

        /// @brief Destructor
        ~SBInterpolatedImage() {}

    protected:

    class SBInterpolatedImageImpl : public SBProfileImpl 
    {
    public:
        template <typename T> 
        SBInterpolatedImageImpl(
            const BaseImage<T>& image, 
            boost::shared_ptr<Interpolant2d> xInterp,
            boost::shared_ptr<Interpolant2d> kInterp,
            double dx, double padFactor);

        SBInterpolatedImageImpl(
            const MultipleImageHelper& multi, const std::vector<double>& weights,
            boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp);

        ~SBInterpolatedImageImpl();

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& p) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_max_size; xmax = _max_size; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_max_size; ymax = _max_size; }

        bool isAxisymmetric() const { return false; }

        // We'll use false here, but really, there's not an easy way to tell.
        // Certainly an Image _could_ have hard edges.
        bool hasHardEdges() const { return false; }

        // This class will be set up so that both x and k domain values
        // are found by interpolation of a table:
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const;

        /**
         *
         * @brief Shoot photons through this object
         *
         * SBInterpolatedImage will assign photons to its input pixels with probability
         * proportional to their flux.  Each photon will then be displaced from its pixel center
         * by an (x,y) amount drawn from the interpolation kernel.  Note that if either the input
         * image or the interpolation kernel have negative regions, then negative-flux photons can
         * be generated.  Noisy images or ring-y kernels will generate a lot of shot noise in
         * the shoot() output.  Not all kernels have photon-shooting implemented.  It may be best to
         * stick to nearest-neighbor and linear interpolation kernels if you wish to avoid these 
         * issues.
         *
         * Use the `Delta` Interpolant if you do not want to waste time moving the photons from 
         * their pixel centers.  But you will regret any attempt to draw images analytically with 
         * that one.
         *
         * Photon shooting with the Sinc kernel is a bad idea and is currently forbidden.
         *
         * @param[in] N Total umber of photons to produce.
         * @param[in] u UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate u) const;

        double getFlux() const;

        double getPositiveFlux() const { checkReadyToShoot(); return _positiveFlux; }
        double getNegativeFlux() const { checkReadyToShoot(); return _negativeFlux; }

        template <typename T>
        double fillXImage(ImageView<T>& I, double dx, double gain) const;

        // Overrides for better efficiency with separable kernels:
        void fillKGrid(KTable& kt) const;
        void fillXGrid(XTable& xt) const;

        // These are the virtual functions, but we don't want to have to duplicate the
        // code implement these.  So each one just calls the template version.  The
        // C++ overloading rules mean that it will call the local fillXImage template 
        // function defined above, not the one in SBProfile (which would lead to an 
        // infinite loop!). 
        //
        // So here is what happens when someone calls fillXImage(I,dx):
        // 1) If they are calling this from an SBInterpolatedImage object, then
        //    it just directly uses the above template version.
        // 2) If they are calling this from an SBProfile object, the template version
        //    there immediately calls doFillXImage for the appropriate type.
        //    That's a virtual function, so if the SBProfile is really an SBInterpolatedImage,
        //    it will find these virtual functions instead of the ones defined in
        //    SBProfile.  Then these functions immediately call the template version
        //    of fillXImage defined above.
        //
        double doFillXImage(ImageView<float>& I, double dx, double gain) const
        { return fillXImage(I,dx,gain); }
        double doFillXImage(ImageView<double>& I, double dx, double gain) const
        { return fillXImage(I,dx,gain); }

    private:

        MultipleImageHelper _multi;
        std::vector<double> _wts;

        boost::shared_ptr<Interpolant2d> _xInterp; ///< Interpolant used in real space.
        boost::shared_ptr<Interpolant2d> _kInterp; ///< Interpolant used in k space.

        boost::shared_ptr<XTable> _xtab; ///< Final padded real-space image.
        mutable boost::shared_ptr<KTable> _ktab; ///< Final k-space image.

        /// @brief Make ktab if necessary.
        void checkK() const;

        double _max_size; ///< Calculated value: Ninitial+2*xInterp->xrange())*dx

        void initialize(); ///< Put code common to both constructors here.

        /// @brief Set true if the data structures for photon-shooting are valid
        mutable bool _readyToShoot;

        /// @brief Set up photon-shooting quantities, if not ready
        void checkReadyToShoot() const;

        // Structures used for photon shooting
        /**
         * @brief Simple structure used to index all pixels for photon shooting
         */
        struct Pixel {
            double x;
            double y;
            bool isPositive;
            double flux;

            Pixel(double x_=0., double y_=0., double flux_=0.): 
                x(x_), y(y_), flux(flux_) { isPositive = flux>=0.; }
            double getFlux() const { return flux; }
        };
        mutable double _positiveFlux;    ///< Sum of all positive pixels' flux
        mutable double _negativeFlux;    ///< Sum of all negative pixels' flux
        mutable ProbabilityTree<Pixel> _pt; ///< Binary tree of pixels, for photon-shooting

        // Copy constructor and op= are undefined.
        SBInterpolatedImageImpl(const SBInterpolatedImageImpl& rhs);
        void operator=(const SBInterpolatedImageImpl& rhs);
    };

    private:
        // op= is undefined
        void operator=(const SBInterpolatedImage& rhs);
    };
}

#endif // SBINTERPOLATED_IMAGE_H
