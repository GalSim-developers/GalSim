// -*- c++ -*-
#ifndef SBINTERPOLATED_IMAGE_IMPL_H
#define SBINTERPOLATED_IMAGE_IMPL_H

#include "SBProfileImpl.h"
#include "SBInterpolatedImage.h"
#include "ProbabilityTree.h"

namespace galsim {

    class SBInterpolatedImage::SBInterpolatedImageImpl : public SBProfile::SBProfileImpl 
    {
    public:
        template <typename T> 
        SBInterpolatedImageImpl(
            const BaseImage<T>& image, 
            boost::shared_ptr<Interpolant2d> xInterp,
            boost::shared_ptr<Interpolant2d> kInterp,
            double dx, double pad_factor,
            boost::shared_ptr<Image<T> > pad_image);

        SBInterpolatedImageImpl(
            const MultipleImageHelper& multi, const std::vector<double>& weights,
            boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp);

        ~SBInterpolatedImageImpl();

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& p) const;

        double maxK() const { return _maxk; }
        double stepK() const { return _stepk; }

        void calculateMaxK() const;
        void calculateStepK() const;

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

        double getFlux() const { return _flux; }
        double calculateFlux() const;

        double getPositiveFlux() const { checkReadyToShoot(); return _positiveFlux; }
        double getNegativeFlux() const { checkReadyToShoot(); return _negativeFlux; }

        template <typename T>
        double fillXImage(ImageView<T>& I, double gain) const;

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
        double doFillXImage(ImageView<float>& I, double gain) const
        { return fillXImage(I,gain); }
        double doFillXImage(ImageView<double>& I, double gain) const
        { return fillXImage(I,gain); }

    protected:  // Made protected so that these can be used in the derived CorrelationFunction class

        MultipleImageHelper _multi;
        std::vector<double> _wts;

        boost::shared_ptr<Interpolant2d> _xInterp; ///< Interpolant used in real space.
        boost::shared_ptr<Interpolant2d> _kInterp; ///< Interpolant used in k space.

        boost::shared_ptr<XTable> _xtab; ///< Final padded real-space image.
        mutable boost::shared_ptr<KTable> _ktab; ///< Final k-space image.

        /// @brief Make ktab if necessary.
        void checkK() const;

        double _max_size; ///< Calculated value: Ninitial+2*xInterp->xrange())*dx
        mutable double _stepk; ///< Stored value of stepK
        mutable double _maxk; ///< Stored value of maxK
        double _flux;

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

    private:

        // Copy constructor and op= are undefined.
        SBInterpolatedImageImpl(const SBInterpolatedImageImpl& rhs);
        void operator=(const SBInterpolatedImageImpl& rhs);
    };

}

#endif // SBINTERPOLATED_IMAGE_IMPL_H
