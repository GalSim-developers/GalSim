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

    /** 
     * @brief Surface Brightness Profile represented by interpolation over one or more data 
     * tables/images.
     *
     * It is assumed that input images oversample the profiles they represent.  maxK() is set at 
     * the Nyquist frequency of the input image, although it should be noted that interpolants 
     * other than the ideal sinc function may make the max frequency higher than this.  The output
     * is required to be periodic on a scale > original image extent + kernel footprint, and 
     * stepK() is set accordingly.  Multiple images can be stored as data tables in an 
     * SBInterpolatedImage object. A vector weight can then be used to express Surface 
     * Brightness Profiles as sums of these interpolated images.
     * (TODO: Add more!!!)
     */
    class SBInterpolatedImage : public SBProfile 
    {
    public:
        /**
         * @brief Initialize internal quantities and allocate data tables.
         *
         * @param[in] Npix      extent of square image is `Npix` x `Npix`.
         * @param[in] dx_       stepsize between pixels in image data table.
         * @param[in] i         interpolation scheme to adopt between pixels 
         *                      (TODO: Add more, document Interpolant.h, describe the Interpolant2d 
         *                      class).
         * @param[in] Nimages_ number of images.
         */
        SBInterpolatedImage(int Npix, double dx_, const Interpolant2d& i, int Nimages_=1);

        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image.
         *
         * @param[in] img       square input Image (any of ImageF, ImageD, ImageS, ImageI).
         * @param[in] dx_       stepsize between pixels in image data table (default value of 
         *                      `x0_ = 0.` checks the Image header for a suitable stepsize, sets 
         *                      to `1.` if none is found). 
         * @param[in] i         interpolation scheme to adopt between pixels (TODO: Add more, 
         *                      document Interpolant.h, describe the Interpolant2d class).
         * @param[in] padFactor multiple by which to increase the image size when zero-padding for 
         *                      the Fourier transform (default `padFactor = 0.` forces adoption of 
         *                      the currently-hardwired `OVERSAMPLE_X = 4.` parameter value for 
         *                      `padFactor`).
         */
        template <typename T> 
        SBInterpolatedImage(const BaseImage<T>& img, const Interpolant2d& i,
                            double dx_=0., double padFactor=0.);

        /** 
         * @brief Copy Constructor.
         *
         * @param[in] rhs SBInterpolatedImage to be copied.
         */
        SBInterpolatedImage(const SBInterpolatedImage& rhs);

        /// @brief Destructor
        ~SBInterpolatedImage();

        SBProfile* duplicate() const { return new SBInterpolatedImage(*this); }

        // These are all the base class members that must be implemented:
        double xValue(Position<double> p) const;

        std::complex<double> kValue(Position<double> p) const;

        // Notice that interpolant other than sinc may make max frequency higher than
        // the Nyquist frequency of the initial image
        double maxK() const { return xInterp->urange() * 2.*M_PI / dx; }

        // Require output FTs to be period on scale > original image extent + kernel footprint:
        double stepK() const { return 2.*M_PI / ( (Ninitial+2*xInterp->xrange())*dx); }

        bool isAxisymmetric() const { return false; }

        // This class will be set up so that both x and k domain values
        // are found by interpolation of a table:
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const;

        /**
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
        virtual PhotonArray shoot(int N, UniformDeviate& u) const;

        double getFlux() const;
        void setFlux(double flux=1.);  // This will scale the weights vector

        double getPositiveFlux() const {checkReadyToShoot(); return positiveFlux;}
        double getNegativeFlux() const {checkReadyToShoot(); return negativeFlux;}

        /////////////////////
        // Methods peculiar to SBInterpolatedImage

        /**
         * @brief Set the value at one input pixel (without any weights or flux scaling).
         *
         * Note that input data required to have `-Npix/2 <= ix, iy < Npix/2` and 
         * `0 <= iz < Nimages`.  Note also that setPixel resets the Fourier transform tables.
         * @param[in] value value to assign to the (ix, iy)-th pixel
         * @param[in] ix    pixel location along x.
         * @param[in] iy    pixel location along y.
         * @param[in] iz    index of image in multiple image table, counting from zero.
         */
        void setPixel(double value, int ix, int iy, int iz=0);

        /** 
         * @brief Get the value at one input pixel (without any weights or flux scaling).
         *
         * Note that input data required to have `-Npix/2 <= ix, iy < Npix/2` and 
         * `0 <= iz < Nimages`.
         * @param[in] ix  pixel location along x.
         * @param[in] iy  pixel location along y.
         * @param[in] iz  index of image in multiple image table, counting from zero.
         */
        double getPixel(int ix, int iy, int iz=0) const;

        /** 
         * @brief Set the weight vector applied for summing different planes in a multiple image 
         * SBInterpolatedImage.
         *
         * @param[in] wts_ weight vector (??? check dimensions = `Nimages` first!)
         */
        void setWeights(const tmv::Vector<double>& wts_); // ??? check dimensions first!

        /// @brief Get the weight vector applied to different planes in the multiple image 
        /// SBInterpolatedImage.
        tmv::Vector<double> getWeights() const { return wts; }

        /** 
         * @brief Set the interpolant used in real space.
         *
         * @param[in] interp_ interpolation scheme to adopt between pixels in real space (TODO: 
         *                    Add more, document Interpolant.h, describe the Interpolant2d class).
         */
        void setXInterpolant(const Interpolant2d& interp_) { xInterp=&interp_; ready=false; }

        /// @brief Get the interpolant used in real space.
        const Interpolant2d& getXInterpolant() const { return *xInterp; }

        /** 
         * @brief Set the interpolant used in k space.
         *
         * @param[in] interp_ interpolation scheme to adopt between pixels in real space (TODO: 
         *                    Add more, document Interpolant.h, describe the Interpolant2d class).
         */
        void setKInterpolant(const Interpolant2d& interp_) { kInterp=&interp_; }

        /// @brief Get the interpolant used in k space.
        const Interpolant2d& getKInterpolant() const { return *kInterp; }

        /// @brief Returns linear dimension of square input data grid.
        int getNin() const { return Ninitial; }

        /** 
         * @brief Returns linear dimension of square Discrete Fourier transform used to make k 
         * space table.
         */
        int getNft() const { return Nk; }

        // Overrides for better efficiency with separable kernels:
        virtual void fillKGrid(KTable& kt) const;
        virtual void fillXGrid(XTable& xt) const;

        template <typename T>
        double fillXImage(ImageView<T>& I, double dx) const;

    protected:
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
        virtual double doFillXImage(ImageView<float>& I, double dx) const
        { return fillXImage(I,dx); }
        virtual double doFillXImage(ImageView<double>& I, double dx) const
        { return fillXImage(I,dx); }

    private:
        void checkReady() const; ///< Make sure all internal quantities are ok.

        int Ninitial; ///< Size of input pixel grids.
        double dx;  ///< Input pixel scales.
        int Nk;  ///< Size of the padded grids and Discrete Fourier transform table.
        double dk;  ///< Step size in k for Discrete Fourier transform table.
        int Nimages; ///< Number of image planes to sum.

        const Interpolant2d* xInterp; ///< Interpolant used in real space.
        const Interpolant2d* kInterp; ///< Interpolant used in k space.

        /// @brief Vector of weights to use for sum over images of a multiple image.
        tmv::Vector<double> wts;

        /// @brief Vector of fluxes for each image plane of a multiple image.
        mutable tmv::Vector<double> fluxes;

        /// @brief Vector x weighted fluxes for each image plane of a multiple image.
        mutable tmv::Vector<double> xFluxes;

        /// @brief Vector of y weighted fluxes for each image plane of a multiple image.
        mutable tmv::Vector<double> yFluxes;

        // Arrays summed with weights:
        mutable XTable* xsum; ///< Arrays summed with weights in real space.
        mutable KTable* ksum; ///< Arrays summed with weights in k space.
        mutable bool xsumValid; ///< Is `xsum` valid?
        mutable bool ksumValid; ///< Is `ksum` valid?

        /** 
         * @brief Set true if kTables, centroid/flux values,etc., are set for current x pixel 
         * values.
         */
        mutable bool ready; 

        /// @brief Set true if the data structures for photon-shooting are valid
        mutable bool readyToShoot;
        /// @brief Set up photon-shooting quantities, if not ready
        void checkReadyToShoot() const;

        // Structures used for photon shooting
        /**
         * @brief Simple structure used to index all pixels for photon shooting
         */
        struct Pixel {
            Pixel(double x_=0., double y_=0., double flux=0.): 
                x(x_), y(y_), _flux(flux) {isPositive = _flux>=0;}
            double x;
            double y;
            double getFlux() const {return _flux;}
            bool isPositive;
        private:
            double _flux;
        };
        mutable double positiveFlux;    ///< Sum of all positive pixels' flux
        mutable double negativeFlux;    ///< Sum of all negative pixels' flux
        mutable ProbabilityTree<Pixel> pt; ///< Binary tree of pixels, for photon-shooting

        /// @brief The default k-space interpolant
        static InterpolantXY defaultKInterpolant2d;

        /// @brief Vector of input data arrays.
        std::vector<XTable*> vx;

        /// @brief Mutable stuff required for kTables and interpolations.
        mutable std::vector<KTable*> vk;
        void checkXsum() const;  ///< Used to build xsum if it's not current.
        void checkKsum() const;  ///< Used to build ksum if it's not current.
    };

}

#endif // SBINTERPOLATED_IMAGE_H
