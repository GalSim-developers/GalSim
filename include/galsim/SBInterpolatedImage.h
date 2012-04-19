// -*- c++ -*-
#ifndef SBPIXEL_H
#define SBPIXEL_H
/// @file SBInterpolatedImage.h 
/// @brief Contains the class definition for SBInterpolatedImage objects.

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"

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
        *  @param[in] i         interpolation scheme to adopt between pixels 
        *                       (TODO: Add more, document Interpolant.h, describe the Interpolant2d 
        *                       class).
        * @param[in] Nimages_ number of images.
        */
        SBInterpolatedImage(int Npix, double dx_, const Interpolant2d& i, int Nimages_=1);

#ifdef USE_IMAGES
        /** 
         * @brief Initialize internal quantities and allocate data tables based on a supplied 2D 
         * image.
         *
         * @param[in] img       square input Image (not necessarily ImageF).
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
        SBInterpolatedImage(const Image<T> & img, const Interpolant2d& i,
                            double dx_=0., double padFactor=0.);
#endif

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

        double getFlux() const;
        void setFlux(double flux=1.);  // This will scale the weights vector

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

#ifdef USE_IMAGES
        template <typename T>
        virtual double fillXImage(const Image<T> & I, double dx) const;
#endif

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

#endif // SBPIXEL_H
