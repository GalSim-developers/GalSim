
/// \file SBPixel.h contains the class definition for SBPixel objects.

#ifndef SBPIXEL_H
#define SBPIXEL_H

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"

namespace galsim {

    /// Surface Brightness Profile represented by interpolation over one or more data tables/images.
    //
    /// It is assumed that input images oversample the profiles they represent.  maxK() is set at 
    /// the Nyquist frequency of the input image, although it should be noted that interpolants 
    /// other than the ideal sinc function may make the max frequency higher than this.  The output
    /// is required to be periodic on a scale > original image extent + kernel footprint, and 
    /// stepK() is set accordingly.  Multiple images can be stored as data tables in an SBPixel 
    /// object. A vector weight can then be used to
    /// express large families of Surface Brightness Profiles as sums of these interpolated images.
    /// (TODO: Add more!!!)
    class SBPixel : public SBProfile 
    {
    public:
        ///Initialize internal quantities and allocate data tables.
        //
        /// \param Npix Input: extent of square image is `Npix` x `Npix`.
        /// \param dx_ Input: stepsize between pixels in image data table.
        /// \param i Input: interpolation scheme to adopt between pixels (TODO: Add more, document Interpolant.h, describe the Interpolant2d class).
        /// \param Nimages_ Input: number of images.
        SBPixel(int Npix, double dx_, const Interpolant2d& i, int Nimages_=1);

#ifdef USE_IMAGES
        /// Initialize internal quantities and allocate data tables based on a supplied 2D image.
        //
        /// \param img Input: square input Image is.
        /// \param dx_ Input: stepsize between pixels in image data table (default value of `x0_ = 0.` checks the Image header for a suitable stepsize, sets to `1.` if none is found). 
        /// \param i Input: interpolation scheme to adopt between pixels (TODO: Add more, document Interpolant.h, describe the Interpolant2d class).
        /// \param padFactor Input: multiple by which to increase the image size when zero-padding or the Fourier transform (default `padFactor = 0.` forces adoption of the currently-hardwired `OVERSAMPLE_X = 4.` parameter value for `padFactor`).
        SBPixel(Image<float> img, const Interpolant2d& i, double dx_=0., double padFactor=0.);
#endif

        /// Copy Constructor.
        //
        /// \param rhs Input: SBPixel instance to be duplicated.
        SBPixel(const SBPixel& rhs);

        /// Destructor
        ~SBPixel();

        SBProfile* duplicate() const { return new SBPixel(*this); }

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

        double centroidX() const;
        double centroidY() const;

        double getFlux() const;
        void setFlux(double flux=1.);  // This will scale the weights vector

        /////////////////////
        // Methods peculiar to SBPixel

        /// Set the value at one input pixel (without any weights or flux scaling).
        //
        /// Note that input data required to have `-Npix/2 <= ix, iy < Npix/2`
        /// and `0 <= iz < Nimages`.  Note also that setPixel resets the Fourier transform tables.
        /// \param value Input: value at which to set the (ix, iy)-th pixel
        /// \param ix Input: pixel location along x.
        /// \param iy Input: pixel location along y.
        /// \param iz Input: index of image in multiple image table, counting from zero.
        void setPixel(double value, int ix, int iy, int iz=0);

        /// Get the value at one input pixel (without any weights or flux scaling).
        //
        /// Note that input data required to have `-Npix/2 <= ix, iy < Npix/2`
        /// and `0 <= iz < Nimages`.
        /// \param ix Input: pixel location along x.
        /// \param iy Input: pixel location along y.
        /// \param iz Input: index of image in multiple image table, counting from zero.
        double getPixel(int ix, int iy, int iz=0) const;

        /// Set the weight vector applied for summing different planes in a multiple image SBPixel.
        //
        /// \param wts_ Input: weight vector (??? check dimensions = `Nimages` first!)
        void setWeights(const tmv::Vector<double>& wts_); // ??? check dimensions first!

        /// Get the weight vector applied to different planes in the multiple image SBPixel.
        tmv::Vector<double> getWeights() const { return wts; }

        /// Set the interpolant used in real space.
        //
        /// \param interp_ Input: interpolation scheme to adopt between pixels in real space (TODO: Add more, document Interpolant.h, describe the Interpolant2d class).
        void setXInterpolant(const Interpolant2d& interp_) { xInterp=&interp_; ready=false; }

        /// Get the interpolant used in real space.
        const Interpolant2d& getXInterpolant() const { return *xInterp; }

        /// Set the interpolant used in k space.
        //
        /// \param interp_ Input: interpolation scheme to adopt between pixels in real space (TODO: Add more, document Interpolant.h, describe the Interpolant2d class).
        void setKInterpolant(const Interpolant2d& interp_) { kInterp=&interp_; }

        /// Get the interpolant used in k space.
        const Interpolant2d& getKInterpolant() const { return *kInterp; }

        /// Return linear dimension of square input data grid.
        int getNin() const { return Ninitial; }

        /// Return linear dimension of square Discrete Fourier transform used to make k space table.
        int getNft() const { return Nk; }

        // Overrides for better efficiency with separable kernels:
        virtual void fillKGrid(KTable& kt) const;
        virtual void fillXGrid(XTable& xt) const;

#ifdef USE_IMAGES
        virtual double fillXImage(Image<float> I, double dx) const;
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

        /// Vector of weights to use for summing each image plane of a multiple image SBPixel.
        tmv::Vector<double> wts;

        /// Vector of fluxes for each image plane of a multiple image SBPixel.
        mutable tmv::Vector<double> fluxes;

        /// Vector x weighted fluxes for each image plane of a multiple image SBPixel.
        mutable tmv::Vector<double> xFluxes;

        /// Vector of y weighted fluxes for each image plane of a multiple image SBPixel.
        mutable tmv::Vector<double> yFluxes;

        // Arrays summed with weights:
        mutable XTable* xsum; ///< Arrays summed with weights in real space.
        mutable KTable* ksum; ///< Arrays summed with weights in k space.
        mutable bool xsumValid; ///< Is `xsum` valid?
        mutable bool ksumValid; ///< Is `ksum` valid?

        /// Set true if kTables, centroid/flux values,etc., are set for current x pixel values:
        mutable bool ready; 

        /// The default k-space interpolant
        static InterpolantXY defaultKInterpolant2d;

        /// Vector of input data arrays.
        std::vector<XTable*> vx;

        /// Mutable stuff required for kTables and interpolations
        mutable std::vector<KTable*> vk;
        void checkXsum() const;  ///< Used to build xsum if it's not current
        void checkKsum() const;  ///< Used to build ksum if it's not current.
    };

}

#endif // SBPIXEL_H
