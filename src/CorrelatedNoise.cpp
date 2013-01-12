// -*- c++ -*-

//#define DEBUGLOGGING

#include <complex>
#include "Image.h"
#include "CorrelatedNoiseImpl.h"
#include "CorrelatedNoise.h"

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

namespace galsim {

    template <typename T>
    CorrelationFunction::CorrelationFunction(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) :
        SBInterpolatedImage(new CorrelationFunctionImpl(image,xInterp,kInterp,dx,pad_factor)) {}

    CorrelationFunction::CorrelationFunction(
        const CorrelationFunction& rhs
    ) : SBInterpolatedImage(rhs) {}
  
    CorrelationFunction::~CorrelationFunction() {}

    template <typename T>
    CorrelationFunction::CorrelationFunctionImpl::CorrelationFunctionImpl(
        const BaseImage<T>& image, 
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) :
        SBInterpolatedImageImpl(image, xInterp, kInterp, dx, pad_factor),
        _Ni(1 + image.getXMax() - image.getXMin()), _Nj(1 + image.getYMax() - image.getYMin())
        { initialize(); }

    void CorrelationFunction::CorrelationFunctionImpl::initialize()
    {
        dbg<<"Initializing image with _Ni, _Nj = "<<_Ni<<", "<<_Nj<<std::endl;
        // Perform a check for the oddness of both dimensions of the input lookup table
        if (( _Ni % 2 == 0 ) | ( _Nj % 2 == 0) ) { 
            throw ImageError(
                "Input lookup table is not odd in both dimensions as required"
            );
        }
    }

    // Covariance matrix calculation using the dimensions of an input image, and a scale dx
    template <typename T>
    Image<double> CorrelationFunction::getCovarianceMatrix(ImageView<T> image, double dx) const
    {
        // Calculate the required dimensions of the input image
        int idim = 1 + image.getXMax() - image.getXMin();
        int jdim = 1 + image.getYMax() - image.getYMin();
        int covdim = idim * jdim;
        tmv::SymMatrix<double, 
            tmv::FortranStyle|tmv::Upper> symcov = getCovarianceSymMatrix(image, dx);
        Image<double> cov = Image<double>(covdim, covdim, 0.);

        for (int i=1; i<=covdim; i++){ // note that the Image indices use the FITS convention and 
                                       // start from 1!!
            for (int j=i; j<=covdim; j++){
                cov.setValue(i, j, symcov(i, j)); // fill in the upper triangle with the
                                                  // correct CorrFunc value
            }
        }
        return cov;
    }

    template <typename T>
    tmv::SymMatrix<
        double, tmv::FortranStyle|tmv::Upper
    > CorrelationFunction::getCovarianceSymMatrix(
        ImageView<T> image, double dx) const
    {
         // Calculate the required dimensions
        int idim = 1 + image.getXMax() - image.getXMin();
        int jdim = 1 + image.getYMax() - image.getYMin();
        int covdim = idim * jdim;
        
        if (dx <=0.) dx = image.getScale(); // use the image scale if dx is set less than zero

        int k, ell; // k and l are indices that refer to image pixel separation vectors in the 
                    // correlation func.
        double x_k, y_ell; // physical vector separations in the correlation func, dx * k etc.

        tmv::SymMatrix<double, tmv::FortranStyle|tmv::Upper> cov = tmv::SymMatrix<
            double, tmv::FortranStyle|tmv::Upper>(covdim);

        for (int i=1; i<=covdim; i++){ // note that the Image indices use the FITS convention and 
                                       // start from 1!!
            for (int j=i; j<=covdim; j++){

                k = (j / jdim) - (i / idim);  // using integer division rules here
                ell = (j % jdim) - (i % idim);
                x_k = double(k) * dx;
                y_ell = double(ell) * dx;
                Position<double> p = Position<double>(x_k, y_ell);
                cov(i, j) = _pimpl->xValue(p); // fill in the upper triangle with the
                                               // correct CorrFunc value
            }

        }
        return cov;
    }


    /* Here we redefine the xValue (as compared to the SBProfile version) to enforce two-fold
     * rotational symmetry.
     */
    double CorrelationFunction::xValue(const Position<double> &p) const
    {
        assert(_pimpl.get());
        return _pimpl->xValue(p); 
    }

    double CorrelationFunction::CorrelationFunctionImpl::xValue(const Position<double>& p) const 
    {
        /*
         * Here we do some case switching to access only part of the stored data table, enforcing
         * two-fold rotational symmetry by taking data only from the region y <= 0.
         *
         * TODO: Rewrite comments below with updated methodology
         *
         * There is an additional subtlety for even dimensioned data tables. As discussed in the
         * Pull Request for this addition to GalSim, see
         * https://github.com/GalSim-developers/GalSim/pull/329#discussion-diff-2381280, the shape
         * of the region in the data table that needs to be accessed is slightly non-trivial.  We 
         * use the entire region with y < 0., but also need to access to the upper half of the 
         * left-most column of data: this is not present in the lower right quadrant of the data
         * table, and so we redirect to the upper left if necessary.
         */
        if ( p.y <= 0. ) {
            return _xtab->interpolate(p.x, p.y, *_xInterp);
        } else {
            return _xtab->interpolate(-p.x, -p.y, *_xInterp);                
        }
    }

    std::complex<double> CorrelationFunction::kValue(const Position<double> &k) const
    {
        assert(_pimpl.get());
        return _pimpl->kValue(k); 
    }

    std::complex<double> CorrelationFunction::CorrelationFunctionImpl::kValue(
        const Position<double> &p) const
    {
        const double TWOPI = 2.*M_PI;

        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*_multi.getScale()/TWOPI;
        if (std::abs(ux) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*_multi.getScale()/TWOPI;
        if (std::abs(uy) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = _xInterp->uval(ux, uy);

        checkK();  // this, along with a bunch of other stuff, comes from the SBInterpolatedImage

        /* 
         * the stored table will have been derived from the DFT of an image generated by xValue()
         * calls, and so we need not repeat the detailed case switching used in that function here
         */
        return xKernelTransform * _ktab->interpolate(p.x, p.y, *_kInterp);
    }

    // instantiate template functions for expected image types
    template CorrelationFunction::CorrelationFunction(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunction(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunction(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunction(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

    template CorrelationFunction::CorrelationFunctionImpl::CorrelationFunctionImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunctionImpl::CorrelationFunctionImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunctionImpl::CorrelationFunctionImpl(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template CorrelationFunction::CorrelationFunctionImpl::CorrelationFunctionImpl(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

    template Image<double> CorrelationFunction::getCovarianceMatrix(
        ImageView<float> image, double dx) const;
    template Image<double> CorrelationFunction::getCovarianceMatrix(
        ImageView<double> image, double dx) const;
    template Image<double> CorrelationFunction::getCovarianceMatrix(
        ImageView<int> image, double dx) const;
    template Image<double> CorrelationFunction::getCovarianceMatrix(
        ImageView<short> image, double dx) const;

}

