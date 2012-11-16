// -*- c++ -*-

//#define DEBUGLOGGING

#include <complex>
#include "Image.h"
#include "SBInterpolatedImageImpl.h"
#include "SBInterpolatedImage.h"
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
    NoiseCorrFunc::NoiseCorrFunc(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) :
        SBInterpolatedImage(new NoiseCorrFuncImpl(image,xInterp,kInterp,dx,pad_factor)) {}

    NoiseCorrFunc::NoiseCorrFunc(const NoiseCorrFunc& rhs) : SBInterpolatedImage(rhs) {}
  
    NoiseCorrFunc::~NoiseCorrFunc() {}

    template <typename T>
    NoiseCorrFunc::NoiseCorrFuncImpl::NoiseCorrFuncImpl(
        const BaseImage<T>& image, 
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) : 
        SBInterpolatedImageImpl(image, xInterp, kInterp, dx, pad_factor) {}

    //
    template <typename T>
    Image<T> getCovarianceMatrix(ImageView<T> image) const
    {
        int xmin = image.getXMin();
        int ymin = image.getYMin();
        int xmax = image.getXMax();
        int ymax = image.getYMax();
        int xdim = 1 + xmax - xmin;
        int ydim = 1 + ymax - ymin;
        int covdim = xdim * ydim;
        double dx = image.getScale();

        Image<T> cov = Image<T>(covdim, covdim, T(0));
        for (int i=0; i<covdim; i++){

	  for (int j=i; j<covdim; j++){

	    cov.setValue(i, j, );

          }

        }

    }

    // Here we redefine the xValue and kValue (as compared to the SBProfile versions) to enforce
    // two-fold rotational symmetry.

    double NoiseCorrFunc::NoiseCorrFuncImpl::xValue(const Position<double>& p) const 
    {
        if ( p.y >= 0. ) {
            return _xtab->interpolate(p.x, p.y, *_xInterp);
        } else {
            return _xtab->interpolate(-p.x, -p.y, *_xInterp);
        }
    }

    std::complex<double> NoiseCorrFunc::NoiseCorrFuncImpl::kValue(
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

        if ( p.y >= 0. ) {
            return xKernelTransform * _ktab->interpolate(p.x, p.y, *_kInterp);
        } else {
            return xKernelTransform * _ktab->interpolate(-p.x, -p.y, *_kInterp);
        }
    }

    // instantiate template functions for expected image types
    template NoiseCorrFunc::NoiseCorrFunc(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFunc(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFunc(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFunc(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

    template NoiseCorrFunc::NoiseCorrFuncImpl::NoiseCorrFuncImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFuncImpl::NoiseCorrFuncImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFuncImpl::NoiseCorrFuncImpl(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template NoiseCorrFunc::NoiseCorrFuncImpl::NoiseCorrFuncImpl(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

}

