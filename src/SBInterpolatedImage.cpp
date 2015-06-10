/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

//#define DEBUGLOGGING

#include <algorithm>
#include "SBInterpolatedImage.h"
#include "SBInterpolatedImageImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
std::ostream* dbgout = &std::cout;
int verbose_level = 1;
#endif

namespace galsim {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedImage methods

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant> xInterp, boost::shared_ptr<Interpolant> kInterp,
        double pad_factor, double stepk, double maxk, const GSParamsPtr& gsparams) :
        SBProfile(
            new SBInterpolatedImageImpl(
                image,
                boost::shared_ptr<Interpolant2d>(new InterpolantXY(xInterp)),
                boost::shared_ptr<Interpolant2d>(new InterpolantXY(kInterp)),
                pad_factor, stepk, maxk, gsparams)
        ) {}

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double pad_factor, double stepk, double maxk, const GSParamsPtr& gsparams) :
        SBProfile(
            new SBInterpolatedImageImpl(image,xInterp,kInterp,pad_factor,stepk,maxk,gsparams)
        ) {}

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs) : SBProfile(rhs) {}

    SBInterpolatedImage::~SBInterpolatedImage() {}

    boost::shared_ptr<Interpolant> SBInterpolatedImage::getXInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getXInterp();
    }

    boost::shared_ptr<Interpolant> SBInterpolatedImage::getKInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getKInterp();
    }

    void SBInterpolatedImage::calculateStepK(double max_stepk) const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateStepK(max_stepk);
    }

    void SBInterpolatedImage::calculateMaxK(double max_maxk) const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateMaxK(max_maxk);
    }

    ConstImageView<double> SBInterpolatedImage::getImage() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getImage();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedImageImpl methods

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double pad_factor, double stepk, double maxk, const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _xInterp(xInterp), _kInterp(kInterp), _stepk(stepk), _maxk(maxk),
        _readyToShoot(false)
    {
        dbg<<"image bounds = "<<image.getBounds()<<std::endl;
        dbg<<"pad_factor = "<<pad_factor<<std::endl;
        assert(_xInterp.get());
        assert(_kInterp.get());

        _Ninitial = std::max(image.getXMax()-image.getXMin()+1,
                             image.getYMax()-image.getYMin()+1);
        _init_bounds = image.getBounds();
        dbg<<"Ninitial = "<<_Ninitial<<std::endl;
        assert(pad_factor > 0.);
        _Nk = goodFFTSize(int(pad_factor*_Ninitial));
        dbg<<"_Nk = "<<_Nk<<std::endl;
        double sum = 0.;
        double sumx = 0.;
        double sumy = 0.;

        _xtab = boost::shared_ptr<XTable>(new XTable(_Nk, 1.));
        int xStart = -((image.getXMax()-image.getXMin()+1)/2);
        int y = -((image.getYMax()-image.getYMin()+1)/2);
        dbg<<"xStart = "<<xStart<<", yStart = "<<y<<std::endl;
        for (int iy = image.getYMin(); iy<= image.getYMax(); ++iy, ++y) {
            int x = xStart;
            for (int ix = image.getXMin(); ix<= image.getXMax(); ++ix, ++x) {
                double value = image(ix,iy);
                _xtab->xSet(x, y, value);
                sum += value;
                sumx += value*x;
                sumy += value*y;
                xxdbg<<"ix,iy,x,y = "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
                xxdbg<<"value = "<<value<<", sums = "<<sum<<','<<sumx<<','<<sumy<<std::endl;
            }
        }

        _flux = sum;
        _xcentroid = sumx/sum;
        _ycentroid = sumy/sum;
        dbg<<"flux = "<<_flux<<", xcentroid = "<<_xcentroid<<", ycentroid = "<<_ycentroid<<std::endl;
        dbg<<"N = "<<_Ninitial<<", xrange = "<<_xInterp->xrange()<<std::endl;
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;

        if (_stepk <= 0.) {
            // Calculate stepK:
            //
            // The amount of flux missed in a circle of radius pi/stepk should be at
            // most folding_threshold of the flux.
            //
            // We add the size of the image and the size of the interpolant in quadrature.
            // (Note: Since this isn't a radial profile, R isn't really a radius, but rather
            //        the size of the square box that is enclosing all the flux.)
            double R = _Ninitial/2.;
            // Add xInterp range in quadrature just like convolution:
            double R2 = _xInterp->xrange();
            dbg<<"R(image) = "<<R<<", R(interpolant) = "<<R2<<std::endl;
            R = sqrt(R*R + R2*R2);
            dbg<<"=> R = "<<R<<std::endl;
            _stepk = M_PI / R;
            dbg<<"stepk = "<<_stepk<<std::endl;
        }

        _uscale = 1. / (2.*M_PI);
        _maxk1 = _xInterp->urange()/_uscale;
        if (_maxk <= 0.) {
            // Calculate maxk:
            //
            // For now, just set this to where the interpolant's FT is <= maxk_threshold.
            // Note: since we used kvalue_accuracy for the threshold of _xInterp
            // (at least for the default quintic interpolant) rather than maxk_threshold,
            // this will probably be larger than we really need.
            // We could modify the urange method of Interpolant to take a threshold value
            // at that point, rather than just use the constructor's value, but it's
            // probably not worth it.
            //
            // In practice, we will generally call calculateMaxK() after construction to
            // refine the value of maxk based on the actual FT of the image.
            _maxk = _maxk1;
            dbg<<"maxk = "<<_maxk<<std::endl;
        }

        dbg<<"flux = "<<getFlux()<<std::endl;
    }


    SBInterpolatedImage::SBInterpolatedImageImpl::~SBInterpolatedImageImpl() {}

    boost::shared_ptr<Interpolant> SBInterpolatedImage::SBInterpolatedImageImpl::getXInterp() const
    {
        return static_cast<const InterpolantXY&>(*_xInterp).get1d();
    }

    boost::shared_ptr<Interpolant> SBInterpolatedImage::SBInterpolatedImageImpl::getKInterp() const
    {
        return static_cast<const InterpolantXY&>(*_kInterp).get1d();
    }

    double SBInterpolatedImage::SBInterpolatedImageImpl::xValue(const Position<double>& p) const
    { return _xtab->interpolate(p.x, p.y, *_xInterp); }

    std::complex<double> SBInterpolatedImage::SBInterpolatedImageImpl::kValue(
        const Position<double>& k) const
    {
        // Don't bother if the desired k value is cut off by the x interpolant:
        if (std::abs(k.x) > _maxk1 || std::abs(k.y) > _maxk1) return std::complex<double>(0.,0.);
        checkK();
        double xKernelTransform = _xInterp->uval(k.x*_uscale, k.y*_uscale);
        return xKernelTransform * _ktab->interpolate(k.x, k.y, *_kInterp);
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkK() const
    {
        // Conduct FFT
        if (_ktab.get()) return;
        _ktab = _xtab->transform();
        dbg<<"Built ktab\n";
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXValue(
        tmv::MatrixView<double> val,
        double x0, double dx, int izero,
        double y0, double dy, int jzero) const
    {
        dbg<<"SBInterpolatedImage fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();

        if (dynamic_cast<const InterpolantXY*> (_xInterp.get())) {
            // If the interpolant is separable, the XTable interpolation routine
            // will go faster if we make y iteration the inner loop.
            typedef tmv::VIt<double,tmv::Unknown,tmv::NonConj> RMIt;
            for (int i=0;i<m;++i,x0+=dx) {
                double y = y0;
                RMIt valit = val.row(i).begin();
                for (int j=0;j<n;++j,y+=dy) *valit++ = _xtab->interpolate(x0, y, *_xInterp);
            }
        } else {
            // Otherwise, just do the values in storage order
            typedef tmv::VIt<double,1,tmv::NonConj> CMIt;
            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                CMIt valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) *valit++ = _xtab->interpolate(x, y0, *_xInterp);
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXValue(
        tmv::MatrixView<double> val,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBInterpolatedImage fillXValue\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                *valit++ = _xtab->interpolate(x, y, *_xInterp);
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double kx0, double dkx, int izero,
        double ky0, double dky, int jzero) const
    {
        dbg<<"SBInterpolatedImage fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();
        checkK();

        // Assign zeros for range that has |u| > maxu
        double absdkx = std::abs(dkx);
        double absdky = std::abs(dky);
        int i1 = std::max( int(-_maxk1/absdkx-kx0/dkx) , 0 );
        int i2 = std::min( int(_maxk1/absdkx-kx0/dkx)+1 , m );
        int j1 = std::max( int(-_maxk1/absdky-ky0/dky) , 0 );
        int j2 = std::min( int(_maxk1/absdky-ky0/dky)+1 , n );
        xdbg<<"_maxk1 = "<<_maxk1<<std::endl;
        xdbg<<"i1,i2 = "<<i1<<','<<i2<<std::endl;
        xdbg<<"j1,j2 = "<<j1<<','<<j2<<std::endl;

        val.colRange(0,j1).setZero();
        val.subMatrix(0,i1,j1,j2).setZero();
        val.subMatrix(i2,m,j1,j2).setZero();
        val.colRange(j2,n).setZero();

        kx0 += i1*dkx;
        ky0 += j1*dky;

        // For the rest of the range, calculate ux, uy values
        tmv::Vector<double> ux(i2-i1);
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It uxit = ux.begin();
        double kx = kx0;
        for (int i=i1;i<i2;++i,kx+=dkx) *uxit++ = kx * _uscale;

        tmv::Vector<double> uy(j2-j1);
        It uyit = uy.begin();
        double ky = ky0;
        for (int j=j1;j<j2;++j,ky+=dky) *uyit++ = ky * _uscale;

        const InterpolantXY* kInterpXY = dynamic_cast<const InterpolantXY*>(_kInterp.get());
        if (kInterpXY) {
            // Again, the KTable interpolation routine will go faster if we make y iteration
            // the inner loop.
            typedef tmv::VIt<std::complex<double>,tmv::Unknown,tmv::NonConj> RMIt;

            const InterpolantXY* xInterpXY = dynamic_cast<const InterpolantXY*>(_xInterp.get());
            if (xInterpXY) {
                // Then the uval's are separable.  Go ahead and pre-calculate them.
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,++uxit) *uxit = xInterpXY->uval1d(*uxit);
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,++uyit) *uyit = xInterpXY->uval1d(*uyit);

                uxit = ux.begin();
                for (int i=i1;i<i2;++i,kx0+=dkx,++uxit) {
                    double ky = ky0;
                    uyit = uy.begin();
                    RMIt valit = val.row(i,j1,j2).begin();
                    for (int j=j1;j<j2;++j,ky+=dky) {
                        *valit++ = *uxit * *uyit++ * _ktab->interpolate(kx0, ky, *kInterpXY);
                    }
                }
            } else {
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,kx0+=dkx,++uxit) {
                    double ky = ky0;
                    It uyit = uy.begin();
                    RMIt valit = val.row(i,j1,j2).begin();
                    for (int j=j1;j<j2;++j,ky+=dky) {
                        double xKernelTransform = _xInterp->uval(*uxit, *uyit++);
                        *valit++ = xKernelTransform * _ktab->interpolate(kx0, ky, *kInterpXY);
                    }
                }
            }
        } else {
            typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CMIt;
            const InterpolantXY* xInterpXY = dynamic_cast<const InterpolantXY*>(_xInterp.get());
            if (xInterpXY) {
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,++uxit) *uxit = xInterpXY->uval1d(*uxit);
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,++uyit) *uyit = xInterpXY->uval1d(*uyit);

                uyit = uy.begin();
                for (int j=j1;j<j2;++j,ky0+=dky,++uyit) {
                    double kx = kx0;
                    uxit = ux.begin();
                    CMIt valit = val.col(j,i1,i2).begin();
                    for (int i=i1;i<i2;++i,kx+=dkx) {
                        *valit++ = *uxit++ * *uyit * _ktab->interpolate(kx, ky0, *_kInterp);
                    }
                }
            } else {
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,ky0+=dky,++uyit) {
                    double kx = kx0;
                    It uxit = ux.begin();
                    CMIt valit = val.col(j,i1,i2).begin();
                    for (int i=i1;i<i2;++i,kx+=dkx) {
                        double xKernelTransform = _xInterp->uval(*uxit++, *uyit);
                        *valit++ = xKernelTransform * _ktab->interpolate(kx, ky0, *_kInterp);
                    }
                }
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double kx0, double dkx, double dkxy,
        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBInterpolatedImage fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        checkK();

        double ux0 = kx0 * _uscale;
        double uy0 = ky0 * _uscale;
        double dux = dkx * _uscale;
        double duy = dky * _uscale;
        double duxy = dkxy * _uscale;
        double duyx = dkyx * _uscale;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky,ux0+=duxy,uy0+=duy) {
            double kx = kx0;
            double ky = ky0;
            double ux = ux0;
            double uy = uy0;
            for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx,ux+=dux,uy+=duyx) {
                if (std::abs(kx) > _maxk1 || std::abs(ky) > _maxk1) {
                    *valit++ = 0.;
                } else {
                    double xKernelTransform = _xInterp->uval(ux, uy);
                    *valit++ = xKernelTransform * _ktab->interpolate(kx, ky, *_kInterp);
                }
            }
        }
    }

    std::string SBInterpolatedImage::SBInterpolatedImageImpl::repr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBInterpolatedImage(";

        oss << "galsim._galsim.ConstImageViewD(array([";
        ConstImageView<double> im = getImage();
        int N = _xtab->getN();
        for (int y = 0; y<N; ++y) {
            if (y > 0) oss <<",";
            BaseImage<double>::const_iterator it = im.rowBegin(y);
            oss << "[" << *it++;
            for (; it != im.rowEnd(y); ++it) oss << "," << *it;
            oss << "]";
        }
        oss<<"],dtype=float)), ";

        boost::shared_ptr<Interpolant> xinterp = getXInterp();
        boost::shared_ptr<Interpolant> kinterp = getKInterp();
        oss << "galsim.Interpolant('"<<xinterp->makeStr()<<"', "<<xinterp->getTolerance()<<"), ";
        oss << "galsim.Interpolant('"<<kinterp->makeStr()<<"', "<<kinterp->getTolerance()<<"), ";

        oss << "1., "<<stepK()<<", "<<maxK()<<", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getImage() const
    {
        int N = _xtab->getN();
        return ConstImageView<double>(_xtab->getArray(), boost::shared_ptr<double>(),
                                      N, Bounds<int>(0,N-1,0,N-1));
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::getXRange(
        double& xmin, double& xmax, std::vector<double>& splits) const
    {
        Bounds<int> b = _init_bounds;
        double xrange = _xInterp->xrange();
        int N = b.getXMax()-b.getXMin()+1;
        xmin = -(N/2 + xrange);
        xmax = ((N-1)/2 + xrange);
        int ixrange = _xInterp->ixrange();
        if (ixrange > 0) {
            splits.resize(N-2+ixrange);
            double x = xmin-0.5*(ixrange-2);
            for(int i=0;i<N-2+ixrange;++i, ++x) splits[i] = x;
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::getYRange(
        double& ymin, double& ymax, std::vector<double>& splits) const
    {
        Bounds<int> b = _init_bounds;
        double xrange = _xInterp->xrange();
        int N = b.getXMax()-b.getXMin()+1;
        ymin = -(N/2 + xrange);
        ymax = ((N-1)/2 + xrange);
        int ixrange = _xInterp->ixrange();
        if (ixrange > 0) {
            splits.resize(N-2+ixrange);
            double y = ymin-0.5*(ixrange-2);
            for(int i=0;i<N-2+ixrange;++i, ++y) splits[i] = y;
        }
    }

    Position<double> SBInterpolatedImage::SBInterpolatedImageImpl::centroid() const
    {
        double flux = getFlux();
        if (flux == 0.) throw std::runtime_error("Flux == 0.  Centroid is undefined.");
        return Position<double>(_xcentroid, _ycentroid);
    }

    // We provide an option to update the stepk value by directly calculating what
    // size region around the center encloses (1-folding_threshold) of the total flux.
    // This can be useful if you make the image bigger than you need to, just to be
    // safe, but then want to use as large a stepk value as possible.
    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateStepK(double max_stepk) const
    {
        dbg<<"Start SBInterpolatedImage calculateStepK()\n";
        dbg<<"Current value of stepk = "<<_stepk<<std::endl;
        dbg<<"Find box that encloses "<<1.-this->gsparams->folding_threshold<<" of the flux.\n";
        dbg<<"Max_stepk = "<<max_stepk<<std::endl;
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;
        //int N = _xtab->getN();
        double scale = _xtab->getDx();
        double scalesq = scale*scale;
        double fluxTot = getFlux()/scalesq;
        dbg<<"fluxTot = "<<fluxTot<<std::endl;
        double flux = (*_xtab).xval(0,0);
        double thresh = (1.-this->gsparams->folding_threshold) * fluxTot;
        dbg<<"thresh = "<<thresh<<std::endl;

        // d1 = 0 means that we haven't yet found the d that enclosed enough flux.
        // When we find a flux > thresh, we set d1 = d.
        // However, since the function can have negative regions, we need to keep
        // going to make sure an oscillation doesn't bring us back below thresh.
        // When this happens, we set d1 to 0 again and look for a larger value that
        // enclosed enough flux again.
        int d1 = 0;
        const int Nino2 = _Ninitial/2;
        const Bounds<int> b = _init_bounds;
        int dx = b.getXMin() + ((b.getXMax()-b.getXMin()+1)/2);
        int dy = b.getYMin() + ((b.getYMax()-b.getYMin()+1)/2);
        dbg<<"b = "<<b<<std::endl;
        dbg<<"dx,dy = "<<dx<<','<<dy<<std::endl;
        int min_d = max_stepk == 0. ? 0 : int(ceil(M_PI/max_stepk/scale));
        dbg<<"min_d = "<<min_d<<std::endl;
        double max_flux = flux;
        for (int d=1; d<=Nino2; ++d) {
            xdbg<<"d = "<<d<<std::endl;
            xdbg<<"d1 = "<<d1<<std::endl;
            xdbg<<"flux = "<<flux<<std::endl;
            // Add the left, right, top and bottom sides of box:
            for(int x = -d; x < d; ++x) {
                // Note: All 4 corners are added exactly once by including x=-d but omitting
                // x=d from the loop.
                if (b.includes(Position<int>(x+dx,-d+dy))) flux += _xtab->xval(x,-d);  // bottom
                if (b.includes(Position<int>(d+dx,x+dy))) flux += _xtab->xval(d,x);   // right
                if (b.includes(Position<int>(-x+dx,d+dy))) flux += _xtab->xval(-x,d);  // top
                if (b.includes(Position<int>(-d+dx,-x+dy))) flux += _xtab->xval(-d,-x); // left
            }
            if (flux > max_flux) {
                max_flux = flux;
                if (flux > 1.01 * fluxTot) {
                    // If flux w/in some radius is more than the total, then we have a case of
                    // noise artificially lowering the nominal flux.  We will use the radius
                    // of the maximum flux we get during this procedure.
                    d1 = d;
                }
            }
            if (flux < thresh) {
                d1 = 0; // Mark that we haven't gotten to a good enclosing radius yet.
            } else if (d > min_d) {
                if (d1 == 0) d1 = d; // Mark this radius as a good one.
            }
        }
        dbg<<"Done: flux = "<<flux<<", d1 = "<<d1<<std::endl;
        dbg<<"max_flux = "<<max_flux<<", current fluxTot = "<<fluxTot<<std::endl;
        // Should have added up to the total flux.
        assert( std::abs(flux - fluxTot) < 1.e-3 * std::abs(fluxTot) );

        if (d1 == 0) {
            dbg<<"No smaller radius found.  Keep current value of stepk\n";
            return;
        }
        // (Note: Since this isn't a radial profile, R isn't really a radius, but rather
        //        the size of the square box that is enclosing (1-alias_thresh) of the flux.)
        double R = (d1+0.5) * scale;
        dbg<<"d1 = "<<d1<<" => R = "<<R<<std::endl;
        // Add xInterp range in quadrature just like convolution:
        double R2 = _xInterp->xrange();
        dbg<<"R(image) = "<<R<<", R(interpolant) = "<<R2<<std::endl;
        R = sqrt(R*R + R2*R2);
        dbg<<"=> R = "<<R<<std::endl;
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;
    }

    // The std library norm function uses abs to get a more accurate value.
    // We don't actually care about the slight accuracy gain, so we use a
    // fast norm that just does x^2 + y^2
    inline double fast_norm(const std::complex<double>& z)
    { return real(z)*real(z) + imag(z)*imag(z); }

    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateMaxK(double max_maxk) const
    {
        dbg<<"Start SBInterpolatedImage calculateMaxK()\n";
        dbg<<"Current value of maxk = "<<_maxk<<std::endl;
        dbg<<"max_maxk = "<<max_maxk<<std::endl;
        dbg<<"Find the smallest k such that all values outside of this are less than "
            <<this->gsparams->maxk_threshold<<std::endl;
        checkK();
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;

        double dk = _ktab->getDk();

        // Among the elements with kval > thresh, find the one with the maximum ksq
        double thresh = this->gsparams->maxk_threshold * getFlux();
        thresh *= thresh; // Since values will be |kval|^2.
        double maxk_ix = 0.;
        // When we get 5 rows in a row all below thresh, stop.
        int n_below_thresh = 0;
        int N = _ktab->getN();
        // Don't go past the current value of maxk
        if (max_maxk == 0.) max_maxk = _maxk;
        int max_ix = int(std::ceil(max_maxk / dk));
        if (max_ix > N/2) max_ix = N/2;

        // We take the k value to be maximum of kx and ky.  This is appropriate, because
        // this is how maxK() is eventually used -- it sets the size in k-space for both
        // kx and ky when drawing.  Since kx<0 is just the conjugate of the corresponding
        // point at (-kx,-ky), we only check the right half of the square.  i.e. the
        // upper-right and lower-right quadrants.
        for(int ix=0; ix<=max_ix; ++ix) {
            xdbg<<"Start search for ix = "<<ix<<std::endl;
            // Search along the two sides with either kx = ix or ky = ix.
            for(int iy=0; iy<=ix; ++iy) {
                // The right side of the square in the upper-right quadrant.
                double norm_kval = fast_norm(_ktab->kval2(ix,iy));
                xdbg<<"norm_kval at "<<ix<<','<<iy<<" = "<<norm_kval<<std::endl;
                if (norm_kval <= thresh && iy != ix) {
                    // The top side of the square in the upper-right quadrant.
                    norm_kval = fast_norm(_ktab->kval2(iy,ix));
                    xdbg<<"norm_kval at "<<iy<<','<<ix<<" = "<<norm_kval<<std::endl;
                }
                if (norm_kval <= thresh && iy > 0) {
                    // The right side of the square in the lower-right quadrant.
                    // The ky argument is wrapped to positive values.
                    norm_kval = fast_norm(_ktab->kval2(ix,N-iy));
                    xdbg<<"norm_kval at "<<ix<<','<<-iy<<" = "<<norm_kval<<std::endl;
                }
                if (norm_kval <= thresh && ix > 0) {
                    // The bottom side of the square in the lower-right quadrant.
                    // The ky argument is wrapped to positive values.
                    norm_kval = fast_norm(_ktab->kval2(iy,N-ix));
                    xdbg<<"norm_kval at "<<iy<<','<<-ix<<" = "<<norm_kval<<std::endl;
                }
                if (norm_kval > thresh) {
                    xdbg<<"This one is above thresh\n";
                    // Mark this k value as being aboe the threshold.
                    maxk_ix = ix;
                    // Reset the count to 0
                    n_below_thresh = 0;
                    // Don't bother checking the rest of the pixels with this k value.
                    break;
                }
            }
            xdbg<<"Done ix = "<<ix<<".  Current count = "<<n_below_thresh<<std::endl;
            // If we get through 5 rows with nothing above the threshold, stop looking.
            if (++n_below_thresh == 5) break;
        }
        xdbg<<"Finished.  maxk_ix = "<<maxk_ix<<std::endl;
        // Add 1 to get the first row that is below the threshold.
        ++maxk_ix;
        // Scale by dk
        _maxk = maxk_ix*dk;
        dbg<<"new maxk = "<<_maxk<<std::endl;
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkReadyToShoot() const
    {
        if (_readyToShoot) return;

        dbg<<"SBInterpolatedImage not ready to shoot.  Build _pt:\n";

        // Build the sets holding cumulative fluxes of all Pixels
        _positiveFlux = 0.;
        _negativeFlux = 0.;
        _pt.clear();

        Bounds<int> b = _init_bounds;
        int xStart = -((b.getXMax()-b.getXMin()+1)/2);
        int y = -((b.getYMax()-b.getYMin()+1)/2);

        // We loop over the original bounds, since this is the region over which we
        // always calculate the flux.
        //
        // ix,iy are the indices in the original image
        // x,y are the corresponding indices in _xtab
        // xx,yy are the positions scaled by the pixel scale
        for (int iy = b.getYMin(); iy<= b.getYMax(); ++iy, ++y) {
            int x = xStart;
            double yy = y;
            for (int ix = b.getXMin(); ix<= b.getXMax(); ++ix, ++x) {
                double flux = _xtab->xval(x,y);
                if (flux==0.) continue;
                double xx = x;
                if (flux > 0.) {
                    _positiveFlux += flux;
                } else {
                    _negativeFlux += -flux;
                }
                _pt.push_back(Pixel(xx,yy,flux));
            }
        }
        _pt.buildTree();

        // The above just computes the positive and negative flux for the main image.
        // This is convolved by the interpolant, so we need to correct these values
        // in the same way that SBConvolve does:
        double p1 = _positiveFlux;
        double n1 = _negativeFlux;
        dbg<<"positiveFlux = "<<p1<<", negativeFlux = "<<n1<<std::endl;
        double p2 = _xInterp->getPositiveFlux();
        double n2 = _xInterp->getNegativeFlux();
        dbg<<"Interpolant has positiveFlux = "<<p2<<", negativeFlux = "<<n2<<std::endl;
        _positiveFlux = p1*p2 + n1*n2;
        _negativeFlux = p1*n2 + n1*p2;
        dbg<<"positiveFlux => "<<_positiveFlux<<", negativeFlux => "<<_negativeFlux<<std::endl;

        _readyToShoot = true;
    }

    // Photon-shooting
    boost::shared_ptr<PhotonArray> SBInterpolatedImage::SBInterpolatedImageImpl::shoot(
        int N, UniformDeviate ud) const
    {
        dbg<<"InterpolatedImage shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        assert(N>=0);
        checkReadyToShoot();
        /* The pixel coordinates are stored by cumulative absolute flux in
         * a C++ standard-libary set, so the inversion is done with a binary
         * search tree.  There are no doubt speed gains available from sorting the
         * pixels by flux, and somehow weighting the tree search to the elements holding
         * the most flux.  But I'm doing it the simplest way right now.
         */
        assert(N>=0);

        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        if (N<=0 || _pt.empty()) return result;
        double totalAbsFlux = _positiveFlux + _negativeFlux;
        double fluxPerPhoton = totalAbsFlux / N;
        dbg<<"posFlux = "<<_positiveFlux<<", negFlux = "<<_negativeFlux<<std::endl;
        dbg<<"totFlux = "<<_positiveFlux-_negativeFlux<<", totAbsFlux = "<<totalAbsFlux<<std::endl;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;
        for (int i=0; i<N; ++i) {
            double unitRandom = ud();
            const Pixel* p = _pt.find(unitRandom);
            result->setPhoton(i, p->x, p->y,
                              p->isPositive ? fluxPerPhoton : -fluxPerPhoton);
        }
        dbg<<"result->getTotalFlux = "<<result->getTotalFlux()<<std::endl;

        // Last step is to convolve with the interpolation kernel.
        // Can skip if using a 2d delta function
        const InterpolantXY* xyPtr = dynamic_cast<const InterpolantXY*> (_xInterp.get());
        if ( !(xyPtr && dynamic_cast<const Delta*> (xyPtr->get1d().get()))) {
            boost::shared_ptr<PhotonArray> pa_interp = _xInterp->shoot(N, ud);
            pa_interp->scaleXY(_xtab->getDx());
            result->convolve(*pa_interp, ud);
        }

        dbg<<"InterpolatedImage Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImage methods

    template <typename T>
    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<T>& realImage, const BaseImage<T>& imagImage,
        double dk, double stepk,
        boost::shared_ptr<Interpolant> kInterp,
        const GSParamsPtr& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(
            realImage, imagImage, dk, stepk,
            boost::shared_ptr<Interpolant2d>(new InterpolantXY(kInterp)), gsparams)
        ) {}

    template <typename T>
    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<T>& realImage, const BaseImage<T>& imagImage,
        double dk, double stepk,
        boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(
            realImage, imagImage, dk, stepk, kInterp, gsparams)
        ) {}

    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<double>& data,
        double dk, double stepk, double maxk,
        boost::shared_ptr<Interpolant> kInterp,
        double xcen, double ycen, bool cenIsSet,
        const GSParamsPtr& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(
            data, dk, stepk, maxk,
            boost::shared_ptr<Interpolant2d>(new InterpolantXY(kInterp)),
            xcen, ycen, cenIsSet, gsparams)
        ) {}

    SBInterpolatedKImage::SBInterpolatedKImage(const SBInterpolatedKImage& rhs)
        : SBProfile(rhs) {}

    SBInterpolatedKImage::~SBInterpolatedKImage() {}

    boost::shared_ptr<Interpolant> SBInterpolatedKImage::getKInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).getKInterp();
    }

    ConstImageView<double> SBInterpolatedKImage::getKData() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).getKData();
    }

    double SBInterpolatedKImage::dK() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).dK();
    }

    bool SBInterpolatedKImage::cenIsSet() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).cenIsSet();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImageImpl methods

    // "Normal" constructor
    template <typename T>
    SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<T>& realKImage, const BaseImage<T>& imagKImage,
        double dk, double stepk, boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _kInterp(kInterp), _stepk(stepk), _maxk(0.), _dk(dk), _cenIsSet(false) //fill in maxk below
    {
        dbg<<"stepk = "<<stepk<<std::endl;
        dbg<<"kimage bounds = "<<realKImage.getBounds()<<std::endl;
        assert(_kInterp.get());

        _Ninitial = std::max(realKImage.getXMax()-realKImage.getXMin()+1,
                             realKImage.getYMax()-realKImage.getYMin()+1);
        dbg<<"_Ninitial = "<<_Ninitial<<std::endl;
        _Nk = goodFFTSize(int(_Ninitial));
        dbg<<"_Nk = "<<_Nk<<std::endl;

        _ktab = boost::shared_ptr<KTable>(new KTable(_Nk, _dk));
        _maxk = _Ninitial/2 * _dk;
        dbg<<"_dk = "<<_dk<<std::endl;
        dbg<<"_maxk = "<<_maxk<<std::endl;

        // Only need to fill in x>=0 since the negative x's are the Hermitian
        // conjugates of the positive x's.
        int kxStart = 0;
        int ikxStart = (realKImage.getXMin()+realKImage.getXMax()+1)/2;
        int ky = -((realKImage.getYMax()-realKImage.getYMin()+1)/2);
        dbg<<"kxStart = "<<kxStart<<", kyStart = "<<ky<<std::endl;
        for (int iky = realKImage.getYMin(); iky<= realKImage.getYMax(); ++iky, ++ky) {
             int kx = kxStart;
             for (int ikx = ikxStart; ikx<= realKImage.getXMax(); ++ikx, ++kx) {
                 std::complex<double> kvalue(realKImage(ikx, iky), imagKImage(ikx, iky));
                 _ktab->kSet(kx, ky, kvalue);
                 xxdbg<<"ikx,iky,kx,ky = "<<ikx<<','<<iky<<','<<kx<<','<<ky<<std::endl;
                 xxdbg<<"kvalue = "<<kvalue<<std::endl;
             }
        }
        _flux = kValue(Position<double>(0.,0.)).real();
        dbg<<"flux = "<<_flux<<std::endl;
    }

    // "Serialization" constructor.  Only used when unpickling an InterpolatedKImage.
    // Note *not* a template, since getKData() only returns doubles.
    SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<double>& data, double dk, double stepk, double maxk,
        boost::shared_ptr<Interpolant2d> kInterp,
        double xcen, double ycen, bool cenIsSet,
        const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams),
        _xcentroid(xcen), _ycentroid(ycen),
        _kInterp(kInterp), _stepk(stepk), _maxk(maxk), _dk(dk), _cenIsSet(cenIsSet)
    {
        dbg << "Using alternative constructor" << std::endl;
        _Nk = 2*(data.getYMax() - data.getYMin());
        dbg << "_Nk = " << _Nk << std::endl;
        // Original _Ninitial could have been smaller, but setting it equal to _Nk should be
        // safe nonetheless.
        _Ninitial = _Nk;
        _ktab = boost::shared_ptr<KTable>(new KTable(_Nk, _dk));
        double *kptr = reinterpret_cast<double*>(_ktab->getArray());
        const double* ptr = data.getData();
        for(int i=0; i<2*_Nk*(_Nk/2+1); i++)
            kptr[i] = ptr[i];
        _flux = kValue(Position<double>(0.,0.)).real();
    }

    SBInterpolatedKImage::SBInterpolatedKImageImpl::~SBInterpolatedKImageImpl() {}

    boost::shared_ptr<Interpolant>
    SBInterpolatedKImage::SBInterpolatedKImageImpl::getKInterp() const
    {
        return static_cast<const InterpolantXY&>(*_kInterp).get1d();
    }

    std::complex<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::kValue(
        const Position<double>& k) const
    {
        xdbg<<"evaluating kValue("<<k.x<<","<<k.y<<")"<<std::endl;
        xdbg<<"_maxk = "<<_maxk<<std::endl;
        if (std::abs(k.x) > _maxk || std::abs(k.y) > _maxk) return std::complex<double>(0.,0.);
        return _ktab->interpolate(k.x, k.y, *_kInterp);
    }

    Position<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::centroid() const {
        double flux = getFlux();
        if (flux == 0.) throw std::runtime_error("Flux == 0.  Centroid is undefined.");
        if (!_cenIsSet) {
            /*  int x f(x) dx = (x conv f)|x=0 = int FT(x conv f)(k) dk
                              = int FT(x) FT(f) dk
                FT(x) is divergent, but really we want the first integral above to be
                int(x f(x) dx, -L/2..L/2) since f(x) is formally periodic once it's put on a
                grid.  So in the last integral, we really want FT(x if |x|<L/2 else 0),
                which works out to
                2 i ( kx L cos(kx L/2) - 2 sin(kx L/2)) sin (ky L/2) / kx^2 / ky.
                Noting that kx L/2 = ikx pi, the cosines are -1^ikx and the sines are 0.
                Of course, lim kx->0 sin(kx)/kx is 1 though, so that term survives.  Algebra
                eventually reduces the above expression to what's in the code below.
             */
            double xsum(0.0), ysum(0.0);
            int iky = -_Ninitial/2;
            double sign = (iky % 2 == 0) ? 1.0 : -1.0;
            for (; iky < _Ninitial/2; iky++, sign = -sign) {
                if (iky == 0) continue;
                ysum += sign / iky * _ktab->kval(0, iky).imag();
            }
            int ikx = -_Ninitial/2;
            sign = (ikx % 2 == 0) ? 1.0 : -1.0;
            for (; ikx < _Ninitial/2; ikx++, sign = -sign) {
                if (ikx == 0) continue;
                xsum += sign / ikx * _ktab->kval(ikx, 0).imag();
            }
            _xcentroid = xsum/_dk/flux;
            _ycentroid = ysum/_dk/flux;
            _cenIsSet = true;
        }
        return Position<double>(_xcentroid, _ycentroid);
    }


    ConstImageView<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::getKData() const
    {
        int N = _ktab->getN();
        dbg << "_ktab->getN(): " << N << std::endl;
        double *data = reinterpret_cast<double*>(_ktab->getArray());
        // JM - I'm not completely confident that I got the dimensions below correct,
        //      (i.e., should it be (2N x N/2+1) or (N/2+1 x 2N)?), but it doesn't
        //      actually matter for the intended application, which is just to store and
        //      later retrieve the 2N * N/2+1 memory-contiguous numbers representing the
        //      KTable.
        return ConstImageView<double>(data, boost::shared_ptr<double>(), 2*N,
                                      Bounds<int>(0,2*N-1,0,N/2));
    }

    std::string SBInterpolatedKImage::SBInterpolatedKImageImpl::repr() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBInterpolatedKImage(";

        oss << "galsim._galsim.ConstImageViewD(array([";
        ConstImageView<double> data = getKData();
        for (int y = 0; y<_Nk; ++y) {
            if (y > 0) oss <<",";
            BaseImage<double>::const_iterator it = data.rowBegin(y);
            oss << "[" << *it++;
            for (; it != data.rowEnd(y); ++it) oss << "," << *it;
            oss << "]";
        }
        oss<<"],dtype=float)), ";

        oss << _ktab->getDk() << ", " << stepK() << ", ";
        boost::shared_ptr<Interpolant> kinterp = getKInterp();
        oss << "galsim.Interpolant('"<<kinterp->makeStr()<<"', "<<kinterp->getTolerance()<<"), "
            << "galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    // instantiate template functions for expected image types
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);

    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant> xInterp,
        boost::shared_ptr<Interpolant> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant> xInterp,
        boost::shared_ptr<Interpolant> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);

    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double pad_factor,
        double stepk, double maxk, const GSParamsPtr& gsparams);

    template SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<float>& realImage, const BaseImage<float>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams);
    template SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<double>& realImage, const BaseImage<double>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams);

    template SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<float>& realImage, const BaseImage<float>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant> kInterp,
        const GSParamsPtr& gsparams);
    template SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<double>& realImage, const BaseImage<double>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant> kInterp,
        const GSParamsPtr& gsparams);

    template SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<float>& realImage, const BaseImage<float>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams);
    template SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<double>& realImage, const BaseImage<double>& imagImage,
        double dk, double stepk, boost::shared_ptr<Interpolant2d> kInterp,
        const GSParamsPtr& gsparams);

} // namespace galsim
