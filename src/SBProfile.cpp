/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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

#include "SBProfile.h"
#include "SBTransform.h"
#include "SBProfileImpl.h"
#include "math/Angle.h"

// There are three levels of verbosity which can be helpful when debugging,
// which are written as dbg, xdbg, xxdbg (all defined in Std.h).
// It's Mike's way to have debug statements in the code that are really easy to turn
// on and off.
//
// If DEBUGLOGGING is #defined, then these write out to *dbgout, according to the value
// of verbose_level.
// dbg requires verbose_level >= 1
// xdbg requires verbose_level >= 2
// xxdbg requires verbose_level >= 3
//
// If DEBUGLOGGING is not defined, the all three becomes just `if (false) std::cerr`,
// so the compiler parses the statement fine, but trivially optimizes the code away,
// so there is no efficiency hit from leaving them in the code.

namespace galsim {

    SBProfile::SBProfile() {}

    SBProfile::SBProfile(const SBProfile& rhs) : _pimpl(rhs._pimpl) {}

    SBProfile& SBProfile::operator=(const SBProfile& rhs)
    { _pimpl = rhs._pimpl; return *this; }

    SBProfile::~SBProfile()
    {
        // Not strictly necessary, but it sets the ptr to 0, so if somehow someone
        // manages to use an SBProfile after it was deleted, the assert(_pimpl.get())
        // will trigger an exception.
        _pimpl.reset();
    }

    std::string SBProfile::serialize() const
    {
        assert(_pimpl.get());
        return _pimpl->serialize();
    }

    std::string SBProfile::repr() const
    {
        assert(_pimpl.get());
        return _pimpl->repr();
    }

    GSParams SBProfile::getGSParams() const
    {
        assert(_pimpl.get());
        return _pimpl->gsparams;
    }

    double SBProfile::xValue(const Position<double>& p) const
    {
        assert(_pimpl.get());
        return _pimpl->xValue(p);
    }

    std::complex<double> SBProfile::kValue(const Position<double>& k) const
    {
        assert(_pimpl.get());
        return _pimpl->kValue(k);
    }

    void SBProfile::getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getXRange(xmin,xmax,splits);
    }

    void SBProfile::getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getYRange(ymin,ymax,splits);
    }

    void SBProfile::getYRangeX(
        double x, double& ymin, double& ymax, std::vector<double>& splits) const
    {
        assert(_pimpl.get());
        _pimpl->getYRangeX(x,ymin,ymax,splits);
    }

    double SBProfile::maxK() const
    {
        assert(_pimpl.get());
        return _pimpl->maxK();
    }

    double SBProfile::stepK() const
    {
        assert(_pimpl.get());
        return _pimpl->stepK();
    }

    bool SBProfile::isAxisymmetric() const
    {
        assert(_pimpl.get());
        return _pimpl->isAxisymmetric();
    }

    bool SBProfile::hasHardEdges() const
    {
        assert(_pimpl.get());
        return _pimpl->hasHardEdges();
    }

    bool SBProfile::isAnalyticX() const
    {
        assert(_pimpl.get());
        return _pimpl->isAnalyticX();
    }

    bool SBProfile::isAnalyticK() const
    {
        assert(_pimpl.get());
        return _pimpl->isAnalyticK();
    }

    Position<double> SBProfile::centroid() const
    {
        assert(_pimpl.get());
        return _pimpl->centroid();
    }

    double SBProfile::getFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getFlux();
    }

    double SBProfile::maxSB() const
    {
        assert(_pimpl.get());
        return _pimpl->maxSB();
    }

    void SBProfile::shoot(PhotonArray& photons, BaseDeviate rng) const
    {
        assert(_pimpl.get());
        return _pimpl->shoot(photons,rng);
    }

    double SBProfile::getPositiveFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getPositiveFlux();
    }

    double SBProfile::getNegativeFlux() const
    {
        assert(_pimpl.get());
        return _pimpl->getNegativeFlux();
    }

    SBProfile::SBProfile(SBProfileImpl* pimpl) : _pimpl(pimpl) {}

    SBProfile::SBProfileImpl::SBProfileImpl(const GSParams& _gsparams) :
        gsparams(_gsparams) {}

    SBProfile::SBProfileImpl* SBProfile::GetImpl(const SBProfile& rhs)
    { return rhs._pimpl.get(); }

    SBTransform SBProfile::transform(double dudx, double dudy, double dvdx, double dvdy) const
    {
        return SBTransform(*this, dudx, dudy, dvdx, dvdy,
                           Position<double>(0,0), 1., getGSParams());
    }

    //
    // Common methods of Base Class "SBProfile"
    //

    int SBProfile::getGoodImageSize(double dx) const
    {
        dbg<<"Start getGoodImageSize\n";

        // Find a good size based on dx and stepK
        double Nd = 2.*M_PI/(dx*stepK());
        dbg<<"Nd = "<<Nd<<std::endl;

        // Make it an integer
        // Some slop to keep from getting extra pixels due to roundoff errors in calculations.
        int N = int(std::ceil(Nd*(1.-1.e-12)));
        dbg<<"N = "<<N<<std::endl;

        // Round up to an even value
        N = 2*( (N+1)/2);
        dbg<<"N => "<<N<<std::endl;

        return N;
    }

    // Most derived classes override these functions, since there are usually (at least minor)
    // efficiency gains from doing so.  But in some cases, these straightforward impleentations
    // are perfectly fine.
    template <typename T>
    void SBProfile::SBProfileImpl::defaultFillXImage(ImageView<T> im,
                                                     double x0, double dx, int izero,
                                                     double y0, double dy, int jzero) const
    {
        dbg<<"SBProfile fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
            double x = x0;
            for (int i=0; i<m; ++i,x+=dx)
                *ptr++ = xValue(Position<double>(x,y0));
        }
    }

    template <typename T>
    void SBProfile::SBProfileImpl::defaultFillXImage(ImageView<T> im,
                                                     double x0, double dx, double dxy,
                                                     double y0, double dy, double dyx) const
    {
        dbg<<"SBProfile fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = xValue(Position<double>(x,y));
        }
    }

    template <typename T>
    void SBProfile::SBProfileImpl::defaultFillKImage(ImageView<std::complex<T> > im,
                                                     double kx0, double dkx, int izero,
                                                     double ky0, double dky, int jzero) const
    {
        dbg<<"SBProfile fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            for (int i=0; i<m; ++i,kx+=dkx)
                *ptr++ = kValue(Position<double>(kx,ky0));
        }
    }

    template <typename T>
    void SBProfile::SBProfileImpl::defaultFillKImage(ImageView<std::complex<T> > im,
                                                     double kx0, double dkx, double dkxy,
                                                     double ky0, double dky, double dkyx) const
    {
        dbg<<"SBProfile fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = kValue(Position<double>(kx,ky));
        }
    }

    template <typename T>
    void SBProfile::draw(ImageView<T> image, double dx) const
    {
        dbg<<"Start plainDraw"<<std::endl;
        assert(_pimpl.get());
        assert(image.getStep() == 1);

        const int xmin = image.getXMin();
        const int ymin = image.getYMin();
        const int izero = xmin < 0 ? -xmin : 0;
        const int jzero = ymin < 0 ? -ymin : 0;

        _pimpl->fillXImage(image, xmin*dx, dx, izero, ymin*dx, dx, jzero);
        if (dx != 1.) image *= dx*dx;
    }

    template <typename T>
    void SBProfile::drawK(ImageView<std::complex<T> > image, double dk) const
    {
        dbg<<"Start drawK: \n";
        assert(_pimpl.get());
        assert(image.getStep() == 1);

        const int xmin = image.getXMin();
        const int ymin = image.getYMin();
        const int izero = xmin < 0 ? -xmin : 0;
        const int jzero = ymin < 0 ? -ymin : 0;

        _pimpl->fillKImage(image.view(), xmin*dk, dk, izero, ymin*dk, dk, jzero);
    }

    // The type of T (real or complex) determines whether the call-back is to
    // fillXImage or fillKImage.
    template <typename T>
    struct QuadrantHelper
    {
        template <class Prof>
        static void fill(const Prof& prof, ImageView<T> q,
                         double x0, double dx, double y0, double dy)
        { prof.fillXImage(q,x0,dx,0,y0,dy,0); }
    };

    template <typename T>
    struct QuadrantHelper<std::complex<T> >
    {
        template <class Prof>
        static void fill(const Prof& prof, ImageView<std::complex<T> > q,
                         double kx0, double dkx, double ky0, double dky)
        { prof.fillKImage(q,kx0,dkx,0,ky0,dky,0); }
    };

    // The code is basically the same for X or K.
    template <class Prof, typename T>
    static void FillQuadrant(const Prof& prof, ImageView<T> im,
                             double x0, double dx, int m1, double y0, double dy, int n1)
    {
        dbg<<"Start FillQuadrant\n";
        dbg<<x0<<" "<<dx<<" "<<m1<<"   "<<y0<<" "<<dy<<" "<<n1<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        // m1 is the number of columns left of x==0
        // m2 is the number of columns right of x==0
        // n1 is the number of columns below of y==0
        // n2 is the number of columns above of y==0
        // m = m1 + m2 + 1
        // n = n1 + n2 + 1
        const int m2 = m - m1 - 1;
        const int n2 = n - n1 - 1;

        // Make a smaller single-quadrant image and fill that the normal way.
        ImageAlloc<T> q(std::max(m1,m2)+1, std::max(n1,n2)+1);
        QuadrantHelper<T>::fill(prof, q.view(), m1==0?x0:0., dx, n1==0?y0:0., dy);

        // Use those values to fill the original image.
        T* qptr = q.getData() + n1*q.getStride() + m1;
        int qskip = -q.getStride() + (m1-m2-1);
        assert(q.getStep() == 1);
        for (int j=0; j<n1; ++j,ptr+=skip,qptr+=qskip) {
            for (int i=0; i<m1; ++i) *ptr++ = *qptr--;
            for (int i=0; i<=m2; ++i) *ptr++ = *qptr++;
        }
        assert(qptr == q.getData() + m1);
        qskip = q.getStride() + (m1-m2-1);
        for (int j=0; j<=n2; ++j,ptr+=skip,qptr+=qskip) {
            for (int i=0; i<m1; ++i) *ptr++ = *qptr--;
            for (int i=0; i<=m2; ++i) *ptr++ = *qptr++;
        }
        xdbg<<"Done copying quadrants"<<std::endl;
    }
    template <typename T>
    void SBProfile::SBProfileImpl::fillXImageQuadrant(ImageView<T> im,
                                                      double x0, double dx, int nx1,
                                                      double y0, double dy, int ny1) const
    {
        // Guard against infinite loop.
        assert(nx1 != 0 || ny1 != 0);
        FillQuadrant(*this,im,x0,dx,nx1,y0,dy,ny1);
    }
    template <typename T>
    void SBProfile::SBProfileImpl::fillKImageQuadrant(ImageView<std::complex<T> > im,
                                                      double kx0, double dkx, int nkx1,
                                                      double ky0, double dky, int nky1) const
    {
        // Guard against infinite loop.
        assert(nkx1 != 0 || nky1 != 0);
        FillQuadrant(*this,im,kx0,dkx,nkx1,ky0,dky,nky1);
    }

    void GetKValueRange1d(int& i1, int& i2, int m, double kmax, double ksqmax,
                          double kx0, double dkx, double ky, double& kysq)
    {
        if (std::abs(ky) >= kmax) { i1 = i2 = m; return; }
        kysq = ky*ky;
        // Most of the time, there is no region to skip, so only bother with this
        // calculation if either end is large enough.
        double dsq = ksqmax - kysq;
        if (kx0*kx0 > dsq || (kx0+m*dkx)*(kx0+m*dkx) > dsq) {
            // first and last i are where
            //   (kx0 + dkx*i)^2 + kysq = ksqmax
            double d = sqrt(dsq);
            i1 = int(ceil((-kx0 - d) / dkx));
            i2 = int(floor((-kx0 + d) / dkx));
            if (i1 > i2) std::swap(i1,i2);
            ++i2;
            if (i2 <= 0 || i1 >= m) { i1 = i2 = m; return; }
            if (i1 < 0) i1 = 0;
            if (i2 > m) i2 = m;
        } else {
            i1 = 0;
            i2 = m;
        }
    }

    void GetKValueRange2d(int& i1, int& i2, int m, double kmax, double ksqmax,
                          double kx0, double dkx, double ky0, double dky)
    {
#ifdef DEBUGLOGGING
        xdbg<<"GetKValueRange: kx0, ky0 = "<<kx0<<','<<ky0<<"  kmax = "<<kmax<<std::endl;
        xdbg<<"   "<<std::abs(kx0)<<"  "<<std::abs(kx0+m*dkx)<<"   "<<
            std::abs(ky0)<<"  "<<std::abs(ky0+m+dky)<<std::endl;
#endif
        // Most of the time, there is no region to skip, so only bother with this calculation
        // if at least one of the extreme values of kx or ky is > kmax.
        if (std::abs(kx0) > kmax || std::abs(kx0+m*dkx) > kmax ||
            std::abs(ky0) > kmax || std::abs(ky0+m+dky) > kmax) {
            // first and last i are where
            //   (kx0 + i*dkx)^2 + (ky0 + i*dkyx)^2 = ksq_max
            //   (dkx^2 + dky^2) i^2 + 2 (dkx kx0 + dky ky0) i + (kx0^2 + ky0^2 - ksqmax) = 0
            double a = dkx*dkx + dky*dky;
            double b = dkx*kx0 + dky*ky0;
            double c = kx0*kx0 + ky0*ky0 - ksqmax;
            double d = b*b-a*c;
            xdbg<<"d = "<<d<<std::endl;
            if (d <= 0.) { i1 = i2 = m; return; }
            d = sqrt(d);
            i1 = int(ceil((-b - d) / a));
            i2 = int(floor((-b + d) / a));
#ifdef DEBUGLOGGING
            xdbg<<"i1,i2 = "<<i1<<','<<i2<<std::endl;
            double ksq = (kx0+i1*dkx)*(kx0+i1*dkx) + (ky0+i1*dky)*(ky0+i1*dky);
            xdbg<<"k at i1 = "<<sqrt(ksq)<<std::endl;
            assert(ksq <= ksqmax);
            ksq = (kx0+i2*dkx)*(kx0+i2*dkx) + (ky0+i2*dky)*(ky0+i2*dky);
            xdbg<<"k at i2 = "<<sqrt(ksq)<<std::endl;
            assert(ksq <= ksqmax);
            ksq = (kx0+(i1-1)*dkx)*(kx0+(i1-1)*dkx) + (ky0+(i1-1)*dky)*(ky0+(i1-1)*dky);
            xdbg<<"k at i1-1 = "<<sqrt(ksq)<<std::endl;
            assert(ksq > ksqmax);
            ksq = (kx0+(i2+1)*dkx)*(kx0+(i2+1)*dkx) + (ky0+(i2+1)*dky)*(ky0+(i2+1)*dky);
            xdbg<<"k at i2+1 = "<<sqrt(ksq)<<std::endl;
            assert(ksq > ksqmax);
#endif
            if (i1 > i2) std::swap(i1,i2);
            ++i2;
            if (i2 <= 0 || i1 >= m) { i1 = i2 = m; return; }
            if (i1 < 0) i1 = 0;
            if (i2 > m) i2 = m;
#ifdef DEBUGLOGGING
            xdbg<<"i1,i2 => "<<i1<<','<<i2<<std::endl;
            ksq = (kx0+i1*dkx)*(kx0+i1*dkx) + (ky0+i1*dky)*(ky0+i1*dky);
            xdbg<<"k at i1 = "<<sqrt(ksq)<<std::endl;
            ksq = (kx0+i2*dkx)*(kx0+i2*dkx) + (ky0+i2*dky)*(ky0+i2*dky);
            xdbg<<"k at i2 = "<<sqrt(ksq)<<std::endl;
#endif
        } else {
            i1 = 0;
            i2 = m;
        }
    }

    // instantiate template functions for expected image types
    template void SBProfile::draw(ImageView<float> image, double dx) const;
    template void SBProfile::draw(ImageView<double> image, double dx) const;

    template void SBProfile::drawK(ImageView<std::complex<float> > image, double dk) const;
    template void SBProfile::drawK(ImageView<std::complex<double> > image, double dk) const;

    template void SBProfile::SBProfileImpl::defaultFillXImage(
        ImageView<double> im,
        double x0, double dx, int izero, double y0, double dy, int jzero) const;
    template void SBProfile::SBProfileImpl::defaultFillXImage(
        ImageView<float> im,
        double x0, double dx, int izero, double y0, double dy, int jzero) const;
    template void SBProfile::SBProfileImpl::defaultFillXImage(
        ImageView<double> im,
        double x0, double dx, double dxy, double y0, double dy, double dyx) const;
    template void SBProfile::SBProfileImpl::defaultFillXImage(
        ImageView<float> im,
        double x0, double dx, double dxy, double y0, double dy, double dyx) const;
    template void SBProfile::SBProfileImpl::defaultFillKImage(
        ImageView<std::complex<double> > im,
        double kx0, double dkx, int izero, double ky0, double dky, int jzero) const;
    template void SBProfile::SBProfileImpl::defaultFillKImage(
        ImageView<std::complex<float> > im,
        double kx0, double dkx, int izero, double ky0, double dky, int jzero) const;
    template void SBProfile::SBProfileImpl::defaultFillKImage(
        ImageView<std::complex<double> > im,
        double kx0, double dkx, double dkxy, double ky0, double dky, double dkyx) const;
    template void SBProfile::SBProfileImpl::defaultFillKImage(
        ImageView<std::complex<float> > im,
        double kx0, double dkx, double dkxy, double ky0, double dky, double dkyx) const;

    template void SBProfile::SBProfileImpl::fillXImageQuadrant(
        ImageView<double> im,
        double x0, double dx, int nx1, double y0, double dy, int ny1) const;
    template void SBProfile::SBProfileImpl::fillXImageQuadrant(
        ImageView<float> im,
        double x0, double dx, int nx1, double y0, double dy, int ny1) const;
    template void SBProfile::SBProfileImpl::fillKImageQuadrant(
        ImageView<std::complex<double> > im,
        double kx0, double dkx, int nkx1, double ky0, double dky, int nky1) const;
    template void SBProfile::SBProfileImpl::fillKImageQuadrant(
        ImageView<std::complex<float> > im,
        double kx0, double dkx, int nkx1, double ky0, double dky, int nky1) const;
}
