/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#include "SBBox.h"
#include "SBBoxImpl.h"
#include "math/Sinc.h"
#include "math/Angle.h"
#include "math/Bessel.h"

// cf. comments about USE_COS_SIN in SBGaussian.cpp
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

namespace galsim {


    SBBox::SBBox(double width, double height, double flux, const GSParams& gsparams) :
        SBProfile(new SBBoxImpl(width,height,flux,gsparams)) {}

    SBBox::SBBox(const SBBox& rhs) : SBProfile(rhs) {}

    SBBox::~SBBox() {}

    double SBBox::getWidth() const
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return static_cast<const SBBoxImpl&>(*_pimpl).getWidth();
    }

    double SBBox::getHeight() const
    {
        assert(dynamic_cast<const SBBoxImpl*>(_pimpl.get()));
        return static_cast<const SBBoxImpl&>(*_pimpl).getHeight();
    }

    SBBox::SBBoxImpl::SBBoxImpl(double width, double height, double flux,
                                const GSParams& gsparams) :
        SBProfileImpl(gsparams), _width(width), _height(height), _flux(flux)
    {
        if (_height==0.) _height=_width;
        _norm = _flux / (_width * _height);
        _wo2 = 0.5*_width;
        _ho2 = 0.5*_height;
        _wo2pi = _width/(2.*M_PI);
        _ho2pi = _height/(2.*M_PI);
    }


    double SBBox::SBBoxImpl::xValue(const Position<double>& p) const
    {
        if (fabs(p.x) < _wo2 && fabs(p.y) < _ho2) return _norm;
        else return 0.;  // do not use this function for filling image!
    }

    std::complex<double> SBBox::SBBoxImpl::kValue(const Position<double>& k) const
    {
        return _flux * math::sinc(k.x*_wo2pi)*math::sinc(k.y*_ho2pi);
    }

    template <typename T>
    void SBBox::SBBoxImpl::fillXImage(ImageView<T> im,
                                      double x0, double dx, int izero,
                                      double y0, double dy, int jzero) const
    {
        dbg<<"SBBox fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;

        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 /= dx;
        double wo2 = _wo2 / std::abs(dx);
        y0 /= dy;
        double ho2 = _ho2 / std::abs(dy);

        // Fill the interior with _norm:
        // Fill pixels where:
        //     x0 + i >= -width/2
        //     x0 + i < width/2
        //     y0 + j >= -width/2
        //     y0 + j < width/2

        int i1 = std::max(0, int(std::ceil(-wo2 - x0)));
        int i2 = std::min(m, int(std::ceil(wo2 - x0)));
        int j1 = std::max(0, int(std::ceil(-ho2 - y0)));
        int j2 = std::min(n, int(std::ceil(ho2 - y0)));

        ptr += im.getStride() * j1 + i1;
        skip += m - (i2-i1);

        im.setZero();
        for (int j=j1; j<j2; ++j,ptr+=skip) {
            for (int i=i1;i<i2;++i)
                *ptr++ = _norm;
        }
    }

    template <typename T>
    void SBBox::SBBoxImpl::fillXImage(ImageView<T> im,
                                      double x0, double dx, double dxy,
                                      double y0, double dy, double dyx) const
    {
        dbg<<"SBBox fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;

        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (int j=0;j<n;++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            int i=0;
            // Use the fact that any slice through the box has only one segment that is non-zero.
            // So start with zeroes until in the box (already there), then _norm, then more zeroes.
            for (;i<m && (std::abs(x)>_wo2 || std::abs(y)>_ho2); ++i,x+=dx,y+=dyx)
                *ptr++ = T(0);
            for (;i<m && std::abs(x)<_wo2 && std::abs(y)<_ho2; ++i,x+=dx,y+=dyx)
                *ptr++ = _norm;
            for (;i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = T(0);
        }
    }

    template <typename T>
    void SBBox::SBBoxImpl::fillKImage(ImageView<std::complex<T> > im,
                                      double kx0, double dkx, int izero,
                                      double ky0, double dky, int jzero) const
    {
        dbg<<"SBBox fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKImageQuadrant(im,kx0,dkx,izero,ky0,dky,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            std::complex<T>* ptr = im.getData();
            int skip = im.getNSkip();
            assert(im.getStep() == 1);

            kx0 *= _wo2pi;
            dkx *= _wo2pi;
            ky0 *= _ho2pi;
            dky *= _ho2pi;

            // The Box profile in Fourier space is separable:
            //    val(x,y) = _flux * sinc(x * _width/2pi) * sinc(y * _height/2pi)
            std::vector<double> sinc_kx(m);
            std::vector<double> sinc_ky(n);
            typedef std::vector<double>::iterator It;
            It kxit = sinc_kx.begin();
            for (int i=0; i<m; ++i,kx0+=dkx) *kxit++ = math::sinc(kx0);

            if ((kx0 == ky0) && (dkx == dky) && (m==n)) {
                sinc_ky = sinc_kx;
            } else {
                It kyit = sinc_ky.begin();
                for (int j=0; j<n; ++j,ky0+=dky) *kyit++ = math::sinc(ky0);
            }

            for (int j=0; j<n; ++j,ptr+=skip) {
                for (int i=0; i<m; ++i)
                    *ptr++ = _flux * sinc_kx[i] * sinc_ky[j];
            }
        }
    }

    template <typename T>
    void SBBox::SBBoxImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBBox fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _wo2pi;
        dkx *= _wo2pi;
        dkxy *= _wo2pi;
        ky0 *= _ho2pi;
        dky *= _ho2pi;
        dkyx *= _ho2pi;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx) {
                *ptr++ = _flux * math::sinc(kx) * math::sinc(ky);
            }
        }
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBBox::SBBoxImpl::maxK() const
    {
        return 2. / (this->gsparams.maxk_threshold * std::min(_width,_height));
    }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBBox::SBBoxImpl::stepK() const
    {
        // In this case max(width,height) encloses all the flux, so use that.
        return M_PI / std::max(_width,_height);
    }

    void SBBox::SBBoxImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Box shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++)
            photons.setPhoton(i, _width*(ud()-0.5), _height*(ud()-0.5), fluxPerPhoton);
        dbg<<"Box Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }



    SBTopHat::SBTopHat(double radius, double flux, const GSParams& gsparams) :
        SBProfile(new SBTopHatImpl(radius,flux,gsparams)) {}

    SBTopHat::SBTopHat(const SBTopHat& rhs) : SBProfile(rhs) {}

    SBTopHat::~SBTopHat() {}

    double SBTopHat::getRadius() const
    {
        assert(dynamic_cast<const SBTopHatImpl*>(_pimpl.get()));
        return static_cast<const SBTopHatImpl&>(*_pimpl).getRadius();
    }

    SBTopHat::SBTopHatImpl::SBTopHatImpl(double radius, double flux,
                                         const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _r0(radius), _r0sq(_r0*_r0), _flux(flux),
        _norm(_flux / (M_PI * _r0sq))
    {
    }


    double SBTopHat::SBTopHatImpl::xValue(const Position<double>& p) const
    {
        double rsq = p.x*p.x + p.y*p.y;
        if (rsq < _r0sq) return _norm;
        else return 0.;
    }

    std::complex<double> SBTopHat::SBTopHatImpl::kValue(const Position<double>& k) const
    {
        double kr0sq = (k.x*k.x + k.y*k.y) * _r0sq;
        return kValue2(kr0sq);
    }

    std::complex<double> SBTopHat::SBTopHatImpl::kValue2(double kr0sq) const
    {
        if (kr0sq < 1.e-4) {
            // Use the Taylor expansion for small arguments.
            // Error from omitting next term is about 1.e-16 for kr0sq = 1.e-4
            return _flux * (1. - kr0sq * ( (1./8.) + (1./192.) * kr0sq ));
        } else {
            double kr0 = sqrt(kr0sq);
            return 2.*_flux * math::j1(kr0)/kr0;
        }
    }

    template <typename T>
    void SBTopHat::SBTopHatImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, int izero,
                                            double y0, double dy, int jzero) const
    {
        dbg<<"SBTopHat fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        // The columns to consider have -r0 <= y < r0
        // given that y = y0 + j dy
        double absdx = std::abs(dx);
        double absdy = std::abs(dy);
        int j1 = std::max(0, int(std::ceil(-_r0/absdy - y0/dy)));
        int j2 = std::min(n, int(std::ceil(_r0/absdy - y0/dy)));
        y0 += j1 * dy;
        ptr += j1*im.getStride();

        im.setZero();
        for (int j=j1; j<j2; ++j,y0+=dy,ptr+=skip) {
            double ysq = y0*y0;
            double xmax = std::sqrt(_r0sq - ysq);
            // Set to _norm all pixels with -xmax <= x < xmax
            // given that x = x0 + i dx.
            int i1 = std::max(0, int(std::ceil(-xmax/absdx - x0/dx)));
            int i2 = std::min(m, int(std::ceil(xmax/absdx - x0/dx)));
            int i=0;
            for (; i<i1; ++i) ++ptr;
            for (; i<i2; ++i) *ptr++ = _norm;
            for (; i<m; ++i) ++ptr;
        }
    }

    template <typename T>
    void SBTopHat::SBTopHatImpl::fillXImage(ImageView<T> im,
                                            double x0, double dx, double dxy,
                                            double y0, double dy, double dyx) const
    {
        dbg<<"SBTopHat fillXImage\n";
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
            int i=0;
            // Use the fact that any slice through the circle has only one segment that is non-zero.
            // So start with zeroes until in the circle, then _norm, then more zeroes.
            // Note: this could be sped up somewhat using the same kind of calculation we did
            // for the non-sheared fillXImage (the one with izero, jzero), but I didn't
            // bother.  This is probably plenty fast enough for as often as the function is
            // called (i.e. almost never!)
            for (;i<m && (x*x+y*y > _r0sq); ++i,x+=dx,y+=dyx) *ptr++ = T(0);
            for (;i<m && (x*x+y*y < _r0sq); ++i,x+=dx,y+=dyx) *ptr++ = _norm;
            for (;i<m; ++i,x+=dx,y+=dyx) *ptr++ = T(0);
        }
    }

    template <typename T>
    void SBTopHat::SBTopHatImpl::fillKImage(ImageView<std::complex<T> > im,
                                            double kx0, double dkx, int izero,
                                            double ky0, double dky, int jzero) const
    {
        dbg<<"SBTopHat fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillKImageQuadrant(im,kx0,dkx,izero,ky0,dky,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            std::complex<T>* ptr = im.getData();
            int skip = im.getNSkip();
            assert(im.getStep() == 1);

            kx0 *= _r0;
            dkx *= _r0;
            ky0 *= _r0;
            dky *= _r0;

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                double kx = kx0;
                double kysq = ky0*ky0;
                for (int i=0; i<m; ++i,kx+=dkx)
                    *ptr++ = kValue2(kx*kx + kysq);
            }
        }
    }

    template <typename T>
    void SBTopHat::SBTopHatImpl::fillKImage(ImageView<std::complex<T> > im,
                                            double kx0, double dkx, double dkxy,
                                            double ky0, double dky, double dkyx) const
    {
        dbg<<"SBTopHat fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        assert(im.getStep() == 1);

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0;
        dky *= _r0;
        dkyx *= _r0;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx)
                *ptr++ = kValue2(kx*kx + ky*ky);
        }
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBTopHat::SBTopHatImpl::maxK() const
    {
        // |j1(x)| ~ sqrt(2/(Pi x)) for large x, so using this, we get
        // maxk_thresh = 2 * sqrt(2/(Pi k r0)) / (k r0) = 2 sqrt(2/Pi) (k r0)^-3/2
        return std::pow(2. * sqrt(2./M_PI) / this->gsparams.maxk_threshold, 2./3.) / _r0;
    }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBTopHat::SBTopHatImpl::stepK() const
    {
        // _r0 encloses all the flux, so use that.
        return M_PI / _r0;
    }

    void SBTopHat::SBTopHatImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"TopHat shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        double fluxPerPhoton = _flux/N;
        // cf. SBGaussian's shoot function
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
#ifdef USE_COS_SIN
            double theta = 2.*M_PI*ud();
            double rsq = ud(); // cumulative dist function P(<r) = r^2 for unit circle
            double sint,cost;
            math::sincos(theta, sint, cost);
            // Then map radius to the desired Gaussian with analytic transformation
            double r = sqrt(rsq) * _r0;;
            photons.setPhoton(i, r*cost, r*sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2.*ud()-1.;
                yu = 2.*ud()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1.);
            photons.setPhoton(i, xu * _r0, yu * _r0, fluxPerPhoton);
#endif
        }
        dbg<<"TopHat Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
