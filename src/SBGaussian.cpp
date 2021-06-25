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

#include "SBGaussian.h"
#include "SBGaussianImpl.h"
#include "math/Angle.h"
#include "fmath/fmath.hpp"

// Define this variable to find azimuth (and sometimes radius within a unit disc) of 2d photons by
// drawing a uniform deviate for theta, instead of drawing 2 deviates for a point on the unit
// circle and rejecting corner photons.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using sin/cos was faster for icpc, but not g++ or clang++.
#ifdef _INTEL_COMPILER
#define USE_COS_SIN
#endif

namespace galsim {


    SBGaussian::SBGaussian(double sigma, double flux, const GSParams& gsparams) :
        SBProfile(new SBGaussianImpl(sigma, flux, gsparams)) {}

    SBGaussian::SBGaussian(const SBGaussian& rhs) : SBProfile(rhs) {}

    SBGaussian::~SBGaussian() {}

    double SBGaussian::getSigma() const
    {
        assert(dynamic_cast<const SBGaussianImpl*>(_pimpl.get()));
        return static_cast<const SBGaussianImpl&>(*_pimpl).getSigma();
    }

    SBGaussian::SBGaussianImpl::SBGaussianImpl(double sigma, double flux,
                                               const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _flux(flux), _sigma(sigma), _sigma_sq(_sigma*_sigma),
        _inv_sigma(1./_sigma), _inv_sigma_sq(_inv_sigma*_inv_sigma)
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // exp(-k^2*sigma^2/2) = kvalue_accuracy
        _ksq_max = -2. * std::log(this->gsparams.kvalue_accuracy);

        // For small k, we can use up to quartic in the taylor expansion to avoid the exp.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 1/48 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(this->gsparams.kvalue_accuracy * 48., 1./3.);

        _norm = _flux * _inv_sigma_sq / (2. * M_PI);

        dbg<<"Gaussian:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_sigma = "<<_sigma<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBGaussian::SBGaussianImpl::maxK() const
    { return sqrt(-2.*std::log(this->gsparams.maxk_threshold))*_inv_sigma; }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double SBGaussian::SBGaussianImpl::stepK() const
    {
        // int( exp(-r^2/2) r, r=0..R) = 1 - exp(-R^2/2)
        // exp(-R^2/2) = folding_threshold
        double R = sqrt(-2.*std::log(this->gsparams.folding_threshold));
        // Make sure it is at least 5 hlr
        // half-light radius = sqrt(2ln(2)) * sigma
        const double hlr = 1.177410022515475;
        R = std::max(R,gsparams.stepk_minimum_hlr*hlr);
        return M_PI / (R*_sigma);
    }

    double SBGaussian::SBGaussianImpl::xValue(const Position<double>& p) const
    {
        double rsq = p.x*p.x + p.y*p.y;
        return _norm * fmath::expd( -0.5 * rsq * _inv_sigma_sq );
    }

    std::complex<double> SBGaussian::SBGaussianImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x+k.y*k.y)*_sigma_sq;
        if (ksq > _ksq_max) {
            return 0.;
        } else if (ksq < _ksq_min) {
            return _flux*(1. - 0.5*ksq*(1. - 0.25*ksq));
        } else {
            return _flux * fmath::expd(-0.5*ksq);
        }
    }

    template <typename T>
    void SBGaussian::SBGaussianImpl::fillXImage(ImageView<T> im,
                                                double x0, double dx, int izero,
                                                double y0, double dy, int jzero) const
    {
        dbg<<"SBGaussian fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        if (izero != 0 || jzero != 0) {
            xdbg<<"Use Quadrant\n";
            fillXImageQuadrant(im,x0,dx,izero,y0,dy,jzero);
        } else {
            xdbg<<"Non-Quadrant\n";
            const int m = im.getNCol();
            const int n = im.getNRow();
            T* ptr = im.getData();
            const int skip = im.getNSkip();
            assert(im.getStep() == 1);

            x0 *= _inv_sigma;
            dx *= _inv_sigma;
            y0 *= _inv_sigma;
            dy *= _inv_sigma;

            // The Gaussian profile is separable:
            //    im(x,y) = _norm * exp(-0.5 * (x*x + y*y)
            //            = _norm * exp(-0.5 * x*x) * exp(-0.5 * y*y)
            std::vector<double> gauss_x(m);
            std::vector<double> gauss_y(n);
            typedef std::vector<double>::iterator It;
            It xit = gauss_x.begin();
            for (int i=0; i<m; ++i,x0+=dx) *xit++ = fmath::expd(-0.5 * x0*x0);

            if ((x0 == y0) && (dx == dy) && (m==n)) {
                gauss_y = gauss_x;
            } else {
                It yit = gauss_y.begin();
                for (int j=0; j<n; ++j,y0+=dy) *yit++ = fmath::expd(-0.5 * y0*y0);
            }

            for (int j=0; j<n; ++j,ptr+=skip) {
                for (int i=0; i<m; ++i)
                    *ptr++ = _norm * gauss_x[i] * gauss_y[j];
            }
        }
    }

    template <typename T>
    void SBGaussian::SBGaussianImpl::fillXImage(ImageView<T> im,
                                                double x0, double dx, double dxy,
                                                double y0, double dy, double dyx) const
    {
        dbg<<"SBGaussian fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _inv_sigma;
        dx *= _inv_sigma;
        dxy *= _inv_sigma;
        y0 *= _inv_sigma;
        dy *= _inv_sigma;
        dyx *= _inv_sigma;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _norm * fmath::expd( -0.5 * (x*x + y*y) );
        }
    }

    template <typename T>
    void SBGaussian::SBGaussianImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBGaussian fillKImage\n";
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

            kx0 *= _sigma;
            dkx *= _sigma;
            ky0 *= _sigma;
            dky *= _sigma;

            // The Gaussian profile is separable:
            //    im(kx,ky) = _flux * exp(-0.5 * (kx*kx + ky*ky)
            //              = _flux * exp(-0.5 * kx*kx) * exp(-0.5 * ky*ky)
            std::vector<double> gauss_kx(m);
            std::vector<double> gauss_ky(n);
            typedef std::vector<double>::iterator It;
            It kxit = gauss_kx.begin();

            for (int i=0; i<m; ++i,kx0+=dkx) *kxit++ = fmath::expd(-0.5 * kx0*kx0);

            if ((kx0 == ky0) && (dkx == dky) && (m==n)) {
                gauss_ky = gauss_kx;
            } else {
                It kyit = gauss_ky.begin();
                for (int j=0; j<n; ++j,ky0+=dky) *kyit++ = fmath::expd(-0.5 * ky0*ky0);
            }

            for (int j=0; j<n; ++j,ptr+=skip) {
                for (int i=0; i<m; ++i)
                    *ptr++ = _flux * gauss_kx[i] * gauss_ky[j];
            }
        }
    }

    template <typename T>
    void SBGaussian::SBGaussianImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBGaussian fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _sigma;
        dkx *= _sigma;
        dkxy *= _sigma;
        ky0 *= _sigma;
        dky *= _sigma;
        dkyx *= _sigma;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx) {
                double ksq = kx*kx + ky*ky;
                if (ksq > _ksq_max) {
                    *ptr++ = 0.;
                } else if (ksq < _ksq_min) {
                    *ptr++ = _flux * (1. - 0.5*ksq*(1. - 0.25*ksq));
                } else {
                    *ptr++ =  _flux * fmath::expd(-0.5*ksq);
                }
            }
        }
    }

    void SBGaussian::SBGaussianImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Gaussian shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        double fluxPerPhoton = _flux/N;
        for (int i=0; i<N; i++) {
            // First get a point uniformly distributed on unit circle
#ifdef USE_COS_SIN
            double theta = 2.*M_PI*ud();
            double rsq = ud(); // cumulative dist function P(<r) = r^2 for unit circle
            double sint,cost;
            math::sincos(theta, sint, cost);
            // Then map radius to the desired Gaussian with analytic transformation
            double rFactor = _sigma * std::sqrt( -2. * std::log(rsq));
            photons.setPhoton(i, rFactor*cost, rFactor*sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2.*ud()-1.;
                yu = 2.*ud()-1.;
                rsq = xu*xu+yu*yu;
            } while (rsq>=1. || rsq==0.);
            // Then map radius to the desired Gaussian with analytic transformation
            double rFactor = _sigma * std::sqrt( -2. * std::log(rsq) / rsq);
            photons.setPhoton(i, rFactor*xu, rFactor*yu, fluxPerPhoton);
#endif
        }
        dbg<<"Gaussian Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
