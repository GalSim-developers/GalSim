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

#include "SBExponential.h"
#include "SBExponentialImpl.h"
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

// Define this use the Newton-Raphson method for solving the radial value in SBExponential::shoot
// rather than using OneDimensionalDeviate.
// The relative speed of the two methods was tested as part of issue #163, and the results
// are collated in devutils/external/time_photon_shooting.
// The conclusion was that using OneDimensionalDeviate was universally quite a bit faster.
// However, we leave this option here in case someone has an idea for massively speeding up
// the solution that might be faster than the table lookup.
//#define USE_NEWTON_RAPHSON

namespace galsim {

    SBExponential::SBExponential(double r0, double flux, const GSParams& gsparams) :
        SBProfile(new SBExponentialImpl(r0, flux, gsparams)) {}

    SBExponential::SBExponential(const SBExponential& rhs) : SBProfile(rhs) {}

    SBExponential::~SBExponential() {}

    double SBExponential::getScaleRadius() const
    {
        assert(dynamic_cast<const SBExponentialImpl*>(_pimpl.get()));
        return static_cast<const SBExponentialImpl&>(*_pimpl).getScaleRadius();
    }

    LRUCache<GSParamsPtr, ExponentialInfo> SBExponential::SBExponentialImpl::cache(
        sbp::max_exponential_cache);

    SBExponential::SBExponentialImpl::SBExponentialImpl(
        double r0, double flux, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _flux(flux), _r0(r0), _r0_sq(_r0*_r0), _inv_r0(1./r0), _inv_r0_sq(_inv_r0*_inv_r0),
        _info(cache.get(GSParamsPtr(gsparams)))
    {
        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-1.5 = kvalue_accuracy
        _ksq_max = (std::pow(this->gsparams.kvalue_accuracy,-1./1.5)-1.);
        _k_max = std::sqrt(_ksq_max);

        // For small k, we can use up to quartic in the taylor expansion to avoid the sqrt.
        // This is acceptable when the next term is less than kvalue_accuracy.
        // 35/16 (k^2 r0^2)^3 = kvalue_accuracy
        _ksq_min = std::pow(this->gsparams.kvalue_accuracy * 16./35., 1./3.);

        _flux_over_2pi = _flux / (2. * M_PI);
        _norm = _flux_over_2pi * _inv_r0_sq;

        dbg<<"Exponential:\n";
        dbg<<"_flux = "<<_flux<<std::endl;
        dbg<<"_r0 = "<<_r0<<std::endl;
        dbg<<"_ksq_max = "<<_ksq_max<<std::endl;
        dbg<<"_ksq_min = "<<_ksq_min<<std::endl;
        dbg<<"_norm = "<<_norm<<std::endl;
        dbg<<"maxK() = "<<maxK()<<std::endl;
        dbg<<"stepK() = "<<stepK()<<std::endl;
    }

    double SBExponential::SBExponentialImpl::maxK() const
    { return _info->maxK() * _inv_r0; }
    double SBExponential::SBExponentialImpl::stepK() const
    { return _info->stepK() * _inv_r0; }

    double SBExponential::SBExponentialImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x * p.x + p.y * p.y);
        return _norm * fmath::expd(-r * _inv_r0);
    }

    std::complex<double> SBExponential::SBExponentialImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y)*_r0_sq;

        if (ksq < _ksq_min) {
            return _flux*(1. - 1.5*ksq*(1. - 1.25*ksq));
        } else {
            double ksqp1 = 1. + ksq;
            return _flux / (ksqp1 * sqrt(ksqp1));
            // NB: flux*std::pow(ksqp1,-1.5) is slower.
        }
    }

    // A helper class for doing the inner loops in the below fill*Image functions.
    // This lets us do type-specific optimizations on just this portion.
    // First the normal (legible) version that we use if there is no SSE support. (HA!)
    template <typename T>
    struct InnerLoopHelper
    {
        static inline void kloop_1d(std::complex<T>*& ptr, int n,
                                    double kx, double dkx, double kysq, double flux)
        {
            const double kysqp1 = kysq + 1.;
            for (; n; --n, kx+=dkx) {
                double ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
        static inline void kloop_2d(std::complex<T>*& ptr, int n,
                                    double kx, double dkx, double ky, double dky, double flux)
        {
            for (; n; --n, kx+=dkx, ky+=dky) {
                double ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
    };

#ifdef __SSE__
    template <>
    struct InnerLoopHelper<float>
    {
        static inline void kloop_1d(std::complex<float>*& ptr, int n,
                                    float kx, float dkx, float kysq, float flux)
        {
            const float kysqp1 = kysq + 1.;

            // First get the pointer to an aligned boundary.  This usually requires at most one
            // iteration (often 0), but if the input is pathalogically not aligned on a 64 bit
            // boundary, then this will just run through the whole thing and produce the corrent
            // answer.  Just without any SSE speed up.
            for (; n && !IsAligned(ptr); --n,kx+=dkx) {
                float ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }

            int n4 = n>>2;
            int na = n4<<2;
            n -= na;

            // Do 4 at a time as far as possible.
            if (n4) {
                __m128 zero = _mm_setzero_ps();
                __m128 xflux = _mm_set1_ps(flux);
                __m128 xkysqp1 = _mm_set1_ps(kysqp1);
                __m128 xdkx = _mm_set1_ps(4.*dkx);
                // I never really understood why these are backwards, but that's just how
                // this function works.  They need to be in reverse order.
                __m128 xkx = _mm_set_ps(kx+3.*dkx, kx+2.*dkx, kx+dkx, kx);
                do {
                    // kxsq = kx * kx
                    __m128 kxsq = _mm_mul_ps(xkx, xkx);
                    // ksqp1 = kxsq + kysqp1
                    __m128 ksqp1 = _mm_add_ps(kxsq, xkysqp1);
                    // kx += 4*dkx
                    xkx = _mm_add_ps(xkx, xdkx);
                    // denom = ksqp1 * ksqp1 * ksqp1
                    __m128 denom = _mm_mul_ps(ksqp1,_mm_mul_ps(ksqp1, ksqp1));
                    // final = flux / denom
                    __m128 final = _mm_div_ps(xflux, _mm_sqrt_ps(denom));
                    // lo = unpacked final[0], 0.F, final[1], 0.F
                    __m128 lo = _mm_unpacklo_ps(final, zero);
                    // hi = unpacked final[2], 0.F, final[3], 0.F
                    __m128 hi = _mm_unpackhi_ps(final, zero);
                    // store these into the ptr array
                    _mm_store_ps(reinterpret_cast<float*>(ptr), lo);
                    _mm_store_ps(reinterpret_cast<float*>(ptr+2), hi);
                    ptr += 4;
                } while (--n4);
            }
            kx += na * dkx;

            // Finally finish up the last few values
            for (; n; --n,kx+=dkx) {
                float ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
        static inline void kloop_2d(std::complex<float>*& ptr, int n,
                                    float kx, float dkx, float ky, float dky, float flux)
        {
            for (; n && !IsAligned(ptr); --n,kx+=dkx,ky+=dky) {
                float ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }

            int n4 = n>>2;
            int na = n4<<2;
            n -= na;

            // Do 4 at a time as far as possible.
            if (n4) {
                __m128 zero = _mm_setzero_ps();
                __m128 one = _mm_set1_ps(1.);
                __m128 xflux = _mm_set1_ps(flux);
                __m128 xdkx = _mm_set1_ps(4.*dkx);
                __m128 xdky = _mm_set1_ps(4.*dky);
                __m128 xkx = _mm_set_ps(kx+3.*dkx, kx+2.*dkx, kx+dkx, kx);
                __m128 xky = _mm_set_ps(ky+3.*dky, ky+2.*dky, ky+dky, ky);
                do {
                    // kxsq = kx * kx
                    __m128 kxsq = _mm_mul_ps(xkx, xkx);
                    // kysq = ky * ky
                    __m128 kysq = _mm_mul_ps(xky, xky);
                    // ksqp1 = 1 + kxsq + kysq
                    __m128 ksqp1 = _mm_add_ps(one, _mm_add_ps(kxsq, kysq));
                    // kx += 4*dkx
                    xkx = _mm_add_ps(xkx, xdkx);
                    // ky += 4*dky
                    xky = _mm_add_ps(xky, xdky);
                    // denom = ksqp1 * ksqp1 * ksqp1
                    __m128 denom = _mm_mul_ps(ksqp1,_mm_mul_ps(ksqp1, ksqp1));
                    // final = flux / denom
                    __m128 final = _mm_div_ps(xflux, _mm_sqrt_ps(denom));
                    // lo = unpacked final[0], 0.F, final[1], 0.F
                    __m128 lo = _mm_unpacklo_ps(final, zero);
                    // hi = unpacked final[2], 0.F, final[3], 0.F
                    __m128 hi = _mm_unpackhi_ps(final, zero);
                    // store these into the ptr array
                    _mm_store_ps(reinterpret_cast<float*>(ptr), lo);
                    _mm_store_ps(reinterpret_cast<float*>(ptr+2), hi);
                    ptr += 4;
                } while (--n4);
            }
            kx += na * dkx;
            ky += na * dky;

            // Finally finish up the last few values
            for (; n; --n,kx+=dkx,ky+=dky) {
                float ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
    };
#endif
#ifdef __SSE2__
    template <>
    struct InnerLoopHelper<double>
    {
        static inline void kloop_1d(std::complex<double>*& ptr, int n,
                                    double kx, double dkx, double kysq, double flux)
        {
            const double kysqp1 = kysq + 1.;

            // If ptr isn't aligned, there is no hope in getting it there by incrementing,
            // since complex<double> is 128 bits, so just do the regular loop.
            if (!IsAligned(ptr)) {
                for (; n; --n,kx+=dkx) {
                    double ksqp1 = kx*kx + kysqp1;
                    *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
                }
                return;
            }

            int n2 = n>>1;
            int na = n2<<1;
            n -= na;

            // Do 2 at a time as far as possible.
            if (n2) {
                __m128d zero = _mm_set1_pd(0.);
                __m128d xflux = _mm_set1_pd(flux);
                __m128d xkysqp1 = _mm_set1_pd(kysqp1);
                __m128d xdkx = _mm_set1_pd(2.*dkx);
                __m128d xkx = _mm_set_pd(kx+dkx, kx);
                do {
                    // kxsq = kx * kx
                    __m128d kxsq = _mm_mul_pd(xkx, xkx);
                    // ksqp1 = kxsq + kysqp1
                    __m128d ksqp1 = _mm_add_pd(kxsq, xkysqp1);
                    // kx += 2*dkx
                    xkx = _mm_add_pd(xkx, xdkx);
                    // ksqp13 = ksqp1 * ksqp1 * ksqp1
                    __m128d denom = _mm_mul_pd(ksqp1,_mm_mul_pd(ksqp1, ksqp1));
                    // final = flux / denom
                    __m128d final = _mm_div_pd(xflux, _mm_sqrt_pd(denom));
                    // lo = unpacked final[0], 0.
                    __m128d lo = _mm_unpacklo_pd(final, zero);
                    // hi = unpacked final[1], 0.
                    __m128d hi = _mm_unpackhi_pd(final, zero);
                    // store these into the ptr array
                    _mm_store_pd(reinterpret_cast<double*>(ptr), lo);
                    _mm_store_pd(reinterpret_cast<double*>(ptr+1), hi);
                    ptr += 2;
                } while (--n2);
            }

            // Finally finish up the last value, if any
            if (n) {
                kx += na * dkx;
                double ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
        static inline void kloop_2d(std::complex<double>*& ptr, int n,
                                    double kx, double dkx, double ky, double dky, double flux)
        {
            if (!IsAligned(ptr)) {
                for (; n; --n,kx+=dkx) {
                    double ksqp1 = 1. + kx*kx + ky*ky;
                    *ptr++ = flux/(ksqp1*std::sqrt(ksqp1));
                }
                return;
            }

            int n2 = n>>1;
            int na = n2<<1;
            n -= na;

            // Do 2 at a time as far as possible.
            if (n2) {
                __m128d zero = _mm_set1_pd(0.);
                __m128d one = _mm_set1_pd(1.);
                __m128d xflux = _mm_set1_pd(flux);
                __m128d xdkx = _mm_set1_pd(2.*dkx);
                __m128d xdky = _mm_set1_pd(2.*dky);
                __m128d xkx = _mm_set_pd(kx+dkx, kx);
                __m128d xky = _mm_set_pd(ky+dky, ky);
                do {
                    // kxsq = kx * kx
                    __m128d kxsq = _mm_mul_pd(xkx, xkx);
                    // kysq = ky * ky
                    __m128d kysq = _mm_mul_pd(xky, xky);
                    // ksqp1 = 1 + kxsq + kysq
                    __m128d ksqp1 = _mm_add_pd(one, _mm_add_pd(kxsq, kysq));
                    // kx += 2*dkx
                    xkx = _mm_add_pd(xkx, xdkx);
                    // ky += 2*dky
                    xky = _mm_add_pd(xky, xdky);
                    // denom = ksqp1 * ksqp1 * ksqp1
                    __m128d denom = _mm_mul_pd(ksqp1,_mm_mul_pd(ksqp1, ksqp1));
                    // final = flux / denom
                    __m128d final = _mm_div_pd(xflux, _mm_sqrt_pd(denom));
                    // lo = unpacked final[0], 0.
                    __m128d lo = _mm_unpacklo_pd(final, zero);
                    // hi = unpacked final[1], 0.
                    __m128d hi = _mm_unpackhi_pd(final, zero);
                    // store these into the ptr array
                    _mm_store_pd(reinterpret_cast<double*>(ptr), lo);
                    _mm_store_pd(reinterpret_cast<double*>(ptr+1), hi);
                    ptr += 2;
                } while (--n2);
            }

            // Finally finish up the last value, if any
            if (n) {
                kx += na * dkx;
                ky += na * dky;
                double ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux / (ksqp1*std::sqrt(ksqp1));
            }
        }
    };
#endif

    template <typename T>
    void SBExponential::SBExponentialImpl::fillXImage(ImageView<T> im,
                                                      double x0, double dx, int izero,
                                                      double y0, double dy, int jzero) const
    {
        dbg<<"SBExponential fillXImage\n";
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

            x0 *= _inv_r0;
            dx *= _inv_r0;
            y0 *= _inv_r0;
            dy *= _inv_r0;

            for (int j=0; j<n; ++j,y0+=dy,ptr+=skip) {
                double x = x0;
                double ysq = y0*y0;
                for (int i=0;i<m;++i,x+=dx)
                    *ptr++ = _norm * fmath::expd(-sqrt(x*x + ysq));
            }
        }
    }

    template <typename T>
    void SBExponential::SBExponentialImpl::fillXImage(ImageView<T> im,
                                                      double x0, double dx, double dxy,
                                                      double y0, double dy, double dyx) const
    {
        dbg<<"SBExponential fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        x0 *= _inv_r0;
        dx *= _inv_r0;
        dxy *= _inv_r0;
        y0 *= _inv_r0;
        dy *= _inv_r0;
        dyx *= _inv_r0;

        for (int j=0; j<n; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx)
                *ptr++ = _norm * fmath::expd(-sqrt(x*x + y*y));
        }
    }

    template <typename T>
    void SBExponential::SBExponentialImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBExponential fillKImage\n";
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
                int i1,i2;
                double kysq; // GetKValueRange1d will compute this i1 != m
                GetKValueRange1d(i1, i2, m, _k_max, _ksq_max, kx0, dkx, ky0, kysq);
                for (int i=i1; i; --i) *ptr++ = T(0);
                if (i1 == m) continue;
                double kx = kx0 + i1 * dkx;
                InnerLoopHelper<T>::kloop_1d(ptr, i2-i1, kx, dkx, kysq, _flux);
                for (int i=m-i2; i; --i) *ptr++ = T(0);
            }
        }
    }

    template <typename T>
    void SBExponential::SBExponentialImpl::fillKImage(ImageView<std::complex<T> > im,
                                                      double kx0, double dkx, double dkxy,
                                                      double ky0, double dky, double dkyx) const
    {
        dbg<<"SBExponential fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= _r0;
        dkx *= _r0;
        dkxy *= _r0;
        ky0 *= _r0;
        dky *= _r0;
        dkyx *= _r0;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            int i1,i2;
            GetKValueRange2d(i1, i2, m, _k_max, _ksq_max, kx0, dkx, ky0, dkyx);
            for (int i=i1; i; --i) *ptr++ = T(0);
            if (i1 == m) continue;
            double kx = kx0 + i1 * dkx;
            double ky = ky0 + i1 * dkyx;
            InnerLoopHelper<T>::kloop_2d(ptr, i2-i1, kx, dkx, ky, dkyx, _flux);
            for (int i=m-i2; i; --i) *ptr++ = T(0);
        }
    }

    // Constructor to initialize Exponential functions for 1D deviate photon shooting
    ExponentialInfo::ExponentialInfo(const GSParamsPtr& gsparams)
    {
        dbg<<"Start ExponentialInfo with gsparams = "<<*gsparams<<std::endl;
#ifndef USE_NEWTON_RAPHSON
        // Next, set up the classes for photon shooting
        _radial.reset(new ExponentialRadialFunction());
        dbg<<"Made radial"<<std::endl;
        std::vector<double> range(2,0.);
        range[1] = -std::log(gsparams->shoot_accuracy);
        _sampler.reset(new OneDimensionalDeviate(*_radial, range, true, 2.*M_PI, *gsparams));
        dbg<<"Made sampler"<<std::endl;
#endif

        // Calculate maxk:
        _maxk = std::pow(gsparams->maxk_threshold, -1./3.);
        dbg<<"maxk = "<<_maxk<<std::endl;

        // Calculate stepk:
        // int( exp(-r) r, r=0..R) = (1 - exp(-R) - Rexp(-R))
        // Fraction excluded is thus (1+R) exp(-R)
        // A fast solution to (1+R)exp(-R) = x:
        // log(1+R) - R = log(x)
        // R = log(1+R) - log(x)
        double logx = std::log(gsparams->folding_threshold);
        double R = -logx;
        for (int i=0; i<3; i++) R = std::log(1.+R) - logx;
        // Make sure it is at least 5 hlr
        // half-light radius = 1.6783469900166605 * r0
        const double hlr = 1.6783469900166605;
        R = std::max(R,gsparams->stepk_minimum_hlr*hlr);
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double ExponentialInfo::maxK() const
    { return _maxk; }

    // The amount of flux missed in a circle of radius pi/stepk should be at
    // most folding_threshold of the flux.
    double ExponentialInfo::stepK() const
    { return _stepk; }

    void ExponentialInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        assert(_sampler.get());
        _sampler->shoot(photons,ud);
        dbg<<"ExponentialInfo Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    void SBExponential::SBExponentialImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Exponential shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
#ifdef USE_NEWTON_RAPHSON
        // The cumulative distribution of flux is 1-(1+r)exp(-r).
        // Here is a way to solve for r by an initial guess followed
        // by Newton-Raphson iterations.  Probably not
        // the most efficient thing since there are logs in the iteration.

        // Accuracy to which to solve for (log of) cumulative flux distribution:
        const double Y_TOLERANCE=this->gsparams.shoot_accuracy;

        double fluxPerPhoton = _flux / N;

        for (int i=0; i<N; i++) {
            double y = ud();
            if (y==0.) {
                // In case of infinite radius - just set to origin:
                photons.setPhoton(i,0.,0.,fluxPerPhoton);
                continue;
            }
            // Convert from y = (1+r)exp(-r)
            // to y' = -log(y) = r - log(1+r)
            y = -std::log(y);
            // Initial guess.  Good to +- 0.1 out to quite large values of r.
            dbg<<"y = "<<y<<std::endl;
            double r = y<0.07 ? sqrt(2.*y) : y<0.9 ? 1.8*y+0.37 : 1.3*y+0.83;
            double dy = y - r + std::log(1.+r);
            dbg<<"dy, r = \n";
            dbg<<dy<<"  "<<r<<std::endl;
            while ( std::abs(dy) > Y_TOLERANCE) {
                // Newton step: dy/dr = r / (1+r)
                r += (1.+r)*dy/r;
                dy = y - r + std::log(1.+r);
                dbg<<dy<<"  "<<r<<std::endl;
            }

            // Draw another (or multiple) randoms for azimuthal angle
#ifdef USE_COS_SIN
            double theta = 2. * M_PI * ud();
            double sint,cost;
            math::sincos(theta, sint, cost);
            double rFactor = r * _r0;
            photons.setPhoton(i, rFactor * cost, rFactor * sint, fluxPerPhoton);
#else
            double xu, yu, rsq;
            do {
                xu = 2. * ud() - 1.;
                yu = 2. * ud() - 1.;
                rsq = xu*xu+yu*yu;
            } while (rsq >= 1. || rsq == 0.);
            double rFactor = r * _r0 / std::sqrt(rsq);
            photons.setPhoton(i, rFactor * xu, rFactor * yu, fluxPerPhoton);
#endif
        }
#else
        // Get photons from the ExponentialInfo structure, rescale flux and size for this instance
        dbg<<"flux scaling = "<<_flux_over_2pi<<std::endl;
        dbg<<"r0 = "<<_r0<<std::endl;
        _info->shoot(photons,ud);
        photons.scaleFlux(_flux_over_2pi);
        photons.scaleXY(_r0);
#endif
        dbg<<"Exponential Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
