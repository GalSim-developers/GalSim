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

#include "SBSpergel.h"
#include "SBSpergelImpl.h"
#include "Solve.h"
#include "math/Bessel.h"
#include "math/Gamma.h"
#include "fmath/fmath.hpp"

namespace galsim {

    inline double fast_pow(double x, double y)
    { return fmath::expd(y * std::log(x)); }

    inline float fast_pow(float x, float y)
    { return fmath::exp(y * fmath::log(x)); }

    SBSpergel::SBSpergel(double nu, double scale_radius, double flux,
                         const GSParams& gsparams) :
        SBProfile(new SBSpergelImpl(nu, scale_radius, flux, gsparams)) {}

    SBSpergel::SBSpergel(const SBSpergel& rhs) : SBProfile(rhs) {}

    SBSpergel::~SBSpergel() {}

    double SBSpergel::getNu() const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).getNu();
    }

    double SBSpergel::getScaleRadius() const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).getScaleRadius();
    }

    double SBSpergel::calculateIntegratedFlux(double r) const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).calculateIntegratedFlux(r);
    }

    double SBSpergel::calculateFluxRadius(double f) const
    {
        assert(dynamic_cast<const SBSpergelImpl*>(_pimpl.get()));
        return static_cast<const SBSpergelImpl&>(*_pimpl).calculateFluxRadius(f);
    }

    LRUCache<Tuple<double,GSParamsPtr>,SpergelInfo> SBSpergel::SBSpergelImpl::cache(
        sbp::max_spergel_cache);

    SBSpergel::SBSpergelImpl::SBSpergelImpl(double nu, double scale_radius,
                                            double flux, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _nu(nu), _flux(flux), _r0(scale_radius),
        _info(cache.get(MakeTuple(_nu, GSParamsPtr(this->gsparams))))
    {
        dbg<<"Start SBSpergel constructor:\n";
        dbg<<"nu = "<<_nu<<std::endl;
        dbg<<"scale_radius = "<<scale_radius<<std::endl;
        dbg<<"flux = "<<_flux<<std::endl;

        // For large k, we clip the result of kValue to 0.
        // We do this when the correct answer is less than kvalue_accuracy.
        // (1+k^2 r0^2)^-(nu+1) = kvalue_accuracy
        _ksq_max = std::pow(this->gsparams.kvalue_accuracy,-1./(nu+1.))-1.;
        _k_max = std::sqrt(_ksq_max);
        _r0_sq = _r0 * _r0;
        _inv_r0 = 1. / _r0;
        _shootnorm = _flux * _info->getXNorm();
        _xnorm = _shootnorm / _r0_sq;

        dbg<<"scale radius = "<<_r0<<std::endl;
    }

    double SBSpergel::SBSpergelImpl::maxK() const { return _info->maxK() * _inv_r0; }
    double SBSpergel::SBSpergelImpl::stepK() const { return _info->stepK() * _inv_r0; }

    double SBSpergel::SBSpergelImpl::calculateIntegratedFlux(double r) const
    { return _info->calculateIntegratedFlux(r*_inv_r0);}
    double SBSpergel::SBSpergelImpl::calculateFluxRadius(double f) const
    { return _info->calculateFluxRadius(f) * _r0; }

    // Equations (3, 4) of Spergel (2010)
    double SBSpergel::SBSpergelImpl::xValue(const Position<double>& p) const
    {
        double r = sqrt(p.x * p.x + p.y * p.y) * _inv_r0;
        return _xnorm * _info->xValue(r);
    }

    // Equation (2) of Spergel (2010)
    std::complex<double> SBSpergel::SBSpergelImpl::kValue(const Position<double>& k) const
    {
        double ksq = (k.x*k.x + k.y*k.y) * _r0_sq;
        return _flux * _info->kValue(ksq);
    }

    // A helper class for doing the inner loops in the below fill*Image functions.
    // This lets us do type-specific optimizations on just this portion.
    // First the normal (legible) version that we use if there is no SSE support.
    template <typename T>
    struct InnerLoopHelper
    {
        static inline void kloop_1d(std::complex<T>*& ptr, int n, double mnup1,
                                    double kx, double dkx, double kysq, double flux)
        {
            const double kysqp1 = kysq + 1.;
            for (; n; --n,kx+=dkx) {
                double ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux * std::pow(ksqp1, mnup1);
            }
        }
        static inline void kloop_2d(std::complex<T>*& ptr, int n, double mnup1,
                                    double kx, double dkx, double ky, double dky, double flux)
        {
            for (; n; --n,kx+=dkx,ky+=dky) {
                double ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux * std::pow(ksqp1, mnup1);
            }
        }
    };

#ifdef __SSE__
    template <>
    struct InnerLoopHelper<float>
    {
        static inline void kloop_1d(std::complex<float>*& ptr, int n, float mnup1,
                                    float kx, float dkx, float kysq, float flux)
        {
            const float kysqp1 = kysq + 1.;

            // First get the pointer to an aligned boundary.  This usually requires at most one
            // iteration (often 0), but if the input is pathalogically not aligned on a 64 bit
            // boundary, then this will just run through the whole thing and produce the corrent
            // answer.  Just without any SSE speed up.
            for (; n && !IsAligned(ptr); --n,kx+=dkx) {
                float ksqp1 = kx*kx + kysqp1;
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }

            int n4 = n>>2;
            int na = n4<<2;
            n -= na;

            // Do 4 at a time as far as possible.
            if (n4) {
                __m128 zero = _mm_setzero_ps();
                __m128 xmnup1 = _mm_set1_ps(mnup1);
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
                    // temp = pow(ksqp1, mnup1) = exp(mnup1 * log(ksqp1))
                    __m128 temp = fmath::exp_ps(_mm_mul_ps(xmnup1,fmath::log_ps(ksqp1)));
                    // final = flux * temp
                    __m128 final = _mm_mul_ps(xflux, temp);
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
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }
        }
        static inline void kloop_2d(std::complex<float>*& ptr, int n, float mnup1,
                                    float kx, float dkx, float ky, float dky, float flux)
        {
            for (; n && !IsAligned(ptr); --n,kx+=dkx,ky+=dky) {
                float ksqp1 = 1. + kx*kx + ky*ky;
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }

            int n4 = n>>2;
            int na = n4<<2;
            n -= na;

            // Do 4 at a time as far as possible.
            if (n4) {
                __m128 zero = _mm_setzero_ps();
                __m128 one = _mm_set1_ps(1.);
                __m128 xmnup1 = _mm_set1_ps(mnup1);
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
                    // temp = pow(ksqp1, mnup1) = exp(mnup1 * log(ksqp1))
                    __m128 temp = fmath::exp_ps(_mm_mul_ps(xmnup1,fmath::log_ps(ksqp1)));
                    // final = flux * temp
                    __m128 final = _mm_mul_ps(xflux, temp);
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
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }
        }
    };
#endif
#ifdef __SSE2__
    // fmath doesn't have a log_pd function, so this does the equivalent using std::log.
    inline __m128d log_pd(__m128d x)
    {
        union { __m128d m; double d[2]; } logx;
        logx.d[0] = std::log(*reinterpret_cast<double*>(&x));
        logx.d[1] = std::log(*(reinterpret_cast<double*>(&x)+1));
        return logx.m;
    }

    template <>
    struct InnerLoopHelper<double>
    {
        static inline void kloop_1d(std::complex<double>*& ptr, int n, double mnup1,
                                    double kx, double dkx, double kysq, double flux)
        {
            const double kysqp1 = kysq + 1.;

            // If ptr isn't aligned, there is no hope in getting it there by incrementing,
            // since complex<double> is 128 bits, so just do the regular loop.
            if (!IsAligned(ptr)) {
                for (; n; --n,kx+=dkx) {
                    double ksqp1 = kx*kx + kysqp1;
                    *ptr++ = flux * fast_pow(ksqp1, mnup1);
                }
                return;
            }

            int n2 = n>>1;
            int na = n2<<1;
            n -= na;

            // Do 2 at a time as far as possible.
            if (n2) {
                __m128d zero = _mm_set1_pd(0.);
                __m128d xmnup1 = _mm_set1_pd(mnup1);
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
                    // temp = pow(ksqp1, mnup1) = exp(mnup1 * logksqp1)
                    __m128d temp = fmath::exp_pd(_mm_mul_pd(xmnup1,log_pd(ksqp1)));
                    // final = flux * temp
                    __m128d final = _mm_mul_pd(xflux, temp);
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
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }
        }
        static inline void kloop_2d(std::complex<double>*& ptr, int n, double mnup1,
                                    double kx, double dkx, double ky, double dky, double flux)
        {
            if (!IsAligned(ptr)) {
                for (; n; --n,kx+=dkx) {
                    double ksqp1 = 1. + kx*kx + ky*ky;
                    *ptr++ = flux * fast_pow(ksqp1, mnup1);
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
                __m128d xmnup1 = _mm_set1_pd(mnup1);
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
                    // temp = pow(ksqp1, mnup1) = exp(mnup1 * logksqp1)
                    __m128d temp = fmath::exp_pd(_mm_mul_pd(xmnup1,log_pd(ksqp1)));
                    // final = flux * temp
                    __m128d final = _mm_mul_pd(xflux, temp);
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
                *ptr++ = flux * fast_pow(ksqp1, mnup1);
            }
        }
    };
#endif

    template <typename T>
    void SBSpergel::SBSpergelImpl::fillXImage(ImageView<T> im,
                                              double x0, double dx, int izero,
                                              double y0, double dy, int jzero) const
    {
        dbg<<"SBSpergel fillXImage\n";
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
                for (int i=0; i<m; ++i,x+=dx)
                    *ptr++ = _xnorm * _info->xValue(sqrt(x*x + ysq));
            }
        }
    }

    template <typename T>
    void SBSpergel::SBSpergelImpl::fillXImage(ImageView<T> im,
                                              double x0, double dx, double dxy,
                                              double y0, double dy, double dyx) const
    {
        dbg<<"SBSpergel fillXImage\n";
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
            for (int i=0; i<m; ++i,x+=dx,y+=dyx)
                *ptr++ = _xnorm * _info->xValue(sqrt(x*x + y*y));
        }
    }

    template <typename T>
    void SBSpergel::SBSpergelImpl::fillKImage(ImageView<std::complex<T> > im,
                                              double kx0, double dkx, int izero,
                                              double ky0, double dky, int jzero) const
    {
        dbg<<"SBSpergel fillKImage\n";
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

            double mnup1 = -(_nu + 1.);

            for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
                int i1,i2;
                double kysq; // GetKValueRange1d will compute this i1 != m
                GetKValueRange1d(i1, i2, m, _k_max, _ksq_max, kx0, dkx, ky0, kysq);
                for (int i=i1; i; --i) *ptr++ = T(0);
                if (i1 == m) continue;
                double kx = kx0 + i1 * dkx;
                InnerLoopHelper<T>::kloop_1d(ptr, i2-i1, mnup1, kx, dkx, kysq, _flux);
                for (int i=m-i2; i; --i) *ptr++ = T(0);
            }
        }
    }

    template <typename T>
    void SBSpergel::SBSpergelImpl::fillKImage(ImageView<std::complex<T> > im,
                                              double kx0, double dkx, double dkxy,
                                              double ky0, double dky, double dkyx) const
    {
        dbg<<"SBSpergel fillKImage\n";
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
        double mnup1 = -(_nu + 1.);

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            int i1,i2;
            GetKValueRange2d(i1, i2, m, _k_max, _ksq_max, kx0, dkx, ky0, dkyx);
            for (int i=i1; i; --i) *ptr++ = T(0);
            if (i1 == m) continue;
            double kx = kx0 + i1 * dkx;
            double ky = ky0 + i1 * dkyx;
            InnerLoopHelper<T>::kloop_2d(ptr, i2-i1, mnup1, kx, dkx, ky, dkyx, _flux);
            for (int i=m-i2; i; --i) *ptr++ = T(0);
        }
    }

    SpergelInfo::SpergelInfo(double nu, const GSParamsPtr& gsparams) :
        _nu(nu), _gsparams(gsparams),
        _gamma_nup1(math::tgamma(_nu+1.0)),
        _gamma_nup2(_gamma_nup1 * (_nu+1)),
        _xnorm0((_nu > 0.) ? _gamma_nup1 / (2. * _nu) * std::pow(2., _nu) : INFINITY),
        _maxk(0.), _stepk(0.), _re(0.)
    {
        dbg<<"Start SpergelInfo constructor for nu = "<<_nu<<std::endl;

        if (_nu < sbp::minimum_spergel_nu || _nu > sbp::maximum_spergel_nu)
            throw SBError("Requested Spergel index out of range");
    }

    class SpergelIntegratedFlux
    {
    public:
        SpergelIntegratedFlux(double nu, double gamma_nup2, double flux_frac=0.0)
            : _nu(nu), _gamma_nup2(gamma_nup2),  _target(flux_frac) {}

        double operator()(double u) const
        {
            // Return flux integrated up to radius `u` in units of r0, minus `flux_frac`
            // (i.e., make a residual so this can be used to search for a target flux.
            // This result is derived in Spergel (2010) eqn. 8 by going to Fourier space
            // and integrating by parts.
            // The key Bessel identities:
            // int(r J0(k r), r=0..R) = R J1(k R) / k
            // d[-J0(k R)]/dk = R J1(k R)
            // The definition of the radial surface brightness profile and Fourier transform:
            // Sigma_nu(r) = (r/2)^nu K_nu(r)/Gamma(nu+1)
            //             = int(k J0(k r) / (1+k^2)^(1+nu), k=0..inf)
            // and the main result:
            // F(R) = int(2 pi r Sigma(r), r=0..R)
            //      = int(r int(k J0(k r) / (1+k^2)^(1+nu), k=0..inf), r=0..R) // Do the r-integral
            //      = int(R J1(k R)/(1+k^2)^(1+nu), k=0..inf)
            // Now integrate by parts with
            //      u = 1/(1+k^2)^(1+nu)                 dv = R J1(k R) dk
            // =>  du = -2 k (1+nu)/(1+k^2)^(2+nu) dk     v = -J0(k R)
            // => F(R) = u v | k=0,inf - int(v du, k=0..inf)
            //         = (0 + 1) - 2 (1+nu) int(k J0(k R) / (1+k^2)^2+nu, k=0..inf)
            //         = 1 - 2 (1+nu) (R/2)^(nu+1) K_{nu+1}(R) / Gamma(nu+2)
            double fnup1 = std::pow(u/2., _nu+1.) * math::cyl_bessel_k(_nu+1., u) / _gamma_nup2;
            double f = 1.0 - 2.0 * (1.+_nu)*fnup1;
            return f - _target;
        }
    private:
        double _nu;
        double _gamma_nup2;
        double _target;
    };

    static double CalculateFluxRadius(double flux_frac, double nu, double gamma_nup2)
    {
        // Calcute r such that L(r/r0) / L_tot == flux_frac

        // These bracket the range of calculateFluxRadius(0.5) for -0.85 < nu < 4.0.
        double z1=0.1;
        double z2=3.5;
        SpergelIntegratedFlux func(nu, gamma_nup2, flux_frac);
        Solve<SpergelIntegratedFlux> solver(func, z1, z2);
        solver.setXTolerance(1.e-25); // Spergels can be super peaky, so need a tight tolerance.
        solver.setMethod(Brent);
        if (flux_frac < 0.5)
            solver.bracketLowerWithLimit(0.0);
        else
            solver.bracketUpper();
        double R = solver.root();
        dbg<<"flux_frac = "<<flux_frac<<std::endl;
        dbg<<"r/r0 = "<<R<<std::endl;
        return R;
    }

    double SpergelInfo::calculateFluxRadius(double flux_frac) const
    {
        return CalculateFluxRadius(flux_frac, _nu, _gamma_nup2);
    }

    double SpergelCalculateHLR(double nu)
    {
        return CalculateFluxRadius(0.5, nu, math::tgamma(nu+2.));
    }

    double SpergelInfo::calculateIntegratedFlux(double r) const
    {
        SpergelIntegratedFlux func(_nu, _gamma_nup2);
        return func(r);
    }

    double SpergelInfo::stepK() const
    {
        if (_stepk == 0.) {
            double R = calculateFluxRadius(1.0 - _gsparams->folding_threshold);
            // Go to at least 5*re
            R = std::max(R,_gsparams->stepk_minimum_hlr * getHLR());
            dbg<<"R => "<<R<<std::endl;
            _stepk = M_PI / R;
            dbg<<"stepk = "<<_stepk<<std::endl;
        }
        return _stepk;
    }

    double SpergelInfo::maxK() const
    {
        if(_maxk == 0.) {
            // Solving (1+k^2)^(-1-nu) = maxk_threshold for k
            _maxk = std::sqrt(std::pow(_gsparams->maxk_threshold, -1./(1+_nu))-1.0);
        }
        return _maxk;
    }

    double SpergelInfo::getHLR() const
    {
        if (_re == 0.0) _re = calculateFluxRadius(0.5);
        return _re;
    }

    double SpergelInfo::getXNorm() const
    { return std::pow(2., -_nu) / _gamma_nup1 / (2.0 * M_PI); }

    double SpergelInfo::xValue(double r) const
    {
        if (r == 0.) return _xnorm0;
        else return math::cyl_bessel_k(_nu, r) * fast_pow(r, _nu);
    }

    double SpergelInfo::kValue(double ksq) const
    {
        return fast_pow(1. + ksq, -1. - _nu);
    }

    class SpergelNuPositiveRadialFunction: public FluxDensity
    {
    public:
        SpergelNuPositiveRadialFunction(double nu, double xnorm0):
            _nu(nu), _xnorm0(xnorm0) {}
        double operator()(double r) const {
            if (r == 0.) return _xnorm0;
            else return math::cyl_bessel_k(_nu, r) * fast_pow(r,_nu);
        }
    private:
        double _nu;
        double _xnorm0;
    };

    class SpergelNuNegativeRadialFunction: public FluxDensity
    {
    public:
        SpergelNuNegativeRadialFunction(double nu, double rmin, double a, double b):
            _nu(nu), _rmin(rmin), _a(a), _b(b) {}
        double operator()(double r) const {
            if (r <= _rmin) return _a + _b*r;
            else return math::cyl_bessel_k(_nu, r) * fast_pow(r,_nu);
        }
    private:
        double _nu;
        double _rmin;
        double _a;
        double _b;
    };

    void SpergelInfo::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        if (!_sampler) {
            // Set up the classes for photon shooting
            double shoot_rmax = calculateFluxRadius(1. - _gsparams->shoot_accuracy);
            if (_nu > 0.) {
                std::vector<double> range(2,0.);
                range[1] = shoot_rmax;
                _radial.reset(new SpergelNuPositiveRadialFunction(_nu, _xnorm0));
                double nominal_flux = 2.*M_PI*std::pow(2.,_nu)*_gamma_nup1;
                _sampler.reset(new OneDimensionalDeviate(*_radial, range, true, nominal_flux,
                                                         *_gsparams));
            } else {
                // exact s.b. profile diverges at origin, so replace the inner most circle
                // (defined such that enclosed flux is shoot_acccuracy) with a linear function
                // that contains the same flux and has the right value at r = rmin.
                // So need to solve the following for a and b:
                // int(2 pi r (a + b r) dr, 0..rmin) = shoot_accuracy
                // a + b rmin = K_nu(rmin) * rmin^nu
                double flux_target = _gsparams->shoot_accuracy;
                double shoot_rmin = calculateFluxRadius(flux_target);
                double knur = math::cyl_bessel_k(_nu, shoot_rmin) * fast_pow(shoot_rmin, _nu);
                double b = 3./shoot_rmin*(knur - flux_target/(M_PI*shoot_rmin*shoot_rmin));
                double a = knur - shoot_rmin*b;
                dbg<<"flux target: "<<flux_target<<std::endl;
                dbg<<"shoot rmin: "<<shoot_rmin<<std::endl;
                dbg<<"shoot rmax: "<<shoot_rmax<<std::endl;
                dbg<<"knur: "<<knur<<std::endl;
                dbg<<"b: "<<b<<std::endl;
                dbg<<"a: "<<a<<std::endl;
                dbg<<"a+b*rmin:"<<a+b*shoot_rmin<<std::endl;
                std::vector<double> range(3,0.);
                range[1] = shoot_rmin;
                range[2] = shoot_rmax;
                _radial.reset(new SpergelNuNegativeRadialFunction(_nu, shoot_rmin, a, b));
                double nominal_flux = 2.*M_PI*std::pow(2.,_nu)*_gamma_nup1;
                _sampler.reset(new OneDimensionalDeviate(*_radial, range, true, nominal_flux,
                                                         *_gsparams));
            }
        }

        assert(_sampler.get());
        _sampler->shoot(photons,ud);
        dbg<<"SpergelInfo Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    void SBSpergel::SBSpergelImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        dbg<<"Spergel shoot: N = "<<photons.size()<<std::endl;
        // Get photons from the SpergelInfo structure, rescale flux and size for this instance
        _info->shoot(photons,ud);
        photons.scaleFlux(_shootnorm);
        photons.scaleXY(_r0);
        dbg<<"Spergel Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }
}
