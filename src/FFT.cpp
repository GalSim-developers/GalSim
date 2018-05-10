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
// Routines for FFTW interface objects
// This time to use Version 3 of FFTW.

//#define DEBUGLOGGING

#include <limits>
#include <vector>
#include <cassert>
#include "FFT.h"
#include "Std.h"

#ifdef __SSE2__
#include "xmmintrin.h"
#endif

namespace galsim {

    template <typename T>
    void FFTW_Array<T>::resize(size_t n)
    {
        if (_n != n) {
            _n = n;
            // cf. BaseImage::allocateMem, which uses the same code.
            char* mem = new char[_n * sizeof(T) + sizeof(char*) + 15];
            _p = reinterpret_cast<T*>( (uintptr_t)(mem + sizeof(char*) + 15) & ~(size_t) 0x0F );
            ((char**)_p)[-1] = mem;
        }
    }

    template <typename T>
    FFTW_Array<T>::~FFTW_Array()
    {
        if (_p) {
            delete [] ((char**)_p)[-1];
        }
    }

    KTable::KTable(int N, double dk, std::complex<double> value) : _dk(dk), _invdk(1./dk)
    {
        if (N<=0) throw FFTError("KTable size <=0");
        _N = ((N+1)>>1)<<1; //Round size up to even.
        _No2 = _N>>1;
        _Nd = _N;
        _halfNd = 0.5*_Nd;
        _invNd = 1./_Nd;
        _array.resize(_N*(_No2+1));
        _array.fill(value);
    }

    std::complex<double> KTable::kval(int ix, int iy) const
    {
        check_array();
        std::complex<double> retval=_array[index(ix,iy)];
        if (ix<0) return conj(retval);
        else return retval;
    }

    void KTable::kSet(int ix, int iy, std::complex<double> value)
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        if (ix<0) {
            _array[index(ix,iy)]=conj(value);
            if (ix==-_No2) _array[index(ix,-iy)]=value;
        } else {
            _array[index(ix,iy)]=value;
            if (ix==0 || ix==_No2) _array[index(ix,-iy)]=conj(value);
        }
    }
    void KTable::clear()
    {
        clearCache(); // invalidate any stored interpolations
        _array.fill(0.);
    }

    void KTable::accumulate(const KTable& rhs, double scalar)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        if (_N != rhs._N) throw FFTError("KTable::accumulate() with mismatched sizes");
        if (_dk != rhs._dk) throw FFTError("KTable::accumulate() with mismatched dk");
        for (int i=0; i<_N*(_No2+1); ++i)
            _array[i] += scalar * rhs._array[i];
    }

    void KTable::operator*=(const KTable& rhs)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        if (_N != rhs._N) throw FFTError("KTable::operator*=() with mismatched sizes");
        if (_dk != rhs._dk) throw FFTError("KTable::operator*=() with mismatched dk");
        for (int i=0; i<_N*(_No2+1); ++i)
            _array[i] *= rhs._array[i];
    }

    void KTable::operator*=(double scale)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        for (int i=0; i<_N*(_No2+1); ++i)
            _array[i] *= scale;
    }

    shared_ptr<KTable> KTable::wrap(int Nout) const
    {
        dbg<<"Start KTable wrap: N = "<<_N<<", Nout = "<<Nout<<std::endl;
        // MJ: I found that when using this routing with N not being a multiple of Nout, that
        //     the output image could lose some symmetries.  e.g. the test_flip tests in
        //     test_transform.py could fail when wrapping was involved.  Now the fourierDraw
        //     routing in SBProfile.cpp enforces that N = k Nout, and things seem to be
        //     working fine.  But beware that there may be a subtle bug, probably in the
        //     handling of the N/2 column or row, when the wrap starts and ends somewhere in
        //     the middle of the output grid.
        //     Update: Not anymore. Now that I switched it to only wrap once to include both
        //     the wrapping and the Hermitian edges, it works just fine without being a multiple
        //     of N.
        if (Nout < 0) FormatAndThrow<FFTError>() << "KTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        int Nouto2 = Nout>>1;
        shared_ptr<KTable> out(new KTable(Nout, _dk, std::complex<double>(0.,0.)));
        for (int iyin=-_No2; iyin<=_No2; ++iyin) {
            int iyout = iyin;
            while (iyout < -Nouto2) iyout += Nout;
            while (iyout >= Nouto2) iyout -= Nout;
            int ixin = 0;
            while (ixin <= _No2) {
                // number of points to accumulate without conjugation:
                // Do points that do *not* need to be conjugated:
                int nx = std::min(_No2-ixin+1, Nouto2+1);
                const std::complex<double>* inptr = _array.get() + index(ixin,iyin);
                std::complex<double>* outptr = out->_array.get() + out->index(0,iyout);
                for (int i=0; i<nx; ++i) {
                    *outptr += *inptr;
                    ++inptr;
                    ++outptr;
                }
                ixin += Nouto2;
                if (ixin > _No2) break;
                // Now do any points that *do* need conjugation
                // such that output storage locations go backwards
                inptr = _array.get() + index(ixin,iyin);
                outptr = out->_array.get() + out->index(Nouto2, -iyout);
                nx = std::min(_No2-ixin+1, Nouto2+1);
                for (int i=0; i<nx; ++i) {
                    *outptr += conj(*inptr);
                    ++inptr;
                    --outptr;
                }
                ixin += Nouto2;
            }
        }
        return out;
    }

    shared_ptr<XTable> XTable::wrap(int Nout) const
    {
        if (Nout < 0) FormatAndThrow<FFTError>() << "XTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        int Nouto2 = Nout>>1;
        shared_ptr<XTable> out(new XTable(Nout, _dx, 0.));
        // What is (-N/2) wrapped to (+- Nout/2)?
        int excess = (_N % Nout) / 2;  // Note N and Nout are positive.
        const int startOut = (excess==0) ? -Nouto2 : Nouto2 - excess;
        int iyout = startOut;
        for (int iyin=-_No2; iyin<_No2; ++iyin, ++iyout) {
            if (iyout >= Nouto2) iyout -= Nout;  // wrap y if needed
            int ixin = -_No2;
            int ixout = startOut;
            const double* inptr = _array.get() + index(ixin,iyin);
            while (ixin < _No2) {
                // number of points to write before wrapping:
                int nx = std::min(_No2-ixin, Nouto2-ixout);
                double* outptr = out->_array.get() + out->index(ixout,iyout);
                for (int i=0; i<nx; ++i) {
                    *outptr += *inptr;
                    ++inptr;
                    ++outptr;
                }
                ixin += nx;
                ixout += nx;
            }
        }
        return out;
    }

    // Wrap int(floor(k)) to a number from [-N/2..N/2).
    int KTable::wrapKValue(double k) const
    {
        k += _halfNd;
        return int(k-_Nd*std::floor(k*_invNd)) - _No2;
    }

    template <bool yn>
    struct Maybe // true
    {
        template <typename T>
        static inline void increment(T& p) { ++p; }
        template <typename T>
        static inline void increment(T& p, int n) { p += n; }

        template <typename T>
        static inline std::complex<T> conj(const std::complex<T>& x) { return std::conj(x); }

        template <typename T, typename T2>
        static inline T plus(const T& x, const T2& y) { return x+y; }
    };
    template <>
    struct Maybe<false>
    {
        template <typename T>
        static inline void increment(T& p) { --p; }
        template <typename T>
        static inline void increment(T& p, int n) { p -= n; }

        template <typename T>
        static inline std::complex<T> conj(const std::complex<T>& x) { return x; }

        template <typename T, typename T2>
        static inline T plus(const T& x, const T2& y) { return x-y; }
    };

    // A helper function for fast calculation of a dot product of real and complex vectors
    template <bool c2>
    static std::complex<double> ZDot(int n, const double* A, const std::complex<double>* B)
    {
        if (n) {
#ifdef __SSE2__
            std::complex<double> sum(0);
            while (n && !IsAligned(A) ) {
                sum += *A * *B;
                ++A;
                Maybe<!c2>::increment(B);
                --n;
            }

            int n_2 = (n>>1);
            int nb = n-(n_2<<1);

            if (n_2) {
                union { __m128d xm; double xd[2]; } xsum;
                xsum.xm = _mm_set1_pd(0.);
                __m128d xsum2 = _mm_set1_pd(0.);
                const std::complex<double>* B1 = Maybe<!c2>::plus(B,1);
                assert(IsAligned(A));
                assert(IsAligned(B));
                do {
                    const __m128d& xA = *(const __m128d*)(A);
                    const __m128d& xB1 = *(const __m128d*)(B);
                    const __m128d& xB2 = *(const __m128d*)(B1);
                    A += 2;
                    Maybe<!c2>::increment(B,2);
                    Maybe<!c2>::increment(B1,2);
                    __m128d xA1 = _mm_shuffle_pd(xA,xA,_MM_SHUFFLE2(0,0));
                    __m128d xA2 = _mm_shuffle_pd(xA,xA,_MM_SHUFFLE2(1,1));
                    __m128d x1 = _mm_mul_pd(xA1,xB1);
                    __m128d x2 = _mm_mul_pd(xA2,xB2);
                    xsum.xm = _mm_add_pd(xsum.xm,x1);
                    xsum2 = _mm_add_pd(xsum2,x2);
                } while (--n_2);
                xsum.xm = _mm_add_pd(xsum.xm,xsum2);
                sum += std::complex<double>(xsum.xd[0],xsum.xd[1]);
            }
            if (nb) {
                sum += *A * *B;
                ++A;
                Maybe<!c2>::increment(B);
            }
            return Maybe<c2>::conj(sum);
#else
            std::complex<double> sum = 0.;
            do {
                sum += *A * *B;
                ++A;
                Maybe<!c2>::increment(B);
            } while (--n);
            return Maybe<c2>::conj(sum);
#endif
        } else {
            return 0.;
        }
    }

    // Interpolate table to some specific k.  We WILL wrap the KTable to cover
    // entire interpolation kernel:
    std::complex<double> KTable::interpolate(
        double kx, double ky, const Interpolant2d& interp) const
    {
        dbg<<"Start KTable interpolate at "<<kx<<','<<ky<<std::endl;
        dbg<<"N = "<<_N<<std::endl;
        dbg<<"interp xrage = "<<interp.xrange()<<std::endl;
        kx *= _invdk;
        ky *= _invdk;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes()
             && std::abs(kx - std::floor(kx+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = wrapKValue(kx+0.01);
            ixMax = ixMin+1;
        } else if (interp.xrange() >= _No2) {
            // use all the elements in row:
            ixMin = -_No2;
            ixMax = _No2;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            ixMin = wrapKValue(kx-interp.xrange()+0.99);
            ixMax = -wrapKValue(-kx-interp.xrange()-0.01);
        }
        xassert(ixMin >= -_No2);
        xassert(ixMin < _No2);
        xassert(ixMax > -_No2);
        xassert(ixMax <= _No2);

        if ( interp.isExactAtNodes()
             && std::abs(ky - std::floor(ky+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = wrapKValue(ky+0.01);
            iyMax = iyMin+1;
        } else if (interp.xrange() >= _No2) {
            // use all the elements in row:
            iyMin = -_No2;
            iyMax = _No2;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            iyMin = wrapKValue(ky-interp.xrange()+0.99);
            iyMax = -wrapKValue(-ky-interp.xrange()-0.01);
        }
        xassert(iyMin >= -_No2);
        xassert(iyMin < _No2);
        xassert(iyMax > -_No2);
        xassert(iyMax <= _No2);
        xdbg<<"ix range = "<<ixMin<<"..."<<ixMax<<std::endl;
        xdbg<<"iy range = "<<iyMin<<"..."<<iyMax<<std::endl;

        std::complex<double> sum = 0.;
        const InterpolantXY* ixy = dynamic_cast<const InterpolantXY*> (&interp);
        if (ixy) {
            // Interpolant is seperable
            // We have the opportunity to speed up the calculation by
            // re-using the sums over rows.  So we will keep a
            // cache of them.
            if (kx != _cacheX || ixy != _cacheInterp) {
                clearCache();
                _cacheX = kx;
                _cacheInterp = ixy;
            } else if (iyMax==iyMin+1 && !_cache.empty()) {
                // Special case for interpolation on a single iy value:
                // See if we already have this row in cache:
                int index = iyMin - _cacheStartY;
                if (index < 0) index += _N;
                if (index < int(_cache.size()))
                    // We have it!
                    return _cache[index];
                else
                    // Desired row not in cache - kill cache, continue as normal.
                    // (But don't clear xwt, since that's still good.)
                    _cache.clear();
            }

            const bool simple_xval = ixy->xrange() <= _Nd;

            // Build the x component of interpolant
            int nx = ixMax - ixMin;
            if (nx<=0) nx += _N;
            xdbg<<"nx = "<<nx<<std::endl;
            // This is also cached if possible.  It gets cleared when kx != cacheX above.
            if (_xwt.empty()) {
                _xwt.resize(nx);
                int ix = ixMin;
                if (simple_xval) {
                    // Then simple xval is fine (and faster)
                    // Just need to keep ix-kx to [-N/2,N/2)
                    double arg = ix-kx;
                    if (std::abs(arg) >= _halfNd) arg -= _Nd*std::floor(arg*_invNd+0.5);
                    for (int i=0; i<nx; ++i, ++ix, ++arg) {
                        xdbg<<"Call xval for arg = "<<arg<<std::endl;
                        if (arg > _halfNd) arg -= _Nd;
                        _xwt[i] = ixy->xval1d(arg);
                        xdbg<<"xwt["<<i<<"] = "<<_xwt[i]<<std::endl;
                    }
                } else {
                    // Then might need to wrap to do the sum that's in xvalWrapped...
                    for (int i=0; i<nx; ++i, ++ix) {
                        xdbg<<"Call xvalWrapped1d for ix-kx = "<<ix<<" - "<<kx<<" = "<<
                            ix-kx<<std::endl;
                        _xwt[i] = ixy->xvalWrapped1d(ix-kx, _N);
                        xdbg<<"xwt["<<i<<"] = "<<_xwt[i]<<std::endl;
                    }
                }
            } else {
                assert(int(_xwt.size()) == nx);
            }

            // cache always holds sequential y values (with wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<std::complex<double> >::iterator nextSaved = _cache.begin();
            while (nextSaved != _cache.end() && _cacheStartY != iyMin) {
                _cache.pop_front();
                ++_cacheStartY;
                if (_cacheStartY >= _No2) _cacheStartY -= _N;
                nextSaved = _cache.begin();
            }

            // Accumulate sum of
            //    interp.xvalWrapped(ix-kx, iy-ky, N)*kval(ix,iy);
            // Which separates into
            //    ixy->xvalWrapped(ix-kx) * ixy->xvalWrapped(iy-ky) * kval(ix,iy)
            // The first factor is saved in xwt
            // The second factor is constant for a given iy, so do that at the end of the loop.
            // The third factor is the only one that needs to be computed for each ix,iy.
            int ny = iyMax - iyMin;
            if (ny<=0) ny += _N;
            int iy = iyMin;
            double arg = iy-ky;
            if (simple_xval && std::abs(arg) > _halfNd) {
                arg -= _Nd*std::floor(arg*_invNd+0.5);
            }
            for (; ny; --ny, ++iy, ++arg) {
                if (iy >= _No2) iy -= _N;   // wrap iy if needed
                xdbg<<"ny = "<<ny<<", iy = "<<iy<<std::endl;
                std::complex<double> sumy = 0.;
                if (nextSaved != _cache.end()) {
                    // This row is cached
                    sumy = *nextSaved;
                    ++nextSaved;
                } else {
                    // Need to compute a new row's sum
                    int ix = ixMin;
#if 0
                    // Simple loop preserved for comparison.
                    for (int i=0; i<nx; ++i, ++ix) {
                        if (ix > N/2) ix -= N; //check for wrap
                        sumy += _xwt[i]*kval(ix,iy);
                    }
#else

                    // Faster way using ptrs, which doesn't need to do index(ix,iy) every time.
                    int count = nx;
                    const double* xwt_it = &_xwt[0];
                    // First do any initial negative ix values:
                    if (ix < 0) {
                        xdbg<<"Some initial negative ix: ix = "<<ix<<std::endl;
                        int count1 = std::min(count, -ix);
                        xdbg<<"count1 = "<<count1<<std::endl;
                        count -= count1;
                        const std::complex<double>* ptr = _array.get() + index(ix,iy);
                        sumy += ZDot<true>(count1, xwt_it, ptr);
                        xwt_it += count1;
                        ix = 0;
                    }

                    // Next do positive ix values:
                    if (count) {
                        xdbg<<"Positive ix: ix = "<<ix<<std::endl;
                        const std::complex<double>* ptr = _array.get() + index(ix,iy);
                        int count1 = std::min(count, _No2+1-ix);
                        xdbg<<"count1 = "<<count1<<std::endl;
                        count -= count1;
                        sumy += ZDot<false>(count1, xwt_it, ptr);
                        xwt_it += count1;

                        // Finally if we've wrapped around again, do more negative ix values:
                        if (count) {
                            xdbg<<"More negative ix: ix = "<<ix<<std::endl;
                            xdbg<<"count = "<<count<<std::endl;
                            ix = -_No2 + 1;
                            const std::complex<double>* ptr = _array.get() + index(ix,iy);
                            xassert(count < _No2-1);
                            sumy += ZDot<true>(count, xwt_it, ptr);
                            //xwt_it += count;
                        }
                    }
                    //xassert(xwt_it == &_xwt[0] + _xwt.size());
#endif
                    // Add to back of cache
                    if (_cache.empty()) _cacheStartY = iy;
                    _cache.push_back(sumy);
                    nextSaved = _cache.end();
                }
                if (simple_xval) {
                    if (arg > _halfNd) arg -= _Nd;
                    xdbg<<"Call xval for arg = "<<arg<<std::endl;
                    sum += sumy * ixy->xval1d(arg);
                } else {
                    xdbg<<"Call xvalWrapped1d for iy-ky = "<<iy<<" - "<<ky<<" = "<<iy-ky<<std::endl;
                    sum += sumy * ixy->xvalWrapped1d(arg, _N);
                }
                xdbg<<"After multiply by column xvalWrapped: sum = "<<sum<<std::endl;
            }
        } else {
            // Interpolant is not seperable, calculate weight at each point
            int ny = iyMax - iyMin;
            if (ny<=0) ny += _N;
            int nx = ixMax - ixMin;
            if (nx<=0) nx += _N;
            int iy = iyMin;
            for (; ny; --ny, ++iy) {
                if (iy >= _No2) iy -= _N;   // wrap iy if needed
                int ix = ixMin;
                for (int i=nx; i; --i, ++ix) {
                    if (ix > _No2) ix -= _N; //check for wrap
                    // use kval to keep track of conjugations
                    sum += interp.xvalWrapped(ix-kx, iy-ky, _N)*kval(ix,iy);
                }
            }
        }
        xdbg<<"Done: sum = "<<sum<<std::endl;
        return sum;
    }

    // Fill table from a function:
    void KTable::fill(KTable::function1 func)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        std::complex<double>* zptr=_array.get();
        double kx, ky;
        std::vector<std::complex<double> > tmp1(_No2);
        std::vector<std::complex<double> > tmp2(_No2);

        // [ky/dk] = iy = 0
        for (int ix=0; ix<_No2+1; ++ix) {
            kx = ix*_dk;
            *(zptr++) = func(kx,0);                  // [kx/dk] = ix = 0 to N/2
        }
        // [ky/dk] = iy = 1 to (N/2-1)
        for (int iy=1; iy<_No2; ++iy) {
            ky = iy*_dk;
            *(zptr++) = tmp1[iy] = func(0,ky);        // [kx/dk] = ix = 0
            for (int ix=1; ix<_No2; ++ix) {
                kx = ix*_dk;
                *(zptr++) = func(kx,ky);               // [kx/dk] = ix = 1 to (N/2-1)
            }
            *(zptr++) = tmp2[iy] = func(_halfNd*_dk,ky); // [kx/dk] = ix =N/2
        }
        // Wrap to the negative ky's
        // [ky/dk] = iy = -N/2
        for (int ix=0; ix<_No2+1; ++ix) {
            kx = ix*_dk;
            *(zptr++) = func(kx,-_halfNd*_dk);         // [kx/dk] = ix = 0 to N/2
        }
        // [ky/dk] = iy = (-N/2+1) to (-1)
        for (int iy=-_No2+1; iy<0; ++iy) {
            ky = iy*_dk;
            *(zptr++) = conj(tmp1[-iy]);       // [kx/dk] = ix = 0
            for (int ix=1; ix<_No2; ++ix) {
                kx = ix*_dk;
                *(zptr++) = func(kx,ky);         // [kx/dk] = ix = 1 to (N/2-1)
            }
            *(zptr++) = conj(tmp2[-iy]);      // [kx/dk] = ix = N/2
        }
    }

    // Integrate a function over k - can be function of k or of PSF(k)
    std::complex<double> KTable::integrate(KTable::function2 func) const
    {
        check_array();
        std::complex<double> sum=0.;
        std::complex<double> val;
        double kx, ky;
        const std::complex<double>* zptr=_array.get();
        // Do the positive y frequencies
        for (int iy=0; iy<=_No2; ++iy) {
            ky = iy*_dk;
            val = *(zptr++);
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix<_No2; ++ix) {
                kx = ix*_dk;
                val = *(zptr++);
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = _halfNd*_dk;
            val = *(zptr++);
            sum += func(kx,ky,val); // x Nyquist freq
        }

        // wrap to the negative ky's
        for (int iy=-_No2+1; iy<0; ++iy) {
            ky = iy*_dk;
            val = *(zptr++);
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix<_No2; ++ix) {
                kx = ix*_dk;
                val = *(zptr++);
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = _halfNd*_dk;
            val = *(zptr++);
            sum += func(kx,ky,val); // x Nyquist
        }
        sum *= _dk*_dk;
        return sum;
    }

    // Integrate KTable over d^2k (sum of all pixels * dk * dk)
    std::complex<double> KTable::integratePixels() const
    {
        check_array();
        std::complex<double> sum=0.;
        const std::complex<double>* zptr=_array.get();
        // Do the positive y frequencies
        for (int iy=0; iy<=_No2; ++iy) {
            sum += *(zptr++);    // x DC term
            for (int ix=1; ix<_No2; ++ix) {
                sum += *(zptr);
                sum += conj(*(zptr++));
            }
            sum += *(zptr++);
        }
        // wrap to the negative ky's
        for (int iy=-_No2+1; iy<0; ++iy) {
            sum += *(zptr++);    // x DC term
            for (int ix=1; ix<_No2; ++ix) {
                sum += *(zptr);
                sum += conj(*(zptr++));
            }
            sum += *(zptr++);
        }
        sum *= _dk*_dk;
        return sum;
    }

    // Make a new table that is function of old.
    shared_ptr<KTable> KTable::function(KTable::function2 func) const
    {
        check_array();
        shared_ptr<KTable> lhs(new KTable(_N,_dk));
        std::complex<double> val;
        double kx, ky;
        const std::complex<double>* zptr=_array.get();
        std::complex<double>* lptr=lhs->_array.get();
        // Do the positive y frequencies
        for (int iy=0; iy<_No2; ++iy) {
            ky = iy*_dk;
            for (int ix=0; ix<=_No2; ++ix) {
                kx = ix*_dk;
                val = *(zptr++);
                *(lptr++)= func(kx,ky,val);
            }
        }
        // wrap to the negative ky's
        for (int iy=-_No2; iy<0; ++iy) {
            ky = iy*_dk;
            for (int ix=0; ix<=_No2; ++ix) {
                kx = ix*_dk;
                val = *(zptr++);
                *(lptr++)= func(kx,ky,val);
            }
        }
        return lhs;
    }

    // Transform to a single x point:
    // assumes (x,y) in physical units
    double KTable::xval(double x, double y) const
    {
        check_array();
        x*=_dk; y*=_dk;
        // Don't evaluate if x not in fundamental period +-PI/dk:
#ifdef FFT_DEBUG
        if (std::abs(x) > M_PI || std::abs(y) > M_PI)
            throw FFTOutofRange(" (x,y) too big in xval()");
#endif
        std::complex<double> dxphase=std::polar(1.,x);
        std::complex<double> dyphase=std::polar(1.,y);
        double sum=0.;
        // y DC terms first:
        const std::complex<double>* zptr=_array.get();
        // Do the positive y frequencies
        std::complex<double> yphase=1.;
        std::complex<double> phase,z;
        for (int iy=0; iy<_No2; ++iy) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); //x DC term
            for (int ix=1; ix<_No2; ++ix) {
                phase *= dxphase;
                z= *(zptr++);
                sum += (phase*z).real() * 2.;
            }
            phase *= dxphase; //ix=N/2 has no mirror:
            z= *(zptr++);
            sum += (phase*z).real();
            yphase *= dyphase;
        }

        // wrap to the negative ky's
        yphase = std::polar(1.,y*(-_halfNd));
        for (int iy=-_No2; iy<0; ++iy) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); // x DC term
            for (int ix=1; ix<_No2; ++ix) {
                phase *= dxphase;
                z= *(zptr++);
                sum += (phase*z).real() * 2.;
            }
            phase *= dxphase; //ix=N/2 has no mirror:
            z= *(zptr++);
            sum += (phase*z).real();
            yphase *= dyphase;
        }

        sum *= _dk*_dk/(4.*M_PI*M_PI); //inverse xform has 2pi in it.
        return sum;
    }

    // Translate the PSF to be for source at (x0,y0);
    void KTable::translate(double x0, double y0)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        // convert to phases:
        x0*=_dk; y0*=_dk;
        // too big will just be wrapping around:
        if (x0 > M_PI || y0 > M_PI) throw FFTOutofRange("(x0,y0) too big in translate()");
        std::complex<double> dxphase=std::polar(1.,-x0);
        std::complex<double> dyphase=std::polar(1.,-y0);
        std::complex<double> yphase=1.;
        std::complex<double>* zptr=_array.get();

        std::complex<double> phase,z;
        for (int iy=0; iy<_No2; ++iy) {
            phase = yphase;
            for (int ix=0; ix<=_No2; ++ix) {
                z = *zptr;
                *zptr = phase * z;
                phase *= dxphase;
                ++zptr;
            }
            yphase *= dyphase;
        }

        // wrap to the negative ky's
        yphase = std::polar(1.,_halfNd*y0);
        for (int iy=-_No2; iy<0; ++iy) {
            phase = yphase;
            for (int ix=0; ix<=_No2; ++ix) {
                z = *zptr;
                *zptr = phase* z;
                phase *= dxphase;
                ++zptr;
            }
            yphase *= dyphase;
        }
    }

    XTable::XTable(int N, double dx, double value) : _dx(dx), _invdx(1./dx)
    {
        if (N<=0) throw FFTError("XTable size <=0");
        _N = ((N+1)>>1)<<1; //Round size up to even.
        _No2 = _N>>1;
        _Nd = _N;
        _halfNd = 0.5*_Nd;
        _invNd = 1./_Nd;
        _array.resize(_N*_N);
        _array.fill(value);
    }

    double XTable::xval(int ix, int iy) const
    {
        check_array();
        return _array[index(ix,iy)];
    }

    void XTable::xSet(int ix, int iy, double value)
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        _array[index(ix,iy)]=value;
    }

    void XTable::clear()
    {
        clearCache(); // invalidate any stored interpolations
        _array.fill(0.);
    }

    void XTable::accumulate(const XTable& rhs, double scalar)
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        if (_N != rhs._N) throw FFTError("XTable::accumulate() with mismatched sizes");
        const int Nsq = _N*_N;
        for (int i=0; i<Nsq; ++i)
            _array[i] += scalar * rhs._array[i];
    }

    void XTable::operator*=(double scale)
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        const int Nsq = _N*_N;
        for (int i=0; i<Nsq; ++i)
            _array[i] *= scale;
    }

    // Interpolate table (linearly) to some specific k:
    // x any y in physical units (to be divided by dx for indices)
    double XTable::interpolate(double x, double y, const Interpolant2d& interp) const
    {
        xdbg << "interpolating " << x << " " << y << " " << std::endl;
        x *= _invdx;
        y *= _invdx;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes()
             && std::abs(x - std::floor(x+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = ixMax = int(std::floor(x+0.01));
        } else {
            ixMin = int(std::ceil(x-interp.xrange()));
            ixMax = int(std::floor(x+interp.xrange()));
        }
        ixMin = std::max(ixMin, -_No2);
        ixMax = std::min(ixMax, _No2-1);
        if (ixMin > ixMax) return 0.;

        if ( interp.isExactAtNodes()
             && std::abs(y - std::floor(y+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = iyMax = int(std::floor(y+0.01));
        } else {
            iyMin = int(std::ceil(y-interp.xrange()));
            iyMax = int(std::floor(y+interp.xrange()));
        }
        iyMin = std::max(iyMin, -_No2);
        iyMax = std::min(iyMax, _No2-1);
        if (iyMin > iyMax) return 0.;

        double sum = 0.;
        const InterpolantXY* ixy = dynamic_cast<const InterpolantXY*> (&interp);
        if (ixy) {
            // Interpolant is seperable
            // We have the opportunity to speed up the calculation by
            // re-using the sums over rows.  So we will keep a
            // cache of them.
            if (x != _cacheX || ixy != _cacheInterp) {
                clearCache();
                _cacheX = x;
                _cacheInterp = ixy;
            } else if (iyMax==iyMin && !_cache.empty()) {
                // Special case for interpolation on a single iy value:
                // See if we already have this row in cache:
                int index = iyMin - _cacheStartY;
                if (index < 0) index += _N;
                if (index < int(_cache.size()))
                    // We have it!
                    return _cache[index];
                else
                    // Desired row not in cache - kill cache, continue as normal.
                    // (But don't clear xwt, since that's still good.)
                    _cache.clear();
            }

            // Build x factors for interpolant
            int nx = ixMax - ixMin + 1;
            // This is also cached if possible.  It gets cleared when kx != cacheX above.
            if (_xwt.empty()) {
                _xwt.resize(nx);
                for (int i=0; i<nx; ++i)
                    _xwt[i] = ixy->xval1d(i+ixMin-x);
            } else {
                assert(int(_xwt.size()) == nx);
            }

            // cache always holds sequential y values (no wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<double>::iterator nextSaved = _cache.begin();
            while (nextSaved != _cache.end() && _cacheStartY != iyMin) {
                _cache.pop_front();
                ++_cacheStartY;
                nextSaved = _cache.begin();
            }

            for (int iy=iyMin; iy<=iyMax; ++iy) {
                double sumy = 0.;
                if (nextSaved != _cache.end()) {
                    // This row is cached
                    sumy = *nextSaved;
                    ++nextSaved;
                } else {
                    // Need to compute a new row's sum
                    const double* dptr = _array.get() + index(ixMin, iy);
                    std::vector<double>::const_iterator xwt_it = _xwt.begin();
                    int count = nx;
                    for(; count; --count) sumy += (*xwt_it++) * (*dptr++);
                    xassert(xwt_it == _xwt.end());
                    // Add to back of cache
                    if (_cache.empty()) _cacheStartY = iy;
                    _cache.push_back(sumy);
                    nextSaved = _cache.end();
                }
                sum += sumy * ixy->xval1d(iy-y);
            }
        } else {
            // Interpolant is not seperable, calculate weight at each point
            for (int iy=iyMin; iy<=iyMax; ++iy) {
                const double* dptr = _array.get() + index(ixMin, iy);
                for (int ix=ixMin; ix<=ixMax; ++ix, ++dptr)
                    sum += *dptr * interp.xval(ix-x, iy-y);
            }
        }
        return sum;
    }

    // Fill table from a function:
    void XTable::fill(XTable::function1 func)
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        double* zptr=_array.get();
        double x, y;
        for (int iy=0; iy<_N; ++iy) {
            y = (iy-_No2)*_dx;
            for (int ix=0; ix<_N; ++ix) {
                x = (ix-_No2)*_dx;
                *(zptr++) = func(x,y);
            }
        }
    }

    // Integrate a function over x - can be function of x or of PSF(x)
    // Setting the Boolean flag gives sum over samples, not integral.
    double XTable::integrate(XTable::function2 func, bool  sumonly) const
    {
        check_array();
        double sum=0.;
        double val;
        double x, y;
        const double* zptr=_array.get();

        for (int iy=0; iy<_N; ++iy) {
            y = (iy-_No2)*_dx;
            for (int ix=0; ix<_N; ++ix) {
                x = (ix-_No2)*_dx;
                val = *(zptr++);
                sum += func(x,y,val);
            }
        }

        if (!sumonly) sum *= _dx*_dx;
        return sum;
    }

    double XTable::integratePixels() const
    {
        check_array();
        double sum=0.;
        const double* zptr=_array.get();
        for (int iy=-_No2; iy<_No2; ++iy)
            for (int ix=-_N/2; ix<_No2; ++ix) {
                sum += *(zptr++);
            }
        sum *= _dx*_dx;
        return (double) sum;
    }

    // Transform to a single k point:
    std::complex<double> XTable::kval(double kx, double ky) const
    {
        check_array();
        // Don't evaluate if k not in fundamental period
        kx *= _dx;
        ky *= _dx;
#ifdef FFT_DEBUG
        if (std::abs(kx) > M_PI || std::abs(ky) > M_PI)
            throw FFTOutofRange("XTable::kval() args out of range");
#endif
        std::complex<double> dxphase=std::polar(1.,-kx);
        std::complex<double> dyphase=std::polar(1.,-ky);
        std::complex<double> sum=0.;

        const double* zptr=_array.get();
        std::complex<double> yphase=std::polar(1.,_halfNd*kx);
        std::complex<double> xphase=std::polar(1.,_halfNd*kx);
        std::complex<double> phase;
        for (int iy=0; iy<_N; ++iy) {
            phase = yphase * xphase;
            for (int ix=0; ix<_N; ++ix) {
                sum += phase* (*(zptr++));
                phase *= dxphase;
            }
            yphase *= dyphase;
        }
        sum *= _dx*_dx;
        return sum;
    }

    // Have FFTW develop "wisdom" on doing this kind of transform
    void KTable::fftwMeasure() const
    {
        // Copy data into new array to avoid NaN's, etc., but not bothering
        // with scaling, etc.
        FFTW_Array<std::complex<double> > t_array = _array;

        XTable xt( _N, 2.*M_PI*_invNd*_invdk );

        // Note: The fftw_execute function is the only thread-safe FFTW routine.
        // So if we decide to go with some kind of multi-threading (rather than multi-process
        // parallelism) all of the plan creation and destruction calls in this file
        // will need to be placed in critical blocks or the equivalent (mutex locks, etc.).
        fftw_plan plan = fftw_plan_dft_c2r_2d(
            _N, _N, t_array.get_fftw(), xt._array.get_fftw(), FFTW_MEASURE);
        if (plan==NULL) throw FFTInvalid();
        fftw_destroy_plan(plan);
    }

    // Fourier transform from (complex) k to x:
    // This version takes XTable reference as argument
    void KTable::transform(XTable& xt) const
    {
        dbg<<"Start transform K -> X\n";
        dbg<<"N = "<<_N<<std::endl;
        dbg<<"dk = "<<_dk<<std::endl;
        check_array();
        dbg<<"flux = "<<_array[0]<<std::endl;

        // check proper dimensions for xt
        assert(_N==xt.getN());

        // We'll need a new k array because FFTW kills the k array in this
        // operation.  Also, to put x=0 in center of array, we need to flop
        // every other sign of k array, and need to scale.
        FFTW_Array<std::complex<double> > t_array(_N*(_No2+1));
        double fac = _dk * _dk / (4*M_PI*M_PI);
        xdbg<<"fac = "<<fac<<std::endl;
        long int ind=0;
        xdbg<<"t_array.size = "<<t_array.size()<<std::endl;
        for (int iy=0; iy<_N; ++iy) {
            for (int ix=0; ix<=_No2; ++ix) {
                if ( (ix+iy)%2==0) t_array[ind]=fac * _array[ind];
                else t_array[ind] = -fac* _array[ind];
                ++ind;
            }
        }
        xdbg<<"After fill t_array, t_array[0] = "<<t_array[0]<<std::endl;

        fftw_plan plan = fftw_plan_dft_c2r_2d(
            _N, _N, t_array.get_fftw(), xt._array.get_fftw(), FFTW_ESTIMATE);
        xdbg<<"After make plan"<<std::endl;
        if (plan==NULL) throw FFTInvalid();

        // Run the transform:
        fftw_execute(plan);
        xdbg<<"After exec plan"<<std::endl;
        fftw_destroy_plan(plan);
        xdbg<<"After destroy plan"<<std::endl;

        xt._dx = 2.*M_PI*_invNd*_invdk;
        dbg<<"dx = "<<xt._dx<<std::endl;
        dbg<<"Done transform"<<std::endl;
    }

    // Same thing, but return a new XTable
    shared_ptr<XTable> KTable::transform() const
    {
        shared_ptr<XTable> xt(new XTable( _N, 2.*M_PI*_invNd*_invdk ));
        transform(*xt);
        return xt;
    }

    void XTable::fftwMeasure() const
    {
        // Make a new copy of data array since measurement will overwrite:
        // Copy data into new array to avoid NaN's, etc., but not bothering
        // with scaling, etc.
        FFTW_Array<double> t_array = _array;

        KTable kt( _N, 2.*M_PI*_invNd*_invdx );

        fftw_plan plan = fftw_plan_dft_r2c_2d(
            _N,_N, t_array.get_fftw(), kt._array.get_fftw(), FFTW_MEASURE);
        if (plan==NULL) throw FFTInvalid();

        fftw_destroy_plan(plan);
    }

    // Fourier transform from x back to (complex) k:
    void XTable::transform(KTable& kt) const
    {
        check_array();

        // Make a new copy of data array since measurement will overwrite:
        FFTW_Array<double> t_array = _array;

        fftw_plan plan = fftw_plan_dft_r2c_2d(
            _N,_N, t_array.get_fftw(), kt._array.get_fftw(), FFTW_ESTIMATE);
        if (plan==NULL) throw FFTInvalid();
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Now scale the k spectrum and flip signs for x=0 in middle.
        double fac = _dx * _dx;
        size_t ind=0;
        for (int iy=0; iy<_N; ++iy) {
            for (int ix=0; ix<=_N/2; ++ix) {
                if ( (ix+iy)%2==0) kt._array[ind] *= fac;
                else kt._array[ind] *= -fac;
                ++ind;
            }
        }
        kt._dk = 2.*M_PI*_invNd*_invdx;
    }

    // Same thing, but return a new KTable
    shared_ptr<KTable> XTable::transform() const
    {
        shared_ptr<KTable> kt(new KTable( _N, 2.*M_PI*_invNd*_invdx ));
        transform(*kt);
        return kt;
    }

    template class FFTW_Array<double>;
    template class FFTW_Array<std::complex<double> >;

}
