// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
// Routines for FFTW interface objects
// This time to use Version 3 of FFTW.

//#define DEBUGLOGGING

#include <limits>
#include <vector>
#include <cassert>
#include "FFT.h"
#include "Std.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    // A helper function that will return the smallest 2^n or 3x2^n value that is
    // even and >= the input integer.
    int goodFFTSize(int input) 
    {
        if (input<=2) return 2;
        // Reduce slightly to eliminate potential rounding errors:
        double insize = (1.-1.e-5)*input;
        double log2n = std::log(2.)*std::ceil(std::log(insize)/std::log(2.));
        double log2n3 = std::log(3.) 
           + std::log(2.)*std::ceil((std::log(insize)-std::log(3.))/std::log(2.));
        log2n3 = std::max(log2n3, std::log(6.)); // must be even number
        int Nk = int(std::ceil(std::exp(std::min(log2n, log2n3))-1.e-5));
        return Nk;
    }

    KTable::KTable(int N, double dk, std::complex<double> value) : _dk(dk)
    {
        if (N<=0) throw FFTError("KTable size <=0");
        _N = 2*((N+1)/2); //Round size up to even.
        _array.resize(_N);
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
            if (ix==-_N/2) _array[index(ix,-iy)]=value;
        } else {
            _array[index(ix,iy)]=value;
            if (ix==0 || ix==_N/2) _array[index(ix,-iy)]=conj(value);
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
        for (int i=0; i<_N*(_N/2+1); i++)
            _array[i] += scalar * rhs._array[i];
    }

    void KTable::operator*=(const KTable& rhs) 
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        if (_N != rhs._N) throw FFTError("KTable::operator*=() with mismatched sizes");
        if (_dk != rhs._dk) throw FFTError("KTable::operator*=() with mismatched dk");
        for (int i=0; i<_N*(_N/2+1); i++)
            _array[i] *= rhs._array[i];
    }

    void KTable::operator*=(double scale)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        for (int i=0; i<_N*(_N/2+1); i++)
            _array[i] *= scale;
    }

    boost::shared_ptr<KTable> KTable::wrap(int Nout) const 
    {
        if (Nout < 0) FormatAndThrow<FFTError>() << "KTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        boost::shared_ptr<KTable> out(new KTable(Nout, _dk, std::complex<double>(0.,0.)));
        for (int iyin=-_N/2; iyin<_N/2; iyin++) {
            int iyout = iyin;
            while (iyout < -Nout/2) iyout+=Nout;
            while (iyout >= Nout/2) iyout-=Nout;
            int ixin = 0;
            while (ixin < _N/2) {
                // number of points to accumulate without conjugation:
                // Do points that do *not* need to be conjugated:
                int nx = std::min(_N/2-ixin+1, Nout/2+1);
                const std::complex<double>* inptr = _array.get() + index(ixin,iyin);
                std::complex<double>* outptr = out->_array.get() + out->index(0,iyout);
                for (int i=0; i<nx; i++) {
                    *outptr += *inptr;
                    inptr++;
                    outptr++;
                }
                ixin += Nout/2;
                if (ixin >= _N/2) break;
                // Now do any points that *do* need conjugation
                // such that output storage locations go backwards
                inptr = _array.get() + index(ixin,iyin);
                outptr = out->_array.get() + out->index(Nout/2, -iyout);
                nx = std::min(_N/2-ixin+1, Nout/2+1);
                for (int i=0; i<nx; i++) {
                    *outptr += conj(*inptr);
                    inptr++;
                    outptr--;
                }
                ixin += Nout/2;
            }
        }
        return out;
    }

    boost::shared_ptr<XTable> XTable::wrap(int Nout) const 
    {
        if (Nout < 0) FormatAndThrow<FFTError>() << "XTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        boost::shared_ptr<XTable> out(new XTable(Nout, _dx, 0.));
        // What is (-N/2) wrapped to (+- Nout/2)?
        int excess = (_N % Nout) / 2;  // Note N and Nout are positive.
        const int startOut = (excess==0) ? -Nout/2 : Nout/2 - excess;
        int iyout = startOut;
        for (int iyin=-_N/2; iyin<_N/2; iyin++, iyout++) {
            if (iyout >= Nout/2) iyout -= Nout;  // wrap y if needed
            int ixin = -_N/2;
            int ixout = startOut;
            const double* inptr = _array.get() + index(ixin,iyin);
            while (ixin < _N/2) {
                // number of points to write before wrapping:
                int nx = std::min(_N/2-ixin, Nout/2-ixout);
                double* outptr = out->_array.get() + out->index(ixout,iyout);
                for (int i=0; i<nx; i++) {
                    *outptr += *inptr;
                    inptr++;
                    outptr++;
                }
                ixin += nx;
                ixout += nx;
            }
        }
        return out;
    }

    // Wrap int(floor(x)) to a number from [-N/2..N/2).
    inline int Wrap(double x, int N)
    {
        x += N/2.;
        return int(x-N*std::floor(x/N)) - (N>>1);
    }

    // Interpolate table to some specific k.  We WILL wrap the KTable to cover
    // entire interpolation kernel:
    std::complex<double> KTable::interpolate(
        double kx, double ky, const Interpolant2d& interp) const 
    {
        dbg<<"Start KTable interpolate at "<<kx<<','<<ky<<std::endl;
        dbg<<"N = "<<_N<<std::endl;
        const int No2 = _N>>1;  // == _N/2
        dbg<<"interp xrage = "<<interp.xrange()<<std::endl;
        kx /= _dk;
        ky /= _dk;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes() 
             && std::abs(kx - std::floor(kx+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = Wrap(kx+0.01, _N);
            ixMax = ixMin+1;
        } else if (interp.xrange() >= No2) {
            // use all the elements in row:
            ixMin = -No2;
            ixMax = No2;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            ixMin = Wrap(kx-interp.xrange()+0.99, _N);
            ixMax = -Wrap(-kx-interp.xrange()-0.01, _N);
        }
        xassert(ixMin >= -No2);
        xassert(ixMin < No2);
        xassert(ixMax > -No2);
        xassert(ixMax <= No2);

        if ( interp.isExactAtNodes() 
             && std::abs(ky - std::floor(ky+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = Wrap(ky+0.01, _N);
            iyMax = iyMin+1;
        } else if (interp.xrange() >= No2) {
            // use all the elements in row:
            iyMin = -No2;
            iyMax = No2;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            iyMin = Wrap(ky-interp.xrange()+0.99, _N);
            iyMax = -Wrap(-ky-interp.xrange()-0.01, _N);
        }
        xassert(iyMin >= -No2);
        xassert(iyMin < No2);
        xassert(iyMax > -No2);
        xassert(iyMax <= No2);
        dbg<<"ix range = "<<ixMin<<"..."<<ixMax<<std::endl;
        dbg<<"iy range = "<<iyMin<<"..."<<iyMax<<std::endl;

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

            const bool simple_xval = ixy->xrange() <= _N;

            // Build the x component of interpolant
            int nx = ixMax - ixMin;
            if (nx<=0) nx+=_N;
            dbg<<"nx = "<<nx<<std::endl;
            // This is also cached if possible.  It gets cleared when kx != cacheX above.
            if (_xwt.empty()) {
                _xwt.resize(nx);
                int ix = ixMin;
                if (simple_xval) {
                    // Then simple xval is fine (and faster)
                    // Just need to keep ix-kx to [-N/2,N/2)
                    double arg = ix-kx;
                    arg = arg-_N*std::floor(arg/_N+0.5);
                    for (int i=0; i<nx; ++i, ++ix, arg+=1.) {
                        dbg<<"Call xval for arg = "<<arg<<std::endl;
                        if (arg > _N/2.) arg -= _N;
                        _xwt[i] = ixy->xval1d(arg);
                        dbg<<"xwt["<<i<<"] = "<<_xwt[i]<<std::endl;
                    }
                } else {
                    // Then might need to wrap do the sum that's in xvalWrapped...
                    for (int i=0; i<nx; ++i, ++ix) {
                        dbg<<"Call xvalWrapped1d for ix-kx = "<<ix<<" - "<<kx<<" = "<<
                            ix-kx<<std::endl;
                        _xwt[i] = ixy->xvalWrapped1d(ix-kx, _N);
                        dbg<<"xwt["<<i<<"] = "<<_xwt[i]<<std::endl;
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
                _cacheStartY++;
                if (_cacheStartY >= No2) _cacheStartY-= _N;
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
            if (ny<=0) ny+=_N;
            int iy = iyMin;
            double arg = iy-ky;
            if (simple_xval) {
                arg = arg-_N*std::floor(arg/_N+0.5);
            }
            for (int j=0; j<ny; j++, iy++, arg+=1.) {
                if (iy >= No2) iy-=_N;   // wrap iy if needed
                dbg<<"j = "<<j<<", iy = "<<iy<<std::endl;
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
                    for (int i=0; i<nx; i++, ix++) {
                        if (ix > N/2) ix-=N; //check for wrap
                        dbg<<"i = "<<i<<", ix = "<<ix<<std::endl;
                        dbg<<"xwt = "<<_xwt[i]<<", kval = "<<kval(ix,iy)<<std::endl;
                        sumy += _xwt[i]*kval(ix,iy);
                        dbg<<"index = "<<index(ix,iy)<<", sumy -> "<<sumy<<std::endl;
                    }
#else
                    // Faster way using ptrs, which doesn't need to do index(ix,iy) every time.
                    int count = nx;
                    std::vector<double>::const_iterator xwt_it = _xwt.begin();
                    // First do any initial negative ix values:
                    if (ix < 0) {
                        dbg<<"Some initial negative ix: ix = "<<ix<<std::endl;
                        const std::complex<double>* ptr = _array.get() + index(ix,iy);
                        int count1 = std::min(count, -ix);
                        dbg<<"count1 = "<<count1<<std::endl;
                        count -= count1;
                        // Note: ptr goes down in this loop, since ix is negative.
                        for(; count1; --count1) sumy += (*xwt_it++) * conj(*ptr--);
                        ix = 0;
                    }

                    // Next do positive ix values:
                    if (count) {
                        dbg<<"Positive ix: ix = "<<ix<<std::endl;
                        const std::complex<double>* ptr = _array.get() + index(ix,iy);
                        int count1 = std::min(count, No2+1-ix);
                        dbg<<"count1 = "<<count1<<std::endl;
                        count -= count1;
                        for(; count1; --count1) sumy += (*xwt_it++) * (*ptr++);

                        // Finally if we've wrapped around again, do more negative ix values:
                        if (count) {
                            dbg<<"More negative ix: ix = "<<ix<<std::endl;
                            dbg<<"count = "<<count<<std::endl;
                            ix = -No2 + 1;
                            const std::complex<double>* ptr = _array.get() + index(ix,iy);
                            xassert(count < No2-1);
                            for(; count; --count) sumy += (*xwt_it++) * conj(*ptr--);
                        }
                    }
                    xassert(xwt_it == _xwt.end());
                    xassert(count == 0);
#endif
                    // Add to back of cache
                    if (_cache.empty()) _cacheStartY = iy;
                    _cache.push_back(sumy);
                    nextSaved = _cache.end();
                }
                if (simple_xval) {
                    if (arg > _N/2.) arg -= _N;
                    dbg<<"Call xval for arg = "<<arg<<std::endl;
                    sum += sumy * ixy->xval1d(arg);
                } else {
                    dbg<<"Call xvalWrapped1d for iy-ky = "<<iy<<" - "<<ky<<" = "<<iy-ky<<std::endl;
                    sum += sumy * ixy->xvalWrapped1d(arg, _N);
                }
                dbg<<"After multiply by column xvalWrapped: sum = "<<sum<<std::endl;
            }
        } else {
            // Interpolant is not seperable, calculate weight at each point
            int ny = iyMax - iyMin;
            if (ny<=0) ny+=_N;
            int nx = ixMax - ixMin;
            if (nx<=0) nx+=_N;
            int iy = iyMin;
            for (int j=0; j<ny; j++, iy++) {
                if (iy >= No2) iy-=_N;   // wrap iy if needed
                int ix = ixMin;
                for (int i=0; i<nx; i++, ix++) {
                    if (ix > No2) ix-=_N; //check for wrap
                    // use kval to keep track of conjugations
                    sum += interp.xvalWrapped(ix-kx, iy-ky, _N)*kval(ix,iy);
                }
            }
        }
        dbg<<"Done: sum = "<<sum<<std::endl;
        return sum;
    }

    // Fill table from a function:
    void KTable::fill(KTable::function1 func)
    {
        clearCache(); // invalidate any stored interpolations
        check_array();
        std::complex<double>* zptr=_array.get();
        double kx, ky;
        std::vector<std::complex<double> > tmp1(_N/2);
        std::vector<std::complex<double> > tmp2(_N/2);

        // [ky/dk] = iy = 0
        for (int ix=0; ix< _N/2+1 ; ix++) {
            kx = ix*_dk;
            *(zptr++) = func(kx,0);                  // [kx/dk] = ix = 0 to N/2
        }
        // [ky/dk] = iy = 1 to (N/2-1)
        for (int iy=1; iy< _N/2; iy++) {
            ky = iy*_dk;
            *(zptr++) = tmp1[iy] = func(0,ky);        // [kx/dk] = ix = 0
            for (int ix=1; ix< _N/2 ; ix++) {    
                kx = ix*_dk;
                *(zptr++) = func(kx,ky);               // [kx/dk] = ix = 1 to (N/2-1)
            }
            *(zptr++) = tmp2[iy] = func((_N/2.)*_dk,ky); // [kx/dk] = ix =N/2
        }
        // Wrap to the negative ky's
        // [ky/dk] = iy = -N/2
        for (int ix=0; ix< _N/2+1 ; ix++) {
            kx = ix*_dk;
            *(zptr++) = func(kx,-_N/2.*_dk);         // [kx/dk] = ix = 0 to N/2   
        }
        // [ky/dk] = iy = (-N/2+1) to (-1)
        for (int iy=-_N/2+1; iy< 0; iy++) {
            ky = iy*_dk;
            *(zptr++) = conj(tmp1[-iy]);       // [kx/dk] = ix = 0
            for (int ix=1; ix< _N/2 ; ix++) {
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
        for (int iy=0; iy<= _N/2; iy++) {
            ky = iy*_dk;
            val = *(zptr++);
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
                kx = ix*_dk;
                val = *(zptr++);
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = _dk*_N/2.;
            val = *(zptr++);
            sum += func(kx,ky,val); // x Nyquist freq
        }

        // wrap to the negative ky's
        for (int iy=-_N/2+1; iy< 0; iy++) {
            ky = iy*_dk;
            val = *(zptr++);
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
                kx = ix*_dk;
                val = *(zptr++);
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = _dk*_N/2.;
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
        for (int iy=0; iy<= _N/2; iy++) {
            sum += *(zptr++);    // x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
                sum += *(zptr);
                sum += conj(*(zptr++));
            }
            sum += *(zptr++);
        }
        // wrap to the negative ky's
        for (int iy=-_N/2+1; iy< 0; iy++) {
            sum += *(zptr++);    // x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
                sum += *(zptr);
                sum += conj(*(zptr++));
            }
            sum += *(zptr++);
        }
        sum *= _dk*_dk;
        return sum;
    }

    // Make a new table that is function of old.
    boost::shared_ptr<KTable> KTable::function(KTable::function2 func) const 
    {
        check_array();
        boost::shared_ptr<KTable> lhs(new KTable(_N,_dk));
        std::complex<double> val;
        double kx, ky;
        const std::complex<double>* zptr=_array.get();
        std::complex<double>* lptr=lhs->_array.get();
        // Do the positive y frequencies
        for (int iy=0; iy< _N/2; iy++) {
            ky = iy*_dk;
            for (int ix=0; ix<= _N/2 ; ix++) {
                kx = ix*_dk;
                val = *(zptr++);
                *(lptr++)= func(kx,ky,val);
            }
        }
        // wrap to the negative ky's
        for (int iy=-_N/2; iy< 0; iy++) {
            ky = iy*_dk;
            for (int ix=0; ix<= _N/2 ; ix++) {
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
        for (int iy=0; iy< _N/2; iy++) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); //x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
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
        yphase = std::polar(1.,y*(-_N/2.));
        for (int iy=-_N/2; iy< 0; iy++) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); // x DC term
            for (int ix=1; ix< _N/2 ; ix++) {
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
        for (int iy=0; iy< _N/2; iy++) {
            phase = yphase;
            for (int ix=0; ix<= _N/2 ; ix++) {
                z = *zptr;
                *zptr = phase * z;
                phase *= dxphase;
                zptr++;
            }
            yphase *= dyphase;
        }

        // wrap to the negative ky's
        yphase = std::polar(1.,(_N/2.)*y0);
        for (int iy=-_N/2; iy< 0; iy++) {
            phase = yphase;
            for (int ix=0; ix<= _N/2 ; ix++) {
                z = *zptr;
                *zptr = phase* z;
                phase *= dxphase;
                zptr++;
            }
            yphase *= dyphase;
        }
    }

    XTable::XTable(int N, double dx, double value) : _dx(dx)
    {
        if (N<=0) throw FFTError("XTable size <=0");
        _N = 2*((N+1)/2); //Round size up to even.
        _array.resize(_N);
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
        for (int i=0; i<_N*_N; i++)
            _array[i] += scalar * rhs._array[i];
    }

    void XTable::operator*=(double scale) 
    {
        check_array();
        clearCache(); // invalidate any stored interpolations
        const int Nsq = _N*_N;
        for (int i=0; i<Nsq; i++)
            _array[i] *= scale;
    }

    // Interpolate table (linearly) to some specific k:
    // x any y in physical units (to be divided by dx for indices)
    double XTable::interpolate(double x, double y, const Interpolant2d& interp) const 
    {
        xdbg << "interpolating " << x << " " << y << " " << std::endl;
        x /= _dx;
        y /= _dx;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes() 
             && std::abs(x - std::floor(x+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = ixMax = int(std::floor(x+0.01));
        } else {
            ixMin = int(std::ceil(x-interp.xrange()));
            ixMax = int(std::floor(x+interp.xrange()));
        }
        ixMin = std::max(ixMin, -_N/2);
        ixMax = std::min(ixMax, _N/2-1);
        if (ixMin > ixMax) return 0.;

        if ( interp.isExactAtNodes() 
             && std::abs(y - std::floor(y+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = iyMax = int(std::floor(y+0.01));
        } else {
            iyMin = int(std::ceil(y-interp.xrange()));
            iyMax = int(std::floor(y+interp.xrange()));
        }
        iyMin = std::max(iyMin, -_N/2);
        iyMax = std::min(iyMax, _N/2-1);
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
                for (int i=0; i<nx; i++) 
                    _xwt[i] = ixy->xval1d(i+ixMin-x);
            } else {
                assert(int(_xwt.size()) == nx);
            }

            // cache always holds sequential y values (no wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<double>::iterator nextSaved = _cache.begin();
            while (nextSaved != _cache.end() && _cacheStartY != iyMin) {
                _cache.pop_front();
                _cacheStartY++;
                nextSaved = _cache.begin();
            }

            for (int iy=iyMin; iy<=iyMax; iy++) {
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
            for (int iy=iyMin; iy<=iyMax; iy++) {
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
        for (int iy=0; iy<_N; iy++) {
            y = (iy-_N/2)*_dx;
            for (int ix=0; ix< _N ; ix++) {
                x = (ix-_N/2)*_dx;
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

        for (int iy=0; iy< _N; iy++) {
            y = (iy-_N/2)*_dx;
            for (int ix=0; ix< _N ; ix++) {
                x = (ix-_N/2)*_dx;
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
        for (int iy=-_N/2; iy< _N/2; iy++) 
            for (int ix=-_N/2; ix< _N/2; ix++) {
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
        kx*=_dx; ky*=_dx;
#ifdef FFT_DEBUG
        if (std::abs(kx) > M_PI || std::abs(ky) > M_PI) 
            throw FFTOutofRange("XTable::kval() args out of range");
#endif
        std::complex<double> dxphase=std::polar(1.,-kx);
        std::complex<double> dyphase=std::polar(1.,-ky);
        std::complex<double> sum=0.;

        const double* zptr=_array.get();
        std::complex<double> yphase=std::polar(1.,ky*_N/2.);
        std::complex<double> xphase=std::polar(1.,kx*_N/2.);
        std::complex<double> phase;
        for (int iy=0; iy< _N; iy++) {
            phase = yphase * xphase;
            for (int ix=0; ix< _N ; ix++) {
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

        XTable xt( _N, 2.*M_PI/(_N*_dk) );

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
        check_array();

        // check proper dimensions for xt
        assert(_N==xt.getN());

        // We'll need a new k array because FFTW kills the k array in this
        // operation.  Also, to put x=0 in center of array, we need to flop
        // every other sign of k array, and need to scale.
        dbg<<"Before make t_array"<<std::endl;
        FFTW_Array<std::complex<double> > t_array(_N);
        dbg<<"After make t_array"<<std::endl;
        double fac = _dk * _dk / (4*M_PI*M_PI);
        long int ind=0;
        dbg<<"t_array.size = "<<t_array.size()<<std::endl;
        for (int iy=0; iy<_N; iy++) {
            dbg<<"ind = "<<ind<<std::endl;
            for (int ix=0; ix<=_N/2; ix++) {
                if ( (ix+iy)%2==0) t_array[ind]=fac * _array[ind];
                else t_array[ind] = -fac* _array[ind];
                ind++;
            }
        }
        dbg<<"After fill t_array"<<std::endl;

        fftw_plan plan = fftw_plan_dft_c2r_2d(
            _N, _N, t_array.get_fftw(), xt._array.get_fftw(), FFTW_ESTIMATE);
        dbg<<"After make plan"<<std::endl;
        if (plan==NULL) throw FFTInvalid();

        // Run the transform:
        fftw_execute(plan);
        dbg<<"After exec plan"<<std::endl;
        fftw_destroy_plan(plan);
        dbg<<"After destroy plan"<<std::endl;

        xt._dx = 2.*M_PI/(_N*_dk);
        dbg<<"Done transform"<<std::endl;
    }

    // Same thing, but return a new XTable
    boost::shared_ptr<XTable> KTable::transform() const 
    {
        boost::shared_ptr<XTable> xt(new XTable( _N, 2.*M_PI/(_N*_dk) ));
        transform(*xt);
        return xt;
    }

    void XTable::fftwMeasure() const 
    {
        // Make a new copy of data array since measurement will overwrite:
        // Copy data into new array to avoid NaN's, etc., but not bothering
        // with scaling, etc.
        FFTW_Array<double> t_array = _array;

        KTable kt( _N, 2.*M_PI/(_N*_dx) );

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
        for (int iy=0; iy<_N; iy++) {
            for (int ix=0; ix<=_N/2; ix++) {
                if ( (ix+iy)%2==0) kt._array[ind] *= fac;
                else kt._array[ind] *= -fac;
                ind++;
            }
        }
        kt._dk = 2.*M_PI/(_N*_dx);
    }

    // Same thing, but return a new KTable
    boost::shared_ptr<KTable> XTable::transform() const 
    {
        boost::shared_ptr<KTable> kt(new KTable( _N, 2.*M_PI/(_N*_dx) ));
        transform(*kt);
        return kt;
    }


}
