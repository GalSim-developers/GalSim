// Routines for FFTW interface objects
// This time to use Version 3 of FFTW.

#include <limits>
#include <vector>
#include <cassert>
#include "FFT.h"
#include "Std.h"

namespace sbp {

    // A helper function that will return the smallest 2^n or 3x2^n value that is
    // even and >= the input integer.
    int goodFFTSize(int input) 
    {
        if (input<=2) return 2;
        // Reduce slightly to eliminate potential rounding errors:
        double insize = (1.-1e-5)*input;
        double log2n = log(2.)*ceil(log(insize)/log(2.));
        double log2n3 = log(3.) 
            + log(2.)*ceil((log(insize)-log(3.))/log(2.));
        log2n3 = std::max(log2n3, log(6.)); // must be even number
        int Nk = static_cast<int> (ceil(exp(std::min(log2n, log2n3))-1e-5));
        return Nk;
    }


    KTable::KTable(int _N, double _dk, std::complex<double> value) 
    {
        if (_N<=0) throw FFTError("KTable size <=0");
        N = 2*((_N+1)/2); //Round size up to even.
        dk = _dk;
        get_array(value);
        scaleby=1.;
        return;
    }


    size_t KTable::index(int ix, int iy) const 
    {
        if (ix<-N/2 || ix>N/2 || iy<-N/2 || iy>N/2) 
            FormatAndThrow<FFTOutofRange>() << "KTable index (" << ix << "," << iy
                << ") out of range for N=" << N;
        if (ix<0) {
            ix=-ix; iy=-iy; //need the conjugate in this case
        }
        if (iy<0) iy+=N;
        return iy*(N/2+1)+ix;
    }

    std::complex<double> KTable::kval(int ix, int iy) const 
    { 
        check_array();
        std::complex<double> retval=scaleby*array[index(ix,iy)];
        if (ix<0) return conj(retval);
        else return retval;
    }

    void KTable::kSet(int ix, int iy, std::complex<double> value) 
    {
        check_array();
        cache.clear(); // invalidate any stored interpolations
        if (ix<0) {
            array[index(ix,iy)]=conj(value)/scaleby;
            if (ix==-N/2) array[index(ix,-iy)]=value/scaleby;
        } else {
            array[index(ix,iy)]=value/scaleby;
            if (ix==0 || ix==N/2) array[index(ix,-iy)]=conj(value)/scaleby;
        }
        return;
    }

    void KTable::get_array(const std::complex<double> value) 
    {
        array = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>)*N*(N/2+1));
        for (int i=0; i<N*(N/2+1); i++)
            array[i]=value;
        return;
    }

    void KTable::copy_array(const KTable& rhs) 
    {
        cache.clear(); // invalidate any stored interpolations
        if (rhs.array==0) 
            throw FFTError("KTable::copy_array from null array");
        if (array!=0 && N!=rhs.N) kill_array();
        N = rhs.N; // makes sure our array will be of same size
        if (array==0) array = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>)*N*(N/2+1)); // allocate space
        for (int i=0; i<N*(N/2+1); i++) // copy element by element
            array[i]=rhs.array[i];
        return;
    }

    void KTable::clear() 
    {
        cache.clear(); // invalidate any stored interpolations
        if (!array) return;
        for (int i=0; i<N*(N/2+1); i++)
            array[i]=std::complex<double>(0.,0.);
        scaleby = 1.;
        return;
    }

    void KTable::kill_array() 
    {
        cache.clear(); // invalidate any stored interpolations
        if (!array) return;
        fftw_free(array);
        array=0;
        return;
    }

    void KTable::accumulate(const KTable& rhs, double scalar) 
    {
        cache.clear(); // invalidate any stored interpolations
        check_array();
        scalar *= rhs.scaleby / scaleby;
        if (N != rhs.N) throw FFTError("KTable::accumulate() with mismatched sizes");
        if (dk != rhs.dk) throw FFTError("KTable::accumulate() with mismatched dk");
        for (int i=0; i<N*(N/2+1); i++)
            array[i]+=scalar * rhs.array[i];
        return;
    }

    void KTable::operator*=(const KTable& rhs) 
    {
        cache.clear(); // invalidate any stored interpolations
        check_array();
        if (N != rhs.N) throw FFTError("KTable::operator*=() with mismatched sizes");
        if (dk != rhs.dk) throw FFTError("KTable::operator*=() with mismatched dk");
        scaleby *= rhs.scaleby;
        for (int i=0; i<N*(N/2+1); i++)
            array[i]*=rhs.array[i];
        return;
    }

    KTable* KTable::wrap(int Nout) const 
    {
        if (Nout < 0) FormatAndThrow<FFTError>() << "KTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        KTable* out = new KTable(Nout, dk, std::complex<double>(0.,0.));
        out->scaleby = scaleby;
        for (int iyin=-N/2; iyin<N/2; iyin++) {
            int iyout = iyin;
            while (iyout < -Nout/2) iyout+=Nout;
            while (iyout >= Nout/2) iyout-=Nout;
            int ixin = 0;
            while (ixin < N/2) {
                // number of points to accumulate without conjugation:
                // Do points that do *not* need to be conjugated:
                int nx = std::min(N/2-ixin+1, Nout/2+1);
                const std::complex<double>* inptr = array + index(ixin,iyin);
                std::complex<double>* outptr = out->array + out->index(0,iyout);
                for (int i=0; i<nx; i++) {
                    *outptr += *inptr;
                    inptr++;
                    outptr++;
                }
                ixin += Nout/2;
                if (ixin >= N/2) break;
                // Now do any points that *do* need conjugation
                // such that output storage locations go backwards
                inptr = array + index(ixin,iyin);
                outptr = out->array + out->index(Nout/2, -iyout);
                nx = std::min(N/2-ixin+1, Nout/2+1);
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

    XTable* XTable::wrap(int Nout) const 
    {
        if (Nout < 0) FormatAndThrow<FFTError>() << "XTable::wrap invalid Nout= " << Nout;
        // Make it even:
        Nout = 2*((Nout+1)/2);
        XTable* out = new XTable(Nout, dx, 0.);
        out->scaleby = scaleby;
        // What is (-N/2) wrapped to (+- Nout/2)?
        int excess = (N % Nout) / 2;  // Note N and Nout are positive.
        const int startOut = (excess==0) ? -Nout/2 : Nout/2 - excess;
        int iyout = startOut;
        for (int iyin=-N/2; iyin<N/2; iyin++, iyout++) {
            if (iyout >= Nout/2) iyout -= Nout;  // wrap y if needed
            int ixin = -N/2;
            int ixout = startOut;
            const double* inptr = array + index(ixin,iyin);
            while (ixin < N/2) {
                // number of points to write before wrapping:
                int nx = std::min(N/2-ixin, Nout/2-ixout);
                double* outptr = out->array + out->index(ixout,iyout);
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

    // Interpolate table to some specific k.  We WILL wrap the KTable to cover
    // entire interpolation kernel:
    std::complex<double> KTable::interpolate(
        double kx, double ky, const Interpolant2d& interp) const 
    {
        kx /= dk;
        ky /= dk;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes() 
             && abs(kx - floor(kx+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = static_cast<int> (floor(kx+0.01)) % N;
            if (ixMin < -N/2) ixMin += N;
            if (ixMin >= N/2) ixMin -= N;
            ixMax = ixMin;
        } else if (interp.xrange() >= N/2) {
            // use all the elements in row:
            ixMin = -N/2;
            ixMax = N/2-1;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            ixMin = static_cast<int> (ceil(kx-interp.xrange())) % N;
            if (ixMin < -N/2) ixMin += N;
            if (ixMin >= N/2) ixMin -= N;
            ixMax = static_cast<int> (floor(kx+interp.xrange())) % N;
            if (ixMax < -N/2) ixMax += N;
            if (ixMax >= N/2) ixMax -= N;
        }

        if ( interp.isExactAtNodes() 
             && abs(ky - floor(ky+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = static_cast<int> (floor(ky+0.01)) % N;
            if (iyMin < -N/2) iyMin += N;
            if (iyMin >= N/2) iyMin -= N;
            iyMax = iyMin;
        } else if (interp.xrange() >= N/2) {
            // use all the elements in row:
            iyMin = -N/2;
            iyMax = N/2-1;
        } else {
            // Put both bounds of kernel footprint in range [-N/2,N/2-1]
            iyMin = static_cast<int> (ceil(ky-interp.xrange())) % N;
            if (iyMin < -N/2) iyMin += N;
            if (iyMin >= N/2) iyMin -= N;
            iyMax = static_cast<int> (floor(ky+interp.xrange())) % N;
            if (iyMax < -N/2) iyMax += N;
            if (iyMax >= N/2) iyMax -= N;
        }

        std::complex<double> sum = 0.;
        const InterpolantXY* ixy = dynamic_cast<const InterpolantXY*> (&interp);
        if (ixy) {
            // Interpolant is seperable
            // We have the opportunity to speed up the calculation by
            // re-using the sums over rows.  So we will keep a 
            // cache of them.
            if (ixy != cacheInterp || kx!=cacheX) {
                cache.clear();
                cacheX = kx;
                cacheInterp = ixy;
            }

            // Going to have a special case for interpolation on 
            // a single iy value:
            if (iyMax==iyMin) {
                if (!cache.empty()) {
                    // See if we already have this row in cache:
                    int index = iyMin - cacheStartY;
                    if (index < 0) index += N;
                    if (index < int(cache.size()))
                        // We have it!
                        return cache[index];
                }
                // Desired row not in cache - kill cache, continue as normal.
                cache.clear();
            }

            // Build the x component of interpolant
            int nx = ixMax - ixMin + 1;
            if (nx<=0) nx+=N;
            std::vector<double> xwt(nx);
            for (int i=0; i<nx; i++) 
                xwt[i] = ixy->xvalWrapped1d(i+ixMin-kx, N);

            // cache always holds sequential y values (with wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<std::complex<double> >::iterator nextSaved = cache.begin();
            while (nextSaved != cache.end() && cacheStartY != iyMin) {
                cache.pop_front();
                cacheStartY++;
                if (cacheStartY >= N/2) cacheStartY-= N;
                nextSaved = cache.begin();
            }

            // use kval to keep track of conjugations etc here.
            // ??? There is an opportunity to speed up if I want
            // ??? to do this with array incrementing.
            int ny = iyMax - iyMin + 1;
            if (ny<=0) ny+=N;
            int iy = iyMin;
            for (int j = 0; j<ny; j++, iy++) {
                if (iy >= N/2) iy-=N;   // wrap iy if needed
                std::complex<double> sumy = 0.;
                if (nextSaved != cache.end()) {
                    // This row is cached
                    sumy = *nextSaved;
                    ++nextSaved;
                } else {
                    // Need to compute a new row's sum
                    int ix = ixMin;
                    for (int i=0; i<nx; i++, ix++) {
                        if (ix > N/2) ix-=N; //check for wrap
                        sumy += xwt[i]*kval(ix,iy);
                    }
                    // Add to back of cache
                    if (cache.empty()) cacheStartY = iy;
                    cache.push_back(sumy);
                    nextSaved = cache.end();
                }
                sum += sumy * ixy->xvalWrapped1d(iy-ky, N);
            }
        } else {
            // Interpolant is not seperable, calculate weight at each point
            int ny = iyMax - iyMin + 1;
            if (ny<=0) ny+=N;
            int nx = ixMax - ixMin + 1;
            if (nx<=0) nx+=N;
            int iy = iyMin;
            for (int j = 0; j<ny; j++, iy++) {
                if (iy >= N/2) iy-=N;   // wrap iy if needed
                int ix = ixMin;
                for (int i=0; i<nx; i++, ix++) {
                    if (ix > N/2) ix-=N; //check for wrap
                    // use kval to keep track of conjugations
                    sum += interp.xvalWrapped(ix-kx, iy-ky, N)*kval(ix,iy);
                }
            }
        }
        return sum;
    }

    // Fill table from a function:
    void KTable::fill(KTable::function1 func)
    {
        cache.clear(); // invalidate any stored interpolations
        check_array();
        std::complex<double>* zptr=array;
        double kx, ky;
        std::complex<double>* tmp1 = new std::complex<double>[N/2];
        std::complex<double>* tmp2 = new std::complex<double>[N/2];

        // [ky/dk] = iy = 0
        for (int ix=0; ix< N/2+1 ; ix++) {
            kx = ix*dk;
            *(zptr++) = func(kx,0);                  // [kx/dk] = ix = 0 to N/2
        }
        // [ky/dk] = iy = 1 to (N/2-1)
        for (int iy=1; iy< N/2; iy++) {
            ky = iy*dk;
            *(zptr++) = tmp1[iy] = func(0,ky);        // [kx/dk] = ix = 0
            for (int ix=1; ix< N/2 ; ix++) {    
                kx = ix*dk;
                *(zptr++) = func(kx,ky);               // [kx/dk] = ix = 1 to (N/2-1)
            }
            *(zptr++) = tmp2[iy] = func((N/2)*dk,ky); // [kx/dk] = ix =N/2
        }
        // Wrap to the negative ky's
        // [ky/dk] = iy = -N/2
        for (int ix=0; ix< N/2+1 ; ix++) {
            kx = ix*dk;
            *(zptr++) = func(kx,-N/2*dk);         // [kx/dk] = ix = 0 to N/2   
        }
        // [ky/dk] = iy = (-N/2+1) to (-1)
        for (int iy=-N/2+1; iy< 0; iy++) {
            ky = iy*dk;
            *(zptr++) = conj(tmp1[-iy]);       // [kx/dk] = ix = 0
            for (int ix=1; ix< N/2 ; ix++) {
                kx = ix*dk;
                *(zptr++) = func(kx,ky);         // [kx/dk] = ix = 1 to (N/2-1)
            }
            *(zptr++) = conj(tmp2[-iy]);      // [kx/dk] = ix = N/2
        }
        return;
    } 

    // Integrate a function over k - can be function of k or of PSF(k)
    std::complex<double> KTable::integrate(KTable::function2 func) const
    {
        check_array();
        std::complex<double> sum=0.;
        std::complex<double> val;
        double kx, ky;
        std::complex<double>* zptr=array;
        // Do the positive y frequencies
        for (int iy=0; iy<= N/2; iy++) {
            ky = iy*dk;
            val = *(zptr++) * scaleby;
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix< N/2 ; ix++) {
                kx = ix*dk;
                val = *(zptr++) * scaleby;
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = dk*N/2;
            val = *(zptr++) * scaleby;
            sum += func(kx,ky,val); // x Nyquist freq
        }

        // wrap to the negative ky's
        for (int iy=-N/2+1; iy< 0; iy++) {
            ky = iy*dk;
            val = *(zptr++) * scaleby;
            kx = 0.;
            sum += func(kx,ky,val); //x DC term
            for (int ix=1; ix< N/2 ; ix++) {
                kx = ix*dk;
                val = *(zptr++) * scaleby;
                sum += func(kx,ky,val);
                sum += func(-kx,-ky,conj(val));
            }
            kx = dk*N/2;
            val = *(zptr++) * scaleby;
            sum += func(kx,ky,val); // x Nyquist
        }
        sum *= dk*dk;
        return sum;
    }

    // Integrate KTable over d^2k (sum of all pixels * dk * dk)
    std::complex<double> KTable::integratePixels() const 
    {
        check_array();
        std::complex<double> sum=0.;
        std::complex<double>* zptr=array;
        // Do the positive y frequencies
        for (int iy=0; iy<= N/2; iy++) {
            sum += *(zptr++) * scaleby;    // x DC term
            for (int ix=1; ix< N/2 ; ix++) {
                sum += *(zptr) * scaleby;
                sum += conj(*(zptr++) * scaleby);
            }
            sum += *(zptr++) * scaleby;
        }
        // wrap to the negative ky's
        for (int iy=-N/2+1; iy< 0; iy++) {
            sum += *(zptr++) * scaleby;    // x DC term
            for (int ix=1; ix< N/2 ; ix++) {
                sum += *(zptr) * scaleby;
                sum += conj(*(zptr++) * scaleby);
            }
            sum += *(zptr++) * scaleby;
        }
        sum *= dk*dk;
        return sum;
    }

    // Make a new table that is function of old.
    KTable* KTable::function(KTable::function2 func) const 
    {
        check_array();
        KTable* lhs = new KTable(N,dk);
        std::complex<double> val;
        double kx, ky;
        std::complex<double>* zptr=array;
        std::complex<double>* lptr=lhs->array;
        // Do the positive y frequencies
        for (int iy=0; iy< N/2; iy++) {
            ky = iy*dk;
            for (int ix=0; ix<= N/2 ; ix++) {
                kx = ix*dk;
                val = *(zptr++) * scaleby;
                *(lptr++)= func(kx,ky,val);
            }
        }
        // wrap to the negative ky's
        for (int iy=-N/2; iy< 0; iy++) {
            ky = iy*dk;
            for (int ix=0; ix<= N/2 ; ix++) {
                kx = ix*dk;
                val = *(zptr++) * scaleby;
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
        x*=dk; y*=dk;
        // Don't evaluate if x not in fundamental period +-PI/dk:
        if (abs(x) > M_PI || abs(y) > M_PI) throw FFTOutofRange(" (x,y) too big in xval()");
        std::complex<double> I(0.,1.);
        std::complex<double> dxphase=exp(I*x);
        std::complex<double> dyphase=exp(I*y);
        std::complex<double> phase(1.,0.);
        std::complex<double> z;
        double sum=0.;
        // y DC terms first:
        std::complex<double>* zptr=array;
        // Do the positive y frequencies
        std::complex<double> yphase=1.;
        for (int iy=0; iy< N/2; iy++) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); //x DC term
            for (int ix=1; ix< N/2 ; ix++) {
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
        yphase = exp(I*(y*(-N/2)));
        for (int iy=-N/2; iy< 0; iy++) {
            phase = yphase;
            z= *(zptr++);
            sum += (phase*z).real(); // x DC term
            for (int ix=1; ix< N/2 ; ix++) {
                phase *= dxphase;
                z= *(zptr++);
                sum += (phase*z).real() * 2.;
            }
            phase *= dxphase; //ix=N/2 has no mirror:
            z= *(zptr++);
            sum += (phase*z).real();
            yphase *= dyphase;
        }

        sum *= dk*dk*scaleby/(4.*M_PI*M_PI); //inverse xform has 2pi in it.
        return sum;
    }

    // Translate the PSF to be for source at (x0,y0);
    void KTable::translate(double x0, double y0) 
    {
        cache.clear(); // invalidate any stored interpolations
        check_array();
        // convert to phases:
        x0*=dk; y0*=dk;
        // too big will just be wrapping around:
        if (x0 > M_PI || y0 > M_PI) throw FFTOutofRange("(x0,y0) too big in translate()");
        std::complex<double> I(0.,1.);
        std::complex<double> dxphase=exp(std::complex<double>(0.,-x0));
        std::complex<double> dyphase=exp(std::complex<double>(0.,-y0));
        std::complex<double> phase(1.,0.);

        std::complex<double> yphase=1.;
        std::complex<double> z;

        std::complex<double>* zptr=array;

        for (int iy=0; iy< N/2; iy++) {
            phase = yphase;
            for (int ix=0; ix<= N/2 ; ix++) {
                z = *zptr;
                *zptr = phase * z;
                phase *= dxphase;
                zptr++;
            }
            yphase *= dyphase;
        }

        // wrap to the negative ky's
        yphase = exp(I*((N/2)*y0));
        for (int iy=-N/2; iy< 0; iy++) {
            phase = yphase;
            for (int ix=0; ix<= N/2 ; ix++) {
                z = *zptr;
                *zptr = phase* z;
                phase *= dxphase;
                zptr++;
            }
            yphase *= dyphase;
        }
        return;
    }

    XTable::XTable(int _N, double _dx, double value) 
    {
        if (_N<=0) throw FFTError("XTable size <=0");
        N = 2*((_N+1)/2); //Round size up to even.
        dx = _dx;
        get_array(value);
        scaleby=1.;
        return;
    }

    size_t XTable::index(int ix, int iy) const 
    {
        // origin will be in center.
        ix += N/2;
        iy += N/2;
        if (ix<0 || ix>=N || iy<0 || iy>=N) 
            FormatAndThrow<FFTOutofRange>() << "XTable index (" << ix << "," << iy
                << ") out of range for N=" << N;
        return iy*N+ix;
    }

    double XTable::xval(int ix, int iy) const 
    {
        check_array();
        return scaleby*array[index(ix,iy)];
    }

    void XTable::xSet(int ix, int iy, double value) 
    {
        check_array();
        cache.clear(); // invalidate any stored interpolations
        array[index(ix,iy)]=value/scaleby;
        return;
    }

    void XTable::get_array(const double value) 
    {
        array = (double*) fftw_malloc(sizeof(double)*N*N);
        for (int i=0; i<N*N; i++)
            array[i]=value;
        return;
    }

    void XTable::copy_array(const XTable& rhs) 
    {
        cache.clear(); // invalidate any stored interpolations
        if (rhs.array==0) 
            throw FFTError("XTable::copy_array from null array");
        if (array!=0 && N!=rhs.N) kill_array();
        if (array==0)   array = (double*) fftw_malloc(sizeof(double)*N*N);
        for (int i=0; i<N*N; i++)
            array[i]=rhs.array[i];
        return;
    }

    void XTable::kill_array() 
    {
        if (!array) return;
        cache.clear(); // invalidate any stored interpolations
        fftw_free(array);
        array=0;
        return;
    }

    void XTable::clear() 
    {
        if (!array) return;
        cache.clear(); // invalidate any stored interpolations
        for (int i=0; i<N*N; i++)
            array[i]=0.;
        scaleby = 1.;
        return;
    }

    void XTable::accumulate(const XTable& rhs, double scalar) 
    {
        check_array();
        cache.clear(); // invalidate any stored interpolations
        scalar *= rhs.scaleby / scaleby;
        if (N != rhs.N) throw FFTError("XTable::accumulate() with mismatched sizes");
        for (int i=0; i<N*N; i++)
            array[i] +=scalar * rhs.array[i];
        return;
    }

    // Interpolate table (linearly) to some specific k:
    // x any y in physical units (to be divided by dx for indices)
    double XTable::interpolate(double x, double y, const Interpolant2d& interp) const 
    {
#ifdef DANIELS_TRACING
        cout << "interpolating " << x << " " << y << " " << endl;
#endif
        x /= dx;
        y /= dx;
        int ixMin, ixMax, iyMin, iyMax;
        if ( interp.isExactAtNodes() 
             && abs(x - floor(x+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // x coord lies right on integer value, no interpolation in x direction
            ixMin = ixMax = static_cast<int> (floor(x+0.01));
        } else {
            ixMin = static_cast<int> (ceil(x-interp.xrange()));
            ixMax = static_cast<int> (floor(x+interp.xrange()));
        }
        ixMin = std::max(ixMin, -N/2);
        ixMax = std::min(ixMax, N/2-1);
        if (ixMin > ixMax) return 0.;

        if ( interp.isExactAtNodes() 
             && abs(y - floor(y+0.01)) < 10.*std::numeric_limits<double>::epsilon()) {
            // y coord lies right on integer value, no interpolation in y direction
            iyMin = iyMax = static_cast<int> (floor(y+0.01));
        } else {
            iyMin = static_cast<int> (ceil(y-interp.xrange()));
            iyMax = static_cast<int> (floor(y+interp.xrange()));
        }
        iyMin = std::max(iyMin, -N/2);
        iyMax = std::min(iyMax, N/2-1);
        if (iyMin > iyMax) return 0.;

        double sum = 0.;
        const InterpolantXY* ixy = dynamic_cast<const InterpolantXY*> (&interp);
        if (ixy) {
            // Interpolant is seperable
            // We have the opportunity to speed up the calculation by
            // re-using the sums over rows.  So we will keep a 
            // cache of them.
            if (ixy != cacheInterp || x!=cacheX) {
                cache.clear();
                cacheX = x;
                cacheInterp = ixy;
            }

            // Going to have a special case for interpolation on 
            // a single iy value:
            if (iyMax==iyMin) {
                if (!cache.empty()) {
                    // See if we already have this row in cache:
                    int index = iyMin - cacheStartY;
                    if (index < 0) index += N;
                    if (index < int(cache.size())) {
                        // We have it!
                        return cache[index]*scaleby;
                    }
                }
                // Desired row not in cache - kill cache, continue as normal.
                cache.clear();
            }

            // Build x factors for interpolant
            int nx = ixMax - ixMin + 1;
            std::vector<double> xwt(nx);
            for (int i=0; i<nx; i++) 
                xwt[i] = ixy->xval1d(i+ixMin-x);

            // cache always holds sequential y values (no wrap).  Throw away
            // elements until we get to the one we need first
            std::deque<double>::iterator nextSaved = cache.begin();
            while (nextSaved != cache.end() && cacheStartY != iyMin) {
                cache.pop_front();
                cacheStartY++;
                nextSaved = cache.begin();
            }

            for (int iy = iyMin; iy<=iyMax; iy++) {
                double sumy = 0.;
                if (nextSaved != cache.end()) {
                    // This row is cached
                    sumy = *nextSaved;
                    ++nextSaved;
                } else {
                    // Need to compute a new row's sum
                    double* dptr = array + index(ixMin, iy);
                    for (int i=0; i<nx; i++, dptr++)
                        sumy += xwt[i]* (*dptr);
                    // Add to back of cache
                    if (cache.empty()) cacheStartY = iy;
                    cache.push_back(sumy);
                    nextSaved = cache.end();
                }
                sum += sumy * ixy->xval1d(iy-y);
            }
        } else {
            // Interpolant is not seperable, calculate weight at each point
            for (int iy = iyMin; iy<=iyMax; iy++) {
                double* dptr = array + index(ixMin, iy);
                for (int ix = ixMin; ix<=ixMax; ++ix, ++dptr)
                    sum += *dptr * interp.xval(ix-x, iy-y);
            }
        }
        return sum * scaleby;
    }

    // Fill table from a function:
    void XTable::fill(XTable::function1 func)
    {
        check_array();
        cache.clear(); // invalidate any stored interpolations
        double* zptr=array;
        double x, y;
        for (int iy=0; iy<N; iy++) {
            y = (iy-N/2)*dx;
            for (int ix=0; ix< N ; ix++) {
                x = (ix-N/2)*dx;
                *(zptr++) = func(x,y);
            }
        }
        return;
    }

    // Integrate a function over x - can be function of x or of PSF(x)
    // Setting the Boolean flag gives sum over samples, not integral.
    double XTable::integrate(XTable::function2 func, bool  sumonly) const 
    {
        check_array();
        double sum=0.;
        double val;
        double x, y;
        double* zptr=array;

        for (int iy=0; iy< N; iy++) {
            y = (iy-N/2)*dx;
            for (int ix=0; ix< N ; ix++) {
                x = (ix-N/2)*dx;
                val = *(zptr++) * scaleby;
                sum += func(x,y,val);
            }
        }

        if (!sumonly) sum *= dx*dx;
        return sum;
    }

    double XTable::integratePixels() const 
    {
        check_array();
        double sum=0.;
        double* zptr=array;
        for (int iy=-N/2; iy< N/2; iy++) 
            for (int ix=-N/2; ix< N/2; ix++) {
                sum += *(zptr++) * scaleby;
            }
        sum *= dx*dx;
        return (double) sum;
    }

    // Transform to a single k point:
    std::complex<double> XTable::kval(double kx, double ky) const 
    {
        check_array();
        // Don't evaluate if k not in fundamental period 
        kx*=dx; ky*=dx;
        if (abs(kx) > M_PI || abs(ky) > M_PI) 
            throw FFTOutofRange("XTable::kval() args out of range");
        std::complex<double> I(0.,1.);
        std::complex<double> dxphase=exp(-I*kx);
        std::complex<double> dyphase=exp(-I*ky);
        std::complex<double> phase(1.,0.);
        std::complex<double> z;
        std::complex<double> sum=0.;

        double* zptr=array;
        std::complex<double> yphase=exp(I*(ky*N/2));
        for (int iy=0; iy< N; iy++) {
            phase = yphase;
            phase *= exp(I*(kx*N/2));
            for (int ix=0; ix< N ; ix++) {
                sum += phase* (*(zptr++));
                phase *= dxphase;
            }
            yphase *= dyphase;
        }
        sum *= dx*dx*scaleby;
        return sum;
    }

    // Have FFTW develop "wisdom" on doing this kind of transform
    void KTable::fftwMeasure() const 
    {
        std::complex<double>* t_array = 
            (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>)*N*(N/2+1));
        // Copy data into new array to avoid NaN's, etc., but not bothering
        // with scaling, etc.
        for (int i=0; i<N*(N/2+1); i++)
            t_array[i] = array[i];

        XTable* xt = new XTable( N, 2*M_PI/(N*dk) );

        fftw_plan plan = fftw_plan_dft_c2r_2d(
            N, N, reinterpret_cast<fftw_complex*> (t_array), xt->array, FFTW_MEASURE);
        if (plan==NULL) throw FFTInvalid();
        delete xt;
        fftw_free(t_array);
        fftw_destroy_plan(plan);
    }

    // Fourier transform from (complex) k to x:
    XTable* KTable::transform() const 
    {
        check_array();
        // We'll need a new k array because FFTW kills the k array in this
        // operation.  Also, to put x=0 in center of array, we need to flop
        // every other sign of k array, and need to scale.

        std::complex<double>* t_array = 
            (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>)*N*(N/2+1));
        double fac = scaleby * dk * dk / (4*M_PI*M_PI);
        long int ind=0;
        for (int iy=0; iy<N; iy++)
            for (int ix=0; ix<=N/2; ix++) {
                if ( (ix+iy)%2==0) t_array[ind]=fac * array[ind];
                else t_array[ind] = -fac* array[ind];
                ind++;
            }

        XTable* xt = new XTable( N, 2*M_PI/(N*dk) );

        fftw_plan plan = 
            fftw_plan_dft_c2r_2d(
                N, N, reinterpret_cast<fftw_complex*> (t_array), xt->array, FFTW_ESTIMATE);
        if (plan==NULL) throw FFTInvalid();

        // Run the transform:
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        fftw_free(t_array);
        return xt;
    }

    // same function, takes XTable reference as agrument 
    void KTable::transform(XTable& xt) const 
    {
        check_array();

        // check proper dimensions for xt
        assert(N==xt.getN());

        // We'll need a new k array because FFTW kills the k array in this
        // operation.  Also, to put x=0 in center of array, we need to flop
        // every other sign of k array, and need to scale.

        std::complex<double>* t_array = 
            (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>)*N*(N/2+1));
        double fac = scaleby * dk * dk / (4*M_PI*M_PI);
        long int ind=0;
        for (int iy=0; iy<N; iy++)
            for (int ix=0; ix<=N/2; ix++) {
                if ( (ix+iy)%2==0) t_array[ind]=fac * array[ind];
                else t_array[ind] = -fac* array[ind];
                ind++;
            }

        fftw_plan plan = fftw_plan_dft_c2r_2d(
            N, N, reinterpret_cast<fftw_complex*> (t_array), xt.array, FFTW_ESTIMATE);
        if (plan==NULL) throw FFTInvalid();

        // Run the transform:
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        fftw_free(t_array);

        xt.dx = 2*M_PI/(N*dk);
        xt.scaleby = 1.;
    }

    void XTable::fftwMeasure() const 
    {
        // Make a new copy of data array since measurement will overwrite:
        double* t_array = 
            (double*) fftw_malloc(sizeof(double)*N*N);
        // Copy data into new array to avoid NaN's, etc., but not bothering
        // with scaling, etc.
        for (int i=0; i<N*N; i++)
            t_array[i] = array[i];

        KTable* kt = new KTable( N, 2*M_PI/(N*dx) );

        fftw_plan plan = fftw_plan_dft_r2c_2d(
            N,N, t_array, reinterpret_cast<fftw_complex*> (kt->array), FFTW_MEASURE);
        if (plan==NULL) throw FFTInvalid();

        delete kt;
        fftw_free(t_array);
        fftw_destroy_plan(plan);
    }

    // Fourier transform from x back to (complex) k:
    KTable* XTable::transform() const 
    {
        check_array();

        KTable* kt = new KTable( N, 2*M_PI/(N*dx) );

        fftw_plan plan = fftw_plan_dft_r2c_2d(
            N,N, array, reinterpret_cast<fftw_complex*> (kt->array), FFTW_ESTIMATE);
        if (plan==NULL) throw FFTInvalid();
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Now scale the k spectrum and flip signs for x=0 in middle.
        double fac = scaleby * dx * dx; 
        size_t ind=0;
        for (int iy=0; iy<N; iy++) {
            for (int ix=0; ix<=N/2; ix++) {
                if ( (ix+iy)%2==0) kt->array[ind] *= fac;
                else kt->array[ind] *= -fac;
                ind++;
            }
        }

        return kt;
    }

}
