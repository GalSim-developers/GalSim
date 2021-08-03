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

#include <vector>
#include <map>
#include <algorithm>
#include <cstring>  // For memset
#ifdef __SSE2__
#include "xmmintrin.h"
#endif

#include "SBInterpolatedImage.h"
#include "SBInterpolatedImageImpl.h"
#include "Std.h"

namespace galsim {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedImage methods

    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams) :
        SBProfile(new SBInterpolatedImageImpl(
                image, init_bounds, nonzero_bounds, xInterp, kInterp, stepk, maxk, gsparams)) {}

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs) : SBProfile(rhs) {}

    SBInterpolatedImage::~SBInterpolatedImage() {}

    const Interpolant& SBInterpolatedImage::getXInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getXInterp();
    }

    const Interpolant& SBInterpolatedImage::getKInterp() const
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

    ConstImageView<double> SBInterpolatedImage::getPaddedImage() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getPaddedImage();
    }

    ConstImageView<double> SBInterpolatedImage::getNonZeroImage() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getNonZeroImage();
    }

    ConstImageView<double> SBInterpolatedImage::getImage() const
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).getImage();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedImageImpl methods

#define INVALID -1.e300  // dummy value to indicate that flux or centroid not calculated yet.

    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _image(image.view()), _image_bounds(image.getBounds()),
        _init_bounds(init_bounds), _nonzero_bounds(nonzero_bounds),
        _xInterp(xInterp), _kInterp(kInterp),
        _stepk(stepk), _maxk(maxk),
        _flux(INVALID), _xcentroid(INVALID), _ycentroid(INVALID),
        _readyToShoot(false)
    {
        dbg<<"image bounds = "<<image.getBounds()<<std::endl;
        dbg<<"init bounds = "<<_init_bounds<<std::endl;
        dbg<<"nonzero bounds = "<<_nonzero_bounds<<std::endl;

        if (_stepk <= 0.) {
            // Calculate stepK:
            //
            // The amount of flux missed in a circle of radius pi/stepk should be at
            // most folding_threshold of the flux.
            //
            // We add the size of the image and the size of the interpolant in quadrature.
            // (Note: Since this isn't a radial profile, R isn't really a radius, but rather
            //        the size of the square box that is enclosing all the flux.)
            double R = std::max((_init_bounds.getXMax()-_init_bounds.getXMin())/2.,
                                (_init_bounds.getYMax()-_init_bounds.getYMin())/2.);
            // Add xInterp range in quadrature just like convolution:
            double R2 = _xInterp.xrange();
            dbg<<"R(image) = "<<R<<", R(interpolant) = "<<R2<<std::endl;
            R = sqrt(R*R + R2*R2);
            dbg<<"=> R = "<<R<<std::endl;
            _stepk = M_PI / R;
            dbg<<"stepk = "<<_stepk<<std::endl;
        }

        _uscale = 1. / (2.*M_PI);
        _maxk1 = _xInterp.urange()/_uscale;
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
    }


    SBInterpolatedImage::SBInterpolatedImageImpl::~SBInterpolatedImageImpl() {}

    const Interpolant& SBInterpolatedImage::SBInterpolatedImageImpl::getXInterp() const
    { return _xInterp; }

    const Interpolant& SBInterpolatedImage::SBInterpolatedImageImpl::getKInterp() const
    { return _kInterp; }

    double SBInterpolatedImage::SBInterpolatedImageImpl::maxSB() const
    { return _image.maxAbsElement(); }

    double SBInterpolatedImage::SBInterpolatedImageImpl::xValue(const Position<double>& pos) const
    {
        double x = pos.x;
        double y = pos.y;

        int p1, p2, q1, q2;  // Range over which we need to sum.
        const double SMALL = 10.*std::numeric_limits<double>::epsilon();

        // If x or y are integers, only sum 1 element in that direction.
        if (std::abs(x-std::floor(x+0.01)) < SMALL*(std::abs(x)+1)) {
            p1 = p2 = int(std::floor(x+0.01));
        } else {
            p1 = int(std::ceil(x-_xInterp.xrange()));
            p2 = int(std::floor(x+_xInterp.xrange()));
        }
        if (std::abs(y-std::floor(y+0.01)) < SMALL*(std::abs(y)+1)) {
            q1 = q2 = int(std::floor(y+0.01));
        } else {
            q1 = int(std::ceil(y-_xInterp.xrange()));
            q2 = int(std::floor(y+_xInterp.xrange()));
        }

        // If either range is not in non-zero part, then value is 0.
        if (p2 < _nonzero_bounds.getXMin() ||
            p1 > _nonzero_bounds.getXMax() ||
            q2 < _nonzero_bounds.getYMin() ||
            q1 > _nonzero_bounds.getYMax()) {
            return 0.;
        }
        // Limit to nonzero region
        if (p1 < _nonzero_bounds.getXMin()) p1 = _nonzero_bounds.getXMin();
        if (p2 > _nonzero_bounds.getXMax()) p2 = _nonzero_bounds.getXMax();
        if (q1 < _nonzero_bounds.getYMin()) q1 = _nonzero_bounds.getYMin();
        if (q2 > _nonzero_bounds.getYMax()) q2 = _nonzero_bounds.getYMax();

        // We'll need these for each row.  Save them.
        double xwt[p2-p1+1];
        for (int p=p1, pp=0; p<=p2; ++p, ++pp) xwt[pp] = _xInterp.xval(p-x);

        double sum = 0.;
        for (int q=q1; q<=q2; ++q) {
            double xsum = 0.;
            for (int p=p1, pp=0; p<=p2; ++p, ++pp) {
                xsum += xwt[pp] * _image(p,q);
            }
            sum += xsum * _xInterp.xval(q-y);
        }
        return sum;
    }

    int WrapKIndex(int k, int No2, int N)
    {
        k = (k + No2) % N;
        if (k < 0) k += N;
        return k - No2;
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

    // This is the inner loop in all of the KValue calculations, including both the regular
    // kValue method and both versions of fillKImage.
    std::complex<double> KValueInnerLoop(int n, int p, int q, int No2, int N, double* xwt,
                                         const BaseImage<std::complex<double> >& kimage)
    {
        std::complex<double> sum = 0.;
#if 0
        // This is the more straightforward implementation of this calculation, which we
        // preserve here for readability.
        for (; n; --n, ++p) {
            if (p == No2+1) p -= N;
            // _kimage doesn't store p<0 half, so need to use the fact that
            // _kimage(p,q) = conj(_kimage(-p,-q)) when p < 0.
            if (p < 0)
                if (q == -No2)
                    // N.B. _kimage(p,No2) == _kimage(p,-No2)
                    sum += *xwt++ * std::conj(kimage(-p,q));
                else
                    sum += *xwt++ * std::conj(kimage(-p,-q));
            else
                sum += *xwt++ * kimage(p,q);
        }
#else
        // This version is more efficient, but a little obfuscated.
        // It is equivalent in result to the above calculation.

        // Note: q=No2 is not stored in the kimage, so when doing negative p values
        // we need to use q on the conjugate side, not -q.
        // Figure this out now, so we can ignore this subtlety below.
        int mq = q == -No2 ? q : -q;
        assert(kimage.getStep() == 1);

        // First do any negative p values
        if (p < 0) {
            xdbg<<"Some initial negative p: p = "<<p<<std::endl;
            int n1 = std::min(n, -p);
            xdbg<<"n1 = "<<n1<<std::endl;
            n -= n1;
            const std::complex<double>* ptr = &kimage(-p,mq);
            sum += ZDot<true>(n1, xwt, ptr);
            xwt += n1;
            p = 0;
        }

        // Next do positive p values:
        if (n) {
            xdbg<<"Positive p: p = "<<p<<std::endl;
            const std::complex<double>* ptr = &kimage(p,q);
            int n1 = std::min(n, No2+1-p);
            xdbg<<"n1 = "<<n1<<std::endl;
            n -= n1;
            sum += ZDot<false>(n1, xwt, ptr);
            xwt += n1;
        }

        // Finally if we've wrapped around again, do more negative p values:
        if (n) {
            xdbg<<"More negative p: p = "<<p<<std::endl;
            int n1 = std::min(n, No2);
            // Note: n1 is always n in practice, but this prevents pointer access going past
            // edges of image if there is some unanticipated use case where it isn't.
            xassert(n < No2);
            xdbg<<"n1 = "<<n1<<std::endl;
            p = -No2+1;
            const std::complex<double>* ptr = &kimage(-p,mq);
            sum += ZDot<true>(n1, xwt, ptr);
        }
#endif
        return sum;
    }

    std::complex<double> SBInterpolatedImage::SBInterpolatedImageImpl::kValue(
        const Position<double>& kpos) const
    {
        double kx = kpos.x;
        double ky = kpos.y;

        dbg<<"evaluating kValue("<<kx<<","<<ky<<")"<<std::endl;

        // Don't bother if the desired k value is cut off by the x interpolant:
        if (std::abs(kx) > _maxk1 || std::abs(ky) > _maxk1) return std::complex<double>(0.,0.);

        checkK();
        double xKernelTransform = _xInterp.uval(kx*_uscale) * _xInterp.uval(ky*_uscale);
        dbg<<"xKernelTransform = "<<xKernelTransform<<std::endl;

        int No2 = _kimage->getBounds().getXMax();
        int N = No2 * 2;
        double kscale = No2/M_PI; // This is 1/dk
        xdbg<<"kimage bounds = "<<_kimage->getBounds()<<", scale = "<<kscale<<std::endl;
        kx *= kscale;
        ky *= kscale;

        int p1, p2, q1, q2;  // Range over which we need to sum.
        // Note, unlike for xValue, whre we limited the range to the size of the image,
        // here, we allow i,j to nominally go off the kimage bounds, in which case we
        // wrap around when using it, due to the periodic nature of the fft.
        const double SMALL = 10.*std::numeric_limits<double>::epsilon();

        if (std::abs(kx-std::floor(kx+0.01)) < SMALL*(std::abs(kx)+1)) {
            // If kx or ky are integers, only sum 1 element in that direction.
            p1 = p2 = int(std::floor(kx+0.01));
        } else {
            p1 = int(std::ceil(kx-_kInterp.xrange()));
            p2 = int(std::floor(kx+_kInterp.xrange()));
        }
        if (std::abs(ky-std::floor(ky+0.01)) < SMALL*(std::abs(ky)+1)) {
            q1 = q2 = int(std::floor(ky+0.01));
        } else {
            q1 = int(std::ceil(ky-_kInterp.xrange()));
            q2 = int(std::floor(ky+_kInterp.xrange()));
        }
        dbg<<"p range = "<<p1<<"..."<<p2<<std::endl;
        dbg<<"q range = "<<q1<<"..."<<q2<<std::endl;

        // We'll need these for each row.  Save them.
        double xwt[p2-p1+1];
        for (int p=p1, pp=0; p<=p2; ++p, ++pp) xwt[pp] = _kInterp.xval(p-kx);

        std::complex<double> sum = 0.;
        int pwrap1 = WrapKIndex(p1, No2, N);
        int qwrap1 = WrapKIndex(q1, No2, N);
        dbg<<"pwrap, qwrap = "<<qwrap1<<','<<qwrap1<<std::endl;
        dbg<<"kimage bounds = "<<_kimage->getBounds()<<std::endl;
        for (int q=q1, qwrap=qwrap1; q<=q2; ++q, ++qwrap) {
            if (qwrap == No2) qwrap -= N;
            std::complex<double> xsum = KValueInnerLoop(p2-p1+1,pwrap1,qwrap,No2,N,xwt,*_kimage);
            sum += xsum * _kInterp.xval(q-ky);
        }

        dbg<<"sum = "<<sum<<std::endl;
        sum *= xKernelTransform;
        dbg<<"sum => "<<sum<<std::endl;
        return sum;
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkK() const
    {
        // Conduct FFT
        if (_kimage.get()) return;

        int Nx = _image.getXMax()-_image.getXMin()+1;
        dbg<<"Nx = "<<Nx<<std::endl;
        Bounds<int> b(0,Nx/2,-Nx/2,Nx/2-1);
        _kimage.reset(new ImageAlloc<std::complex<double> >(b));
        rfft(_image, _kimage->view());
        dbg<<"made kimage\n";
        dbg<<"kimage bounds = "<<_kimage->getBounds()<<std::endl;
        dbg<<"kimage flux = "<<(*_kimage)(0,0).real()<<std::endl;
    }

    template <typename T>
    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<T> im,
        double x0, double dx, int izero,
        double y0, double dy, int jzero) const
    {
        dbg<<"SBInterpolatedImage fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        const double SMALL = 10.*std::numeric_limits<double>::epsilon();

        // Notation: We have two images to loop over here, so there are two sets of x,y values.
        //           To distinguish them, I'll use x,y for the output image,
        //           and p,q for the image we are interpolating.
        //           i,j are used for the indices in the output array as usual.

        // Find the min/max x and y values where the output can be nonzero.
        double minx = _nonzero_bounds.getXMin() - _xInterp.xrange();
        double maxx = _nonzero_bounds.getXMax() + _xInterp.xrange();
        double miny = _nonzero_bounds.getYMin() - _xInterp.xrange();
        double maxy = _nonzero_bounds.getYMax() + _xInterp.xrange();
        dbg<<"nonzero bounds = "<<_nonzero_bounds<<std::endl;
        dbg<<"min/max x/y = "<<minx<<"  "<<maxx<<"  "<<miny<<"  "<<maxy<<std::endl;

        // Figure out what range of i,j these correspond to.
        int i1 = int((minx-x0)/dx);
        int i2 = int((maxx-x0)/dx);
        int j1 = int((miny-y0)/dy);
        int j2 = int((maxy-y0)/dy);
        if (i2 < i1) std::swap(i1,i2);
        if (j2 < j1) std::swap(j1,j2);
        ++i2; ++j2;  // Make them one past the end, rather than last index to use.
        if (i1 < 0) i1 = 0;
        if (i2 > m) i2 = m;
        if (j1 < 0) j1 = 0;
        if (j2 > n) j2 = n;
        xdbg<<"Output bounds = "<<im.getBounds()<<std::endl;
        xdbg<<"Old i,j ranges = "<<0<<"  "<<m<<"  "<<0<<"  "<<n<<std::endl;
        xdbg<<"New i,j ranges = "<<i1<<"  "<<i2<<"  "<<j1<<"  "<<j2<<std::endl;
        if (i1 >= m || i2 < 0 || j1 >= n || j2 < 0) return;

        // Fix up x0, y0, ptr, skip to correspond to these i,j ranges.
        x0 += i1*dx;
        y0 += j1*dy;
        if (x0 < minx || x0 > maxx) { x0 += dx; ++i1; } // First points may be able to increase
        if (y0 < miny || y0 > maxy) { y0 += dy; ++j1; } // by one more spot.
        ptr += i1 + j1*im.getStride();
        int mm = i2-i1;  // We'll need this new row length a few times below.
        skip += (m - mm);

        // Each point in the output image is going to be
        //
        //     I(x,y) = Sum_p,q wt(p-x) wt(q-y) _image(p,q)
        //
        // We have a lot of points with the same x or the same y, which means that
        // xwt and ywt values are repeated a lot.  So we want to find a way to reuse
        // those _xInterp.xval values.
        //
        // Finally, it is most efficient if we have the innermost loop be along the
        // direction with step=1 in both the input and output images.
        //
        // The way we try to meet all these goals is to build up the output image one row
        // at a time:
        //
        //     I(:,j) = Sum_q wt(q-y_j) rowq(:)
        //
        // where rowq is an array the same length as the output row.
        // Then for each of these y,q values, we can compute rowq as
        //
        //     rowq(i) = Sum_p wt(p-x_i) _image(p,q)
        //
        // The same set of wt(p-x) values are needed for every rowq calculation, so we
        // compute them once and save them.  Furthermore, the complete rowq calculation for
        // a given q is independent of y, so we save that as well.

        double x = x0;
        double xwt[_xInterp.ixrange() * mm];
        double p1ar[mm];
        double p2ar[mm];
        int k=0;
        for (int i=i1; i<i2; ++i,x+=dx) {
            int p1,p2;
            // If x is (basically) an integer, only 1 p value.
            // Otherwise, have a range based on xInterp.xrange()
            if (std::abs(x-std::floor(x+0.01)) < SMALL*(std::abs(x)+1)) {
                p1 = p2 = int(std::floor(x+0.01));
            } else {
                p1 = int(std::ceil(x-_xInterp.xrange()));
                p2 = int(std::floor(x+_xInterp.xrange()));
            }
            // Limit to nonzero region
            if (p1 < _nonzero_bounds.getXMin()) p1 = _nonzero_bounds.getXMin();
            if (p2 > _nonzero_bounds.getXMax()) p2 = _nonzero_bounds.getXMax();
            p1ar[i-i1] = p1;
            p2ar[i-i1] = p2;
            xdbg<<"i = "<<i<<"  x = "<<x<<": p1,p2 = "<<p1<<','<<p2<<std::endl;
            assert(p2-p1+1 <= _xInterp.ixrange());

            for (int p=p1; p<=p2; ++p) {
                xassert(k < _xInterp.ixrange()*mm);
                xwt[k++] = _xInterp.xval(p-x);
            }
        }

        // The inner calculation for rowq is the same for multiple y values since it is
        // independent of y, so each time we use the same q, the rowq array is the same.
        // Therefore we should cache these calculations to reuse when possible.
        std::map<int, std::vector<double> > rowq_cache;

        im.setZero();
        double y = y0;
        double temp[mm];
        for (int j=j1; j<j2; ++j,y+=dy,ptr+=skip) {
            memset(temp, 0, mm * sizeof(double)); // Zero out temp array
            xdbg<<"j = "<<j<<", y = "<<y<<std::endl;
            // If y is (basically) an integer, only 1 q value.
            // Otherwise, have a range based on xInterp.xrange()
            // Subtlety: also keep track of the minimum q we want to keep in the cache, which
            // may be less than q1 to account for sometimes y being integer, sometimes not.
            int q1,q2,qmin;
            if (std::abs(y-std::floor(y+0.01)) < SMALL*(std::abs(y)+1)) {
                q1 = q2 = int(std::floor(y+0.01));
                qmin = int(std::ceil(y-_xInterp.xrange()));
            } else {
                qmin = q1 = int(std::ceil(y-_xInterp.xrange()));
                q2 = int(std::floor(y+_xInterp.xrange()));
            }
            xdbg<<"q1,q2 = "<<q1<<','<<q2<<std::endl;
            // Limit to nonzero region
            if (q1 < _nonzero_bounds.getYMin()) q1 = _nonzero_bounds.getYMin();
            if (q2 > _nonzero_bounds.getYMax()) q2 = _nonzero_bounds.getYMax();
            xdbg<<"q1,q2 => "<<q1<<','<<q2<<std::endl;

            // Dump any cached rows we don't need anymore.
            while (rowq_cache.size() > 0 && rowq_cache.begin()->first < qmin) {
                rowq_cache.erase(rowq_cache.begin());
            }

            for (int q=q1; q<=q2; ++q) {
                // Get rowq from cache.  If it isn't there, it will be an empty vector.
                std::vector<double>& rowq = rowq_cache[q];

                // If this rowq was not in cache, need to make it.
                if (rowq.size() == 0) {
                    rowq.resize(mm);
                    double x = x0;
                    int k=0;
                    std::vector<double>::iterator row_it=rowq.begin();
                    for (int i=i1; i<i2; ++i,x+=dx,++row_it) {
                        *row_it = 0.;
                        int p1 = p1ar[i-i1];
                        int p2 = p2ar[i-i1];
                        const double* imptr = &_image(p1,q);
                        for (int p=p1; p<=p2; ++p) {
                            *row_it += xwt[k++] * *imptr++;
                        }
                    }
                }

                // Now add that to the output row with the ywt scaling.
                double ywt = _xInterp.xval(q-y);
                double* tptr = temp;
                std::vector<double>::const_iterator row_it = rowq.begin();
                for (int i=i1; i<i2; ++i) {
                    *tptr++ += *row_it++ * ywt;
                }
            }
            // Now finally copy onto the real output image.
            // Note: In addition to getting a tiny efficiency gain from having temp on
            // the stack while doing the calculation above, this is also important for
            // accuracy if the output image is T=float, so we don't gratuitously
            // lose precision by adding floats rather than doubles.
            double* tptr = temp;
            for (int i=i1; i<i2; ++i) *ptr++ = *tptr++;
        }
        dbg<<"Done SBInterpolatedImage fillXImage\n";
    }

    template <typename T>
    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<T> im,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBInterpolatedImage fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        T* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        // In this version every _xInterp.xval call is different, so there's not really any
        // way to save any of those calls.  The only real optimization that still applies
        // is the min/max checks to skip places where the output is zero.

        // Find the min/max x and y values where the output can be nonzero.
        double minx = _nonzero_bounds.getXMin() - _xInterp.xrange();
        double maxx = _nonzero_bounds.getXMax() + _xInterp.xrange();
        double miny = _nonzero_bounds.getYMin() - _xInterp.xrange();
        double maxy = _nonzero_bounds.getYMax() + _xInterp.xrange();
        dbg<<"nonzero bounds = "<<_nonzero_bounds<<std::endl;
        dbg<<"min/max x/y = "<<minx<<"  "<<maxx<<"  "<<miny<<"  "<<maxy<<std::endl;

        // Figure out what range of i,j these correspond to.
        // This is a bit more complicated than the separable case.
        // We need to find the i,j corresponding to each corner of the allowed range.
        //     x = x0 + idx + jdxy
        //     y = y0 + idyx + jdy
        // ->  i = ( (x-x0) dy - (y-y0) dxy ) / (dx dy - dxy dyx)
        //     j = ( (y-y0) dx - (x-x0) dyx ) / (dx dy - dxy dyx)
        double denom = dx*dy - dxy*dyx;
        int ia = int(((minx-x0)*dy - (miny-y0)*dxy)/denom);
        int ja = int(((miny-y0)*dx - (minx-x0)*dyx)/denom);
        int ib = int(((minx-x0)*dy - (maxy-y0)*dxy)/denom);
        int jb = int(((maxy-y0)*dx - (minx-x0)*dyx)/denom);
        int ic = int(((maxx-x0)*dy - (miny-y0)*dxy)/denom);
        int jc = int(((miny-y0)*dx - (maxx-x0)*dyx)/denom);
        int id = int(((maxx-x0)*dy - (maxy-y0)*dxy)/denom);
        int jd = int(((maxy-y0)*dx - (maxx-x0)*dyx)/denom);
        dbg<<"Corners at "<<ia<<','<<ja<<"  "<<ib<<','<<jb<<"  "<<ic<<','<<jc<<"  "<<id<<','<<jd<<std::endl;
        int i1 = std::min( {ia,ib,ic,id} );
        int i2 = std::max( {ia,ib,ic,id} );
        int j1 = std::min( {ja,jb,jc,jd} );
        int j2 = std::max( {ja,jb,jc,jd} );
        dbg<<"i,j ranges = "<<i1<<"  "<<i2<<"  "<<j1<<"  "<<j2<<std::endl;

        ++i2; ++j2;  // Make them one past the end, rather than last index to use.
        if (i1 < 0) i1 = 0;
        if (i2 > m) i2 = m;
        if (j1 < 0) j1 = 0;
        if (j2 > n) j2 = n;
        dbg<<"Output bounds = "<<im.getBounds()<<std::endl;
        dbg<<"Old i,j ranges = "<<0<<"  "<<m<<"  "<<0<<"  "<<n<<std::endl;
        dbg<<"New i,j ranges = "<<i1<<"  "<<i2<<"  "<<j1<<"  "<<j2<<std::endl;
        if (i1 >= m || i2 < 0 || j1 >= n || j2 < 0) return;

        // Fix up x0, y0, ptr, skip to correspond to these i,j ranges.
        x0 += i1*dx + j1*dxy;
        y0 += j1*dy + i1*dyx;
        ptr += i1 + j1*im.getStride();
        int mm = i2-i1;
        skip += (m - mm);

        im.setZero();
        for (int j=j1; j<j2; ++j,x0+=dxy,y0+=dy,ptr+=skip) {
            double x = x0;
            double y = y0;

            for (int i=i1; i<i2; ++i,x+=dx,y+=dyx,++ptr) {
                // Still want this check even with above i1,i2,j1,j2 stuff, since projected
                // region is a parallelogram, so some points can still be sipped.
                if (y > maxy || y < miny || x > maxx || x < minx) continue;

                int p1 = int(std::ceil(x-_xInterp.xrange()));
                int p2 = int(std::floor(x+_xInterp.xrange()));
                int q1 = int(std::ceil(y-_xInterp.xrange()));
                int q2 = int(std::floor(y+_xInterp.xrange()));
                if (p1 < _nonzero_bounds.getXMin()) p1 = _nonzero_bounds.getXMin();
                if (p2 > _nonzero_bounds.getXMax()) p2 = _nonzero_bounds.getXMax();
                if (q1 < _nonzero_bounds.getYMin()) q1 = _nonzero_bounds.getYMin();
                if (q2 > _nonzero_bounds.getYMax()) q2 = _nonzero_bounds.getYMax();

                double xwt[p2-p1+1];
                for (int p=p1, pp=0; p<=p2; ++p, ++pp) {
                    xwt[pp] = _xInterp.xval(p-x);
                }

                double sum=0.;
                for (int q=q1; q<=q2; ++q) {
                    double ywt = _xInterp.xval(q-y);

                    double xsum = 0.;
                    const double* imptr = &_image(p1,q);
                    for (int p=p1, pp=0; p<=p2; ++p, ++pp) {
                        xsum += xwt[pp] * *imptr++;
                    }
                    sum += xsum * ywt;
                }
                xassert(ptr >= im.getData());
                xassert(ptr < im.getData() + im.getNElements());
                *ptr = sum;
            }
        }
    }

    template <typename T>
    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKImage(
        ImageView<std::complex<T> > im,
        double kx0, double dkx, int izero,
        double ky0, double dky, int jzero) const
    {
        dbg<<"SBInterpolatedImage fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        assert(im.getStep() == 1);
        int skip = im.getNSkip();
        checkK();
        const double SMALL = 10.*std::numeric_limits<double>::epsilon();

        // Only non-zero where |u| <= maxu
        double absdkx = std::abs(dkx);
        double absdky = std::abs(dky);
        int i1 = std::max( int(-_maxk1/absdkx-kx0/dkx) , 0 );
        int i2 = std::min( int(_maxk1/absdkx-kx0/dkx)+1 , m );
        int j1 = std::max( int(-_maxk1/absdky-ky0/dky) , 0 );
        int j2 = std::min( int(_maxk1/absdky-ky0/dky)+1 , n );
        dbg<<"i1,i2,j1,j1 = "<<i1<<','<<i2<<','<<j1<<','<<j2<<std::endl;
        if (i1 >= m || i2 < 0 || j1 >= n || j2 < 0) return;

        kx0 += i1*dkx;
        ky0 += j1*dky;
        ptr += i1 + j1*im.getStride();
        int mm = i2-i1;
        skip += (m - mm);

        // For the rest of the range, calculate ux, uy values
        std::vector<double> ux(i2-i1);
        typedef std::vector<double>::iterator It;
        It uxit = ux.begin();
        double kx = kx0;
        for (int i=i1; i<i2; ++i,kx+=dkx) *uxit++ = kx * _uscale;

        std::vector<double> uy(j2-j1);
        It uyit = uy.begin();
        double ky = ky0;
        for (int j=j1; j<j2; ++j,ky+=dky) *uyit++ = ky * _uscale;

        // Rescale the k values by kscale
        int No2 = _kimage->getBounds().getXMax();
        int N = No2 * 2;
        double kscale = No2/M_PI;
        dbg<<"kimage bounds = "<<_kimage->getBounds()<<", scale = "<<kscale<<std::endl;
        kx0 *= kscale;
        ky0 *= kscale;
        dkx *= kscale;
        dky *= kscale;

        // The caching stuff is the same here as it was for fillXValue.  The only difference
        // is that we need to wrap around the p,q values and handle the conjugation possibility
        // correctly.  (cf. comments in kValue method.)
        kx = kx0;
        double xwt[_kInterp.ixrange() * mm];
        double p1ar[mm];
        double p2ar[mm];
        int k=0;
        for (int i=i1; i<i2; ++i,kx+=dkx) {
            int p1, p2;  // Range over which we need to sum.
            if (std::abs(kx-std::floor(kx+0.01)) < SMALL*(std::abs(kx)+1)) {
                // If kx is integer, only sum 1 element in that direction.
                p1 = p2 = int(std::floor(kx+0.01));
            } else {
                p1 = int(std::ceil(kx-_kInterp.xrange()));
                p2 = int(std::floor(kx+_kInterp.xrange()));
            }
            p1ar[i-i1] = p1;
            p2ar[i-i1] = p2;
            xdbg<<"i = "<<i<<"  kx = "<<kx<<": p1,p2 = "<<p1<<','<<p2<<std::endl;
            assert(p2-p1+1 <= _kInterp.ixrange());

            for (int p=p1; p<=p2; ++p) {
                xassert(k < _kInterp.ixrange()*mm);
                xwt[k++] = _kInterp.xval(p-kx);
            }
        }

        // Again, can cache the rowq vectors.
        std::map<int, std::vector<std::complex<double> > > rowq_cache;

        // Pre-calculate xInterp factors in place
        uxit = ux.begin();
        for (int i=i1; i<i2; ++i,++uxit) *uxit = _xInterp.uval(*uxit);
        uyit = uy.begin();
        for (int j=j1; j<j2; ++j,++uyit) *uyit = _xInterp.uval(*uyit);

        im.setZero();
        ky = ky0;
        uyit = uy.begin();
        double temp[2*mm]; // Can't put complex<double> array on stack, so reinterpret_cast below.
        for (int j=j1; j<j2; ++j,ky+=dky,ptr+=skip,++uyit) {
            memset(temp, 0, 2*mm * sizeof(double));
            xdbg<<"j = "<<j<<", ky = "<<ky<<std::endl;
            // If y is (basically) an integer, only 1 q value.
            int q1,q2,qmin;
            if (std::abs(ky-std::floor(ky+0.01)) < SMALL*(std::abs(ky)+1)) {
                q1 = q2 = int(std::floor(ky+0.01));
                qmin = int(std::ceil(ky-_kInterp.xrange()));
            } else {
                qmin = q1 = int(std::ceil(ky-_kInterp.xrange()));
                q2 = int(std::floor(ky+_kInterp.xrange()));
            }
            xdbg<<"q1,q2 = "<<q1<<','<<q2<<std::endl;

            // Dump any cached rows we don't need anymore.
            while (rowq_cache.size() > 0 && rowq_cache.begin()->first < qmin) {
                rowq_cache.erase(rowq_cache.begin());
            }

            int qwrap1 = WrapKIndex(q1, No2, N);
            for (int q=q1, qwrap=qwrap1; q<=q2; ++q, ++qwrap) {
                if (qwrap == No2) qwrap -= N;

                // Get rowq from cache.  If it isn't there, it will be an empty vector.
                std::vector<std::complex<double> >& rowq = rowq_cache[q];

                // If this rowq was not in cache, need to make it.
                if (rowq.size() == 0) {
                    rowq.resize(mm);
                    kx = kx0;
                    int k=0;
                    std::vector<std::complex<double> >::iterator row_it=rowq.begin();
                    for (int i=i1; i<i2; ++i,kx+=dkx) {
                        int p1 = p1ar[i-i1];
                        int p2 = p2ar[i-i1];
                        int pwrap1 = WrapKIndex(p1, No2, N);
                        *row_it++ = KValueInnerLoop(p2-p1+1,pwrap1,qwrap,No2,N,&xwt[k],*_kimage);
                        k += p2-p1+1;
                    }
                }

                // Now add that to the output row with the ywt scaling.
                double ywt = _kInterp.xval(q-ky);
                std::complex<double>* tptr = reinterpret_cast<std::complex<double>*>(temp);
                std::vector<std::complex<double> >::const_iterator row_it = rowq.begin();
                for (int i=i1; i<i2; ++i) {
                    xassert(row_it < rowq.end());
                    xassert((void*)tptr < (void*)(temp + 2*mm));
                    *tptr++ += *row_it++ * ywt;
                }
            }

            // Now account for the x-interpolant
            // And finally copy onto the real output image.
            // Note: In addition to getting a tiny efficiency gain from having temp on
            // the stack while doing the calculation above, this is also important for
            // accuracy if the output image is complex<float>, so we don't gratuitously
            // lose precision by adding floats rather than doubles.
            uxit = ux.begin();
            std::complex<double>* tptr = reinterpret_cast<std::complex<double>*>(temp);
            for (int i=i1; i<i2; ++i) {
                xassert(ptr < im.getData() + im.getNElements());
                xassert(uxit < ux.end());
                xassert(uyit < uy.end());
                xassert((void*)tptr < (void*)(temp + 2*mm));
                *ptr++ = *uxit++ * *uyit * *tptr++;
            }
        }

        dbg<<"Done SBInterpolatedImage fillKImage\n";
    }

    template <typename T>
    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKImage(
        ImageView<std::complex<T> > im,
        double kx0, double dkx, double dkxy,
        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBInterpolatedImage fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);
        checkK();

        double ux0 = kx0 * _uscale;
        double uy0 = ky0 * _uscale;
        double dux = dkx * _uscale;
        double duy = dky * _uscale;
        double duxy = dkxy * _uscale;
        double duyx = dkyx * _uscale;

        int No2 = _kimage->getBounds().getXMax();
        int N = No2 * 2;
        double kscale = No2/M_PI;
        kx0 *= kscale;
        dkx *= kscale;
        dkxy *= kscale;
        ky0 *= kscale;
        dky *= kscale;
        dkyx *= kscale;
        double maxk1 = _maxk1 * kscale;

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ux0+=duxy,uy0+=duy,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            double ux = ux0;
            double uy = uy0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx,ux+=dux,uy+=duyx) {
                if (std::abs(kx) > maxk1 || std::abs(ky) > maxk1) {
                    *ptr++ = T(0);
                } else {
                    int p1 = int(std::ceil(kx-_kInterp.xrange()));
                    int p2 = int(std::floor(kx+_kInterp.xrange()));
                    int q1 = int(std::ceil(ky-_kInterp.xrange()));
                    int q2 = int(std::floor(ky+_kInterp.xrange()));

                    double xwt[p2-p1+1];
                    for (int p=p1, pp=0; p<=p2; ++p, ++pp) xwt[pp] = _kInterp.xval(p-kx);

                    std::complex<double> sum = 0.;
                    int pwrap1 = WrapKIndex(p1, No2, N);
                    int qwrap1 = WrapKIndex(q1, No2, N);
                    for (int q=q1, qwrap=qwrap1; q<=q2; ++q, ++qwrap) {
                        if (qwrap == No2) qwrap -= N;
                        double ywt = _kInterp.xval(q-ky);
                        sum += ywt * KValueInnerLoop(p2-p1+1,pwrap1,qwrap,No2,N,xwt,*_kimage);
                    }
                    *ptr++ = _xInterp.uval(ux) * _xInterp.uval(uy) * sum;
                }
            }
        }
    }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getPaddedImage() const
    { return _image; }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getNonZeroImage() const
    { return _image[_nonzero_bounds]; }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getImage() const
    { return _image[_init_bounds]; }

    void SBInterpolatedImage::SBInterpolatedImageImpl::getXRange(
        double& xmin, double& xmax, std::vector<double>& splits) const
    {
        Bounds<int> b = _init_bounds;
        double xrange = _xInterp.xrange();
        int N = b.getXMax()-b.getXMin()+1;
        xmin = -(N/2 + xrange);
        xmax = ((N-1)/2 + xrange);
        int ixrange = _xInterp.ixrange();
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
        double xrange = _xInterp.xrange();
        int N = b.getYMax()-b.getYMin()+1;
        ymin = -(N/2 + xrange);
        ymax = ((N-1)/2 + xrange);
        int ixrange = _xInterp.ixrange();
        if (ixrange > 0) {
            splits.resize(N-2+ixrange);
            double y = ymin-0.5*(ixrange-2);
            for(int i=0;i<N-2+ixrange;++i, ++y) splits[i] = y;
        }
    }

    Position<double> SBInterpolatedImage::SBInterpolatedImageImpl::centroid() const
    {
        if (_xcentroid == INVALID) {
            double flux = getFlux();
            if (flux == 0.) throw std::runtime_error("Flux == 0.  Centroid is undefined.");

            ConstImageView<double> image = getNonZeroImage();
            int xStart = -((image.getXMax()-image.getXMin()+1)/2);
            int y = -((image.getYMax()-image.getYMin()+1)/2);
            double sumx = 0.;
            double sumy = 0.;
            for (int iy = image.getYMin(); iy <= image.getYMax(); ++iy, ++y) {
                int x = xStart;
                for (int ix = image.getXMin(); ix <= image.getXMax(); ++ix, ++x) {
                    double value = image(ix,iy);
                    sumx += value*x;
                    sumy += value*y;
                }
            }
            _xcentroid = sumx/flux;
            _ycentroid = sumy/flux;
        }

        return Position<double>(_xcentroid, _ycentroid);
    }

    double SBInterpolatedImage::SBInterpolatedImageImpl::getFlux() const
    {
        if (_flux == INVALID) {
            _flux = 0.;
            ConstImageView<double> image = getNonZeroImage();
            for (int iy = image.getYMin(); iy <= image.getYMax(); ++iy) {
                for (int ix = image.getXMin(); ix <= image.getXMax(); ++ix) {
                    double value = image(ix,iy);
                    _flux += value;
                }
            }
        }
        return _flux;
    }

    double CalculateSizeContainingFlux(const BaseImage<double>& im, double target_flux)
    {
        dbg<<"Start CalculateSizeWithFlux\n";
        dbg<<"Find box that encloses flux = "<<target_flux<<std::endl;
        double p = target_flux > 0. ? 1 : -1;  // p for "positive" -- Make flux effectively > 0.

        const Bounds<int> b = im.getBounds();
        int dmax = std::min((b.getXMax()-b.getXMin())/2, (b.getYMax()-b.getYMin())/2);
        dbg<<"dmax = "<<dmax<<std::endl;
        double flux = im(0,0);
        int d=1;
        for (; d<=dmax; ++d) {
            xdbg<<"d = "<<d<<std::endl;
            xdbg<<"flux = "<<flux<<std::endl;
            // Add the left, right, top and bottom sides of box:
            for(int x = -d; x < d; ++x) {
                // Note: All 4 corners are added exactly once by including x=-d but omitting
                // x=d from the loop.
                flux += im(x,-d);  // bottom
                flux += im(d,x);   // right
                flux += im(-x,d);  // top
                flux += im(-d,-x); // left
            }
            if (p * flux >= p * target_flux) break;
        }
        dbg<<"Done: flux = "<<flux<<", d = "<<d<<std::endl;
        return d + 0.5;
    }

    // We provide an option to update the stepk value by directly calculating what
    // size region around the center encloses (1-folding_threshold) of the total flux.
    // This can be useful if you make the image bigger than you need to, just to be
    // safe, but then want to use as large a stepk value as possible.
    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateStepK(double max_stepk) const
    {
        dbg<<"Start SBInterpolatedImage calculateStepK()\n";
        dbg<<"Current value of stepk = "<<_stepk<<std::endl;
        dbg<<"Find box that encloses "<<1.-this->gsparams.folding_threshold<<" of the flux.\n";
        dbg<<"Max_stepk = "<<max_stepk<<std::endl;

        ConstImageView<double> im = getImage();
        double fluxTot = getFlux();
        double thresh = (1.-this->gsparams.folding_threshold) * fluxTot;
        dbg<<"thresh = "<<thresh<<std::endl;
        double R = CalculateSizeContainingFlux(im, thresh);

        dbg<<"R = "<<R<<std::endl;
        // Add xInterp range in quadrature just like convolution:
        double R2 = _xInterp.xrange();
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
            <<this->gsparams.maxk_threshold<<std::endl;
        checkK();

        int No2 = _kimage->getBounds().getXMax();
        double dk = M_PI/No2;

        // Among the elements with kval > thresh, find the one with the maximum ksq
        double thresh = this->gsparams.maxk_threshold * getFlux();
        thresh *= thresh; // Since values will be |kval|^2.
        double maxk_ix = 0.;
        // When we get 5 rows in a row all below thresh, stop.
        int n_below_thresh = 0;
        // Don't go past the current value of maxk
        if (max_maxk == 0.) max_maxk = _maxk;
        int max_ix = int(std::ceil(max_maxk / dk));
        if (max_ix > No2) max_ix = No2;

        // We take the k value to be maximum of kx and ky.  This is appropriate, because
        // this is how maxK() is eventually used -- it sets the size in k-space for both
        // kx and ky when drawing.  Since kx<0 is just the conjugate of the corresponding
        // point at (-kx,-ky), we only check the right half of the square.  i.e. the
        // upper-right and lower-right quadrants.
        for(int ix=0; ix<=max_ix; ++ix) {
            xdbg<<"Start search for ix = "<<ix<<std::endl;
            // Search along the two sides with either kx = ix or ky = ix.
            for(int iy=0; iy<=ix; ++iy) {
                // The bottom side of the square in the lower-right quadrant.
                double norm_kval = fast_norm((*_kimage)(iy,-ix));
                xdbg<<"norm_kval at "<<iy<<','<<-ix<<" = "<<norm_kval<<std::endl;
                if (norm_kval <= thresh && iy != ix && ix != No2) {
                    // The top side of the square in the upper-right quadrant.
                    norm_kval = fast_norm((*_kimage)(iy,ix));
                    xdbg<<"norm_kval at "<<iy<<','<<ix<<" = "<<norm_kval<<std::endl;
                }
                if (norm_kval <= thresh && iy > 0) {
                    // The right side of the square in the lower-right quadrant.
                    norm_kval = fast_norm((*_kimage)(ix,-iy));
                    xdbg<<"norm_kval at "<<ix<<','<<-iy<<" = "<<norm_kval<<std::endl;
                }
                if (norm_kval <= thresh && ix > 0 && iy != No2) {
                    // The right side of the square in the upper-right quadrant.
                    // The ky argument is wrapped to positive values.
                    norm_kval = fast_norm((*_kimage)(ix,iy));
                    xdbg<<"norm_kval at "<<ix<<','<<iy<<" = "<<norm_kval<<std::endl;
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

        Bounds<int> b = _nonzero_bounds;
        int xStart = -((b.getXMax()-b.getXMin()+1)/2);
        int y = -((b.getYMax()-b.getYMin()+1)/2);

        // We loop over the non-zero bounds, since this is the only region with any flux.
        //
        // ix,iy are the indices in the original image
        // x,y are the positions relative to the center point.
        for (int iy = b.getYMin(); iy<= b.getYMax(); ++iy, ++y) {
            int x = xStart;
            for (int ix = b.getXMin(); ix<= b.getXMax(); ++ix, ++x) {
                double flux = _image(ix,iy);
                if (flux==0.) continue;
                if (flux > 0.) {
                    _positiveFlux += flux;
                } else {
                    _negativeFlux += -flux;
                }
                _pt.push_back(shared_ptr<Pixel>(new Pixel(x,y,flux)));
            }
        }

        // The above just computes the positive and negative flux for the main image.
        // This is convolved by the interpolant, so we need to correct these values
        // in the same way that SBConvolve does:
        double p1 = _positiveFlux;
        double n1 = _negativeFlux;
        dbg<<"positiveFlux = "<<p1<<", negativeFlux = "<<n1<<std::endl;
        double p2 = _xInterp.getPositiveFlux2d();
        double n2 = _xInterp.getNegativeFlux2d();
        dbg<<"Interpolant has positiveFlux = "<<p2<<", negativeFlux = "<<n2<<std::endl;
        _positiveFlux = p1*p2 + n1*n2;
        _negativeFlux = p1*n2 + n1*p2;
        dbg<<"positiveFlux => "<<_positiveFlux<<", negativeFlux => "<<_negativeFlux<<std::endl;

        double thresh = std::numeric_limits<double>::epsilon() * (_positiveFlux + _negativeFlux);
        dbg<<"thresh = "<<thresh<<std::endl;
        _pt.buildTree(thresh);

        _readyToShoot = true;
    }

    // Photon-shooting
    void SBInterpolatedImage::SBInterpolatedImageImpl::shoot(
        PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
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

        if (N<=0 || _pt.empty()) return;
        double totalAbsFlux = _positiveFlux + _negativeFlux;
        double fluxPerPhoton = totalAbsFlux / N;
        dbg<<"posFlux = "<<_positiveFlux<<", negFlux = "<<_negativeFlux<<std::endl;
        dbg<<"totFlux = "<<_positiveFlux-_negativeFlux<<", totAbsFlux = "<<totalAbsFlux<<std::endl;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;
        for (int i=0; i<N; ++i) {
            double unitRandom = ud();
            const shared_ptr<Pixel> p = _pt.find(unitRandom);
            photons.setPhoton(i, p->x, p->y, p->isPositive ? fluxPerPhoton : -fluxPerPhoton);
        }
        dbg<<"photons.getTotalFlux = "<<photons.getTotalFlux()<<std::endl;

        // Last step is to convolve with the interpolation kernel.
        // Can skip if using a 2d delta function
        if (!dynamic_cast<const Delta*>(&_xInterp)) {
            PhotonArray temp(N);
            _xInterp.shoot(temp, ud);
            photons.convolve(temp, ud);
        }

        dbg<<"InterpolatedImage Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImage methods

    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<std::complex<double> >& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(kimage, stepk, kInterp, gsparams)) {}

    SBInterpolatedKImage::SBInterpolatedKImage(const SBInterpolatedKImage& rhs)
        : SBProfile(rhs) {}

    SBInterpolatedKImage::~SBInterpolatedKImage() {}

    const Interpolant& SBInterpolatedKImage::getKInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).getKInterp();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImageImpl methods

    // "Normal" constructor
    SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<std::complex<double> >& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _kimage(kimage.view()),
        _kInterp(kInterp), _stepk(stepk), _maxk(0.) //fill in maxk below
    {
        // Note that _stepk indicates the maximum pitch for drawImage() to use when rendering an
        // image, in units of the sampling pitch of the input images.
        // _stepk must be at least 1.0.
        assert(_stepk >= 1.0);

        dbg<<"stepk = "<<_stepk<<std::endl;
        dbg<<"kimage bounds = "<<kimage.getBounds()<<std::endl;

        int No2 = _kimage.getBounds().getXMax();
        _maxk = No2;
        dbg<<"_maxk = "<<_maxk<<std::endl;

        _flux = kValue(Position<double>(0.,0.)).real();
        dbg<<"flux = "<<_flux<<std::endl;
        setCentroid();
    }

    void SBInterpolatedKImage::SBInterpolatedKImageImpl::setCentroid() const
    {
        /*  Centroid:
            int x f(x) dx = (x conv f)|x=0 = int FT(x conv f)(k) dk
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
        int No2 = _kimage.getBounds().getXMax();
        double xsum(0.0), ysum(0.0);
        int iky = -No2;
        double sign = (iky % 2 == 0) ? 1.0 : -1.0;
        for (; iky < No2; iky++, sign = -sign) {
            if (iky == 0) continue;
            ysum += sign / iky * _kimage(0, iky).imag();
        }
        int ikx = -No2;
        sign = (ikx % 2 == 0) ? 1.0 : -1.0;
        for (; ikx < No2; ikx++, sign = -sign) {
            if (ikx == 0) continue;
            if (ikx < 0) {
                // kimage(i,0) = conj(kimage(-i,0)
                // so imag part gets a - sign.
                xsum -= sign / ikx * _kimage(-ikx, 0).imag();
            } else {
                xsum += sign / ikx * _kimage(ikx, 0).imag();
            }
        }
        _xcentroid = xsum/_flux;
        _ycentroid = ysum/_flux;
    }

    SBInterpolatedKImage::SBInterpolatedKImageImpl::~SBInterpolatedKImageImpl() {}

    const Interpolant& SBInterpolatedKImage::SBInterpolatedKImageImpl::getKInterp() const
    {
        return _kInterp;
    }

    std::complex<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::kValue(
        const Position<double>& kpos) const
    {
        // Basically the same as SBInterpolatedImag::kValue, but no x-interpolant to deal with.
        double kx = kpos.x;
        double ky = kpos.y;

        dbg<<"evaluating kValue("<<kx<<","<<ky<<")"<<std::endl;

        if (std::abs(kx) > _maxk || std::abs(ky) > _maxk) return std::complex<double>(0.,0.);

        int No2 = _kimage.getBounds().getXMax();
        int N = No2 * 2;
        xdbg<<"kimage bounds = "<<_kimage.getBounds()<<std::endl;

        int p1, p2, q1, q2;  // Range over which we need to sum.
        const double SMALL = 10.*std::numeric_limits<double>::epsilon();

        if (std::abs(kx-std::floor(kx+0.01)) < SMALL*(std::abs(kx)+1)) {
            // If kx or ky are integers, only sum 1 element in that direction.
            p1 = p2 = int(std::floor(kx+0.01));
        } else {
            p1 = int(std::ceil(kx-_kInterp.xrange()));
            p2 = int(std::floor(kx+_kInterp.xrange()));
        }
        if (std::abs(ky-std::floor(ky+0.01)) < SMALL*(std::abs(ky)+1)) {
            q1 = q2 = int(std::floor(ky+0.01));
        } else {
            q1 = int(std::ceil(ky-_kInterp.xrange()));
            q2 = int(std::floor(ky+_kInterp.xrange()));
        }
        dbg<<"p range = "<<p1<<"..."<<p2<<std::endl;
        dbg<<"q range = "<<q1<<"..."<<q2<<std::endl;

        // We'll need these for each row.  Save them.
        double xwt[p2-p1+1];
        for (int p=p1, pp=0; p<=p2; ++p, ++pp) xwt[pp] = _kInterp.xval(p-kx);

        std::complex<double> sum = 0.;
        int pwrap1 = WrapKIndex(p1, No2, N);
        int qwrap1 = WrapKIndex(q1, No2, N);
        dbg<<"pwrap, qwrap = "<<qwrap1<<','<<qwrap1<<std::endl;
        dbg<<"kimage bounds = "<<_kimage.getBounds()<<std::endl;
        for (int q=q1, qwrap=qwrap1; q<=q2; ++q, ++qwrap) {
            if (qwrap == No2) qwrap -= N;
            std::complex<double> xsum = KValueInnerLoop(p2-p1+1,pwrap1,qwrap,No2,N,xwt,_kimage);
            sum += xsum * _kInterp.xval(q-ky);
        }

        dbg<<"sum = "<<sum<<std::endl;
        return sum;
    }

    double SBInterpolatedKImage::SBInterpolatedKImageImpl::maxSB() const
    {
        // No easy way to even estimate this value, so just raise an exception.
        // Shouldn't be a problem, since this is really only used for photon shooting
        // and we can't photon shoot InterpolatedKImages anyway.
        throw std::runtime_error("InterpolatedKImage does not implement maxSB()");
        return 0.;
    }

    Position<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::centroid() const
    {
        double flux = getFlux();
        if (flux == 0.) throw std::runtime_error("Flux == 0.  Centroid is undefined.");
        return Position<double>(_xcentroid, _ycentroid);
    }

} // namespace galsim
