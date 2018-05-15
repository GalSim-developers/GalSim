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

#include <algorithm>
#include "SBInterpolatedImage.h"
#include "SBInterpolatedImageImpl.h"


namespace galsim {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedImage methods

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& image,
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

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<T>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams) :
        SBProfileImpl(gsparams), _image_bounds(image.getBounds()),
        _init_bounds(init_bounds), _nonzero_bounds(nonzero_bounds),
        _xInterp(xInterp), _kInterp(kInterp),
        _stepk(stepk), _maxk(maxk),
        _flux(-999.), _xcentroid(-999.), _ycentroid(-999.),
        _readyToShoot(false)
    {
        dbg<<"image bounds = "<<image.getBounds()<<std::endl;
        dbg<<"init bounds = "<<_init_bounds<<std::endl;
        dbg<<"nonzero bounds = "<<_nonzero_bounds<<std::endl;

        int Ninitx = image.getXMax()-image.getXMin()+1;
        //int Ninity = image.getYMax()-image.getYMin()+1;
        //assert(Ninitx == Ninity);  // (Ensured by the python layer.)
        _Nk = Ninitx;
        _image_bounds = image.getBounds();
        dbg<<"_Nk = "<<_Nk<<std::endl;

        // Copy the input image to an XTable
        _xtab = shared_ptr<XTable>(new XTable(_Nk, 1.));
        ImageView<double> xtab_view(_xtab->getArray(), shared_ptr<double>(),
                                    1, _Nk, _image_bounds);
        xtab_view.copyFrom(image);

        dbg<<"N = "<<_Nk<<", xrange = "<<_xInterp.xrange()<<std::endl;
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
    { return _xInterp.get1d(); }

    const Interpolant& SBInterpolatedImage::SBInterpolatedImageImpl::getKInterp() const
    { return _kInterp.get1d(); }

    double SBInterpolatedImage::SBInterpolatedImageImpl::maxSB() const
    {
        // Find the pixel with the largest (absolute) value.
        double maxsb = 0.;

        Bounds<int> b = _nonzero_bounds;
        int xStart = -((b.getXMax()-b.getXMin()+1)/2);
        int y = -((b.getYMax()-b.getYMin()+1)/2);

        for (int iy = b.getYMin(); iy<= b.getYMax(); ++iy, ++y) {
            int x = xStart;
            for (int ix = b.getXMin(); ix<= b.getXMax(); ++ix, ++x) {
                double sb = _xtab->xval(x,y);
                if (std::abs(sb) > maxsb) maxsb = std::abs(sb);
            }
        }
        // Since xtab stores surface brightness (not flux), this is directly the value we want.
        // i.e. no need to account for any pixel scale factor.
        return maxsb;
    }

    double SBInterpolatedImage::SBInterpolatedImageImpl::xValue(const Position<double>& p) const
    { return _xtab->interpolate(p.x, p.y, _xInterp); }

    std::complex<double> SBInterpolatedImage::SBInterpolatedImageImpl::kValue(
        const Position<double>& k) const
    {
        // Don't bother if the desired k value is cut off by the x interpolant:
        if (std::abs(k.x) > _maxk1 || std::abs(k.y) > _maxk1) return std::complex<double>(0.,0.);
        checkK();
        double xKernelTransform = _xInterp.uval(k.x*_uscale, k.y*_uscale);
        return xKernelTransform * _ktab->interpolate(k.x, k.y, _kInterp);
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkK() const
    {
        // Conduct FFT
        if (_ktab.get()) return;
        _ktab = _xtab->transform();
        dbg<<"Built ktab\n";
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
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
        assert(im.getStep() == 1);

        // The XTable interpolation routine will go faster if we make y iteration the
        // inner loop.
        const int stride = im.getStride();
        const int skip = 1 - n*stride;
        for (int i=0; i<m; ++i,x0+=dx,ptr+=skip) {
            double y = y0;
            for (int j=0; j<n; ++j,y+=dy,ptr+=stride)
                *ptr = _xtab->interpolate(x0, y, _xInterp);
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
        checkK();

        // Only non-zero where |u| <= maxu
        double absdkx = std::abs(dkx);
        double absdky = std::abs(dky);
        int i1 = std::max( int(-_maxk1/absdkx-kx0/dkx) , 0 );
        int i2 = std::min( int(_maxk1/absdkx-kx0/dkx)+1 , m );
        int j1 = std::max( int(-_maxk1/absdky-ky0/dky) , 0 );
        int j2 = std::min( int(_maxk1/absdky-ky0/dky)+1 , n );

        kx0 += i1*dkx;
        ky0 += j1*dky;
        ptr += i1 + j1*im.getStride();
        xdbg<<"i1,i2,j1,j2 = "<<i1<<','<<i2<<','<<j1<<','<<j2<<"  kx0,ky0 = "<<kx0<<','<<ky0<<std::endl;

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

        im.setZero();
        const int stride = im.getStride();
        const int skip = 1 - (j2-j1)*stride;
        // Then the uval's are separable.  Go ahead and pre-calculate them.
        // Note: We only have separable interpolant, so this is the only branch ever used.
        uxit = ux.begin();
        for (int i=i1; i<i2; ++i,++uxit) *uxit = _xInterp.get1d().uval(*uxit);
        uyit = uy.begin();
        for (int j=j1; j<j2; ++j,++uyit) *uyit = _xInterp.get1d().uval(*uyit);

        uxit = ux.begin();
        for (int i=i1; i<i2; ++i,kx0+=dkx,++uxit,ptr+=skip) {
            double ky = ky0;
            uyit = uy.begin();
            for (int j=j1; j<j2; ++j,ky+=dky,ptr+=stride)
                *ptr = *uxit * *uyit++ * _ktab->interpolate(kx0, ky, _kInterp);
        }
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

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ux0+=duxy,uy0+=duy,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            double ux = ux0;
            double uy = uy0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx,ux+=dux,uy+=duyx) {
                if (std::abs(kx) > _maxk1 || std::abs(ky) > _maxk1) {
                    *ptr++ = T(0);
                } else {
                    double xKernelTransform = _xInterp.uval(ux, uy);
                    *ptr++ = xKernelTransform * _ktab->interpolate(kx, ky, _kInterp);
                }
            }
        }
    }

    std::string SBInterpolatedImage::SBInterpolatedImageImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBInterpolatedImage(";
        oss << "galsim.ImageD(array([";

        ConstImageView<double> im = getPaddedImage();
        const double* ptr = im.getData();
        const int skip = im.getNSkip();
        const int step = im.getStep();
        const int xmin = im.getXMin();
        const int xmax = im.getXMax();
        const int ymin = im.getYMin();
        const int ymax = im.getYMax();
        for (int j=ymin; j<=ymax; j++, ptr+=skip) {
            if (j > ymin) oss <<",";
            oss << "[" << *ptr;
            ptr += step;
            for (int i=xmin+1; i<=xmax; i++, ptr+=step) oss << "," << *ptr;
            oss << "]";
        }

        oss<<"],dtype=float)).image, ";

        oss << getXInterp().makeStr()<<", ";
        oss << getKInterp().makeStr()<<", ";
        oss << stepK()<<", "<<maxK()<<", ";
        oss << "galsim._galsim.GSParams("<<gsparams<<"))";

        return oss.str();
    }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getPaddedImage() const
    {
        return ConstImageView<double>(_xtab->getArray(), shared_ptr<double>(),
                                      1, _Nk, _image_bounds);
    }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getNonZeroImage() const
    {
        return getPaddedImage()[_nonzero_bounds];
    }

    ConstImageView<double> SBInterpolatedImage::SBInterpolatedImageImpl::getImage() const
    {
        return getPaddedImage()[_init_bounds];
    }

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
        if (_xcentroid == -999.) {
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
        if (_flux == -999.) {
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

    template <typename T>
    double CalculateSizeContainingFlux(const BaseImage<T>& im, double target_flux)
    {
        dbg<<"Start CalculateSizeWithFlux\n";
        dbg<<"Find box that encloses flux = "<<target_flux<<std::endl;

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
            if (flux >= target_flux) break;
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
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;

        double dk = _ktab->getDk();

        // Among the elements with kval > thresh, find the one with the maximum ksq
        double thresh = this->gsparams.maxk_threshold * getFlux();
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

        Bounds<int> b = _nonzero_bounds;
        int xStart = -((b.getXMax()-b.getXMin()+1)/2);
        int y = -((b.getYMax()-b.getYMin()+1)/2);

        // We loop over the non-zero bounds, since this is the only region with any flux.
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

        // The above just computes the positive and negative flux for the main image.
        // This is convolved by the interpolant, so we need to correct these values
        // in the same way that SBConvolve does:
        double p1 = _positiveFlux;
        double n1 = _negativeFlux;
        dbg<<"positiveFlux = "<<p1<<", negativeFlux = "<<n1<<std::endl;
        double p2 = _xInterp.getPositiveFlux();
        double n2 = _xInterp.getNegativeFlux();
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
            const Pixel* p = _pt.find(unitRandom);
            photons.setPhoton(i, p->x, p->y, p->isPositive ? fluxPerPhoton : -fluxPerPhoton);
        }
        dbg<<"photons.getTotalFlux = "<<photons.getTotalFlux()<<std::endl;

        // Last step is to convolve with the interpolation kernel.
        // Can skip if using a 2d delta function
        if (!dynamic_cast<const Delta*>(&_xInterp.get1d())) {
            PhotonArray temp(N);
            _xInterp.shoot(temp, ud);
            temp.scaleXY(_xtab->getDx());
            photons.convolve(temp, ud);
        }

        dbg<<"InterpolatedImage Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImage methods

    template <typename T>
    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<T>& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(kimage, stepk, kInterp, gsparams)) {}

    SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<double>& data, double stepk, double maxk,
        const Interpolant& kInterp, const GSParams& gsparams) :
        SBProfile(new SBInterpolatedKImageImpl(data, stepk, maxk, kInterp, gsparams)) {}

    SBInterpolatedKImage::SBInterpolatedKImage(const SBInterpolatedKImage& rhs)
        : SBProfile(rhs) {}

    SBInterpolatedKImage::~SBInterpolatedKImage() {}

    const Interpolant& SBInterpolatedKImage::getKInterp() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).getKInterp();
    }

    ConstImageView<double> SBInterpolatedKImage::getKData() const
    {
        assert(dynamic_cast<const SBInterpolatedKImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedKImageImpl&>(*_pimpl).getKData();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SBInterpolatedKImageImpl methods

    // "Normal" constructor
    template <typename T>
    SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<T>& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _kInterp(kInterp), _stepk(stepk), _maxk(0.) //fill in maxk below
    {
        // Note that _stepk indicates the maximum pitch for drawImage() to use when rendering an
        // image, in units of the sampling pitch of the input images.  _stepk must be greater than
        // 1.0
        assert(_stepk >= 1.0);

        dbg<<"stepk = "<<_stepk<<std::endl;
        dbg<<"kimage bounds = "<<kimage.getBounds()<<std::endl;

        _Ninitx = kimage.getXMax()-kimage.getXMin()+1;
        _Ninity = kimage.getYMax()-kimage.getYMin()+1;
        _Ninitial = std::max(_Ninitx, _Ninity);
        dbg<<"_Ninitial = "<<_Ninitial<<std::endl;
        _Nk = goodFFTSize(int(_Ninitial));
        dbg<<"_Nk = "<<_Nk<<std::endl;

        _ktab = shared_ptr<KTable>(new KTable(_Nk, 1.0));
        _maxk = _Ninitial/2;
        dbg<<"_maxk = "<<_maxk<<std::endl;

        // Only need to fill in x>=0 since the negative x's are the Hermitian
        // conjugates of the positive x's.
        int kxStart = 0;
        int ikxStart = (kimage.getXMin()+kimage.getXMax()+1)/2;
        int ky = -((kimage.getYMax()-kimage.getYMin()+1)/2);
        dbg<<"kxStart = "<<kxStart<<", kyStart = "<<ky<<std::endl;
        for (int iky = kimage.getYMin(); iky<= kimage.getYMax(); ++iky, ++ky) {
             int kx = kxStart;
             for (int ikx = ikxStart; ikx<= kimage.getXMax(); ++ikx, ++kx) {
                 std::complex<double> kvalue = kimage(ikx, iky);
                 _ktab->kSet(kx, ky, kvalue);
                 xxdbg<<"ikx,iky,kx,ky = "<<ikx<<','<<iky<<','<<kx<<','<<ky<<std::endl;
                 xxdbg<<"kvalue = "<<kvalue<<std::endl;
             }
        }
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
        _xcentroid = xsum/_flux;
        _ycentroid = ysum/_flux;
    }

    // "Serialization" constructor.  Only used when unpickling an InterpolatedKImage.
    // Note *not* a template, since getKData() only returns doubles.
    SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<double>& data, double stepk, double maxk,
        const Interpolant& kInterp,
        const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _kInterp(kInterp), _stepk(stepk), _maxk(maxk)
    {
        dbg << "Using alternative constructor" << std::endl;
        _Nk = 2*(data.getYMax() - data.getYMin());
        dbg << "_Nk = " << _Nk << std::endl;
        // Original _Ninitial could have been smaller, but setting it equal to _Nk should be
        // safe nonetheless.
        _Ninitial = _Ninitx = _Ninity = _Nk;
        _ktab = shared_ptr<KTable>(new KTable(_Nk, 1.0));
        double *kptr = reinterpret_cast<double*>(_ktab->getArray());
        const double* ptr = data.getData();
        for(int i=0; i<2*_Nk*(_Nk/2+1); i++)
            kptr[i] = ptr[i];
        _flux = kValue(Position<double>(0.,0.)).real();
        setCentroid();
    }

    SBInterpolatedKImage::SBInterpolatedKImageImpl::~SBInterpolatedKImageImpl() {}

    const Interpolant& SBInterpolatedKImage::SBInterpolatedKImageImpl::getKInterp() const
    {
        return _kInterp.get1d();
    }

    std::complex<double> SBInterpolatedKImage::SBInterpolatedKImageImpl::kValue(
        const Position<double>& k) const
    {
        xdbg<<"evaluating kValue("<<k.x<<","<<k.y<<")"<<std::endl;
        xdbg<<"_maxk = "<<_maxk<<std::endl;
        if (std::abs(k.x) > _maxk || std::abs(k.y) > _maxk) return std::complex<double>(0.,0.);
        return _ktab->interpolate(k.x, k.y, _kInterp);
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
        return ConstImageView<double>(data, shared_ptr<double>(), 1, 2*N,
                                      Bounds<int>(0,2*N-1,0,N/2));
    }

    std::string SBInterpolatedKImage::SBInterpolatedKImageImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBInterpolatedKImage(";
        oss << "galsim.ImageD(array([";

        ConstImageView<double> data = getKData();
        const double* ptr = data.getData();
        const int skip = data.getNSkip();
        const int step = data.getStep();
        const int xmin = data.getXMin();
        const int xmax = data.getXMax();
        const int ymin = data.getYMin();
        const int ymax = data.getYMax();
        for (int j=ymin; j<=ymax; j++, ptr+=skip) {
            if (j > ymin) oss <<",";
            oss << "[" << (*ptr == 0. ? 0. : *ptr);
            ptr += step;
            for (int i=xmin+1; i<=xmax; i++, ptr+=step) oss << "," << (*ptr == 0. ? 0. : *ptr);
            oss << "]";
        }

        oss<<"],dtype=float)).image, ";

        oss << stepK() << ", " << maxK() << ", ";
        oss << getKInterp().makeStr()<<", ";
        oss << "galsim._galsim.GSParams("<<gsparams<<"))";

        return oss.str();
    }

    // instantiate template functions for expected image types
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams);

    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<float>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image,
        const Bounds<int>& init_bounds, const Bounds<int>& nonzero_bounds,
        const Interpolant& xInterp, const Interpolant& kInterp,
        double stepk, double maxk, const GSParams& gsparams);

    typedef std::complex<double> cdouble;
    template SBInterpolatedKImage::SBInterpolatedKImage(
        const BaseImage<cdouble>& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams);

    template SBInterpolatedKImage::SBInterpolatedKImageImpl::SBInterpolatedKImageImpl(
        const BaseImage<cdouble>& kimage, double stepk,
        const Interpolant& kInterp, const GSParams& gsparams);

    template double CalculateSizeContainingFlux(const BaseImage<double>& im, double target_flux);
    template double CalculateSizeContainingFlux(const BaseImage<float>& im, double target_flux);

} // namespace galsim
