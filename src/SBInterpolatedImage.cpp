
//#define DEBUGLOGGING

#include <algorithm>
#include "SBInterpolatedImage.h"
#include "SBInterpolatedImageImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    template <typename T> 
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) :
        SBProfile(new SBInterpolatedImageImpl(image,xInterp,kInterp,dx,pad_factor)) {}

    SBInterpolatedImage::SBInterpolatedImage(
        const MultipleImageHelper& multi, const std::vector<double>& weights,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp) :
        SBProfile(new SBInterpolatedImageImpl(multi,weights,xInterp,kInterp)) {}

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs) : SBProfile(rhs) {}

    SBInterpolatedImage::~SBInterpolatedImage() {}

    void SBInterpolatedImage::calculateStepK() const 
    { 
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return dynamic_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateStepK(); 
    }

    void SBInterpolatedImage::calculateMaxK() const {

        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return dynamic_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateMaxK(); 
    }

    template <class T>
    MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<T> > >& images,
        double dx, double pad_factor) :
        _pimpl(new MultipleImageHelperImpl)
    {
        if (images.size() == 0) 
            throw std::runtime_error("No images passed into MultipleImageHelper");

        _pimpl->Ninitial = std::max( images[0]->getYMax()-images[0]->getYMin()+1,
                                     images[0]->getXMax()-images[0]->getXMin()+1 );
        for (size_t i=1; i<images.size(); ++i) {
            int Ni = std::max( images[i]->getYMax()-images[i]->getYMin()+1,
                               images[i]->getXMax()-images[i]->getXMin()+1 );
            if (Ni > _pimpl->Ninitial) _pimpl->Ninitial = Ni;
        }
        _pimpl->Ninitial = _pimpl->Ninitial + _pimpl->Ninitial%2;
        assert(_pimpl->Ninitial%2==0);
        assert(_pimpl->Ninitial>=2);

        if (dx<=0.) {
            _pimpl->dx = images[0]->getScale();
            for (size_t i=1; i<images.size(); ++i) {
                double dxi = images[i]->getScale();
                if (dxi != _pimpl->dx) throw std::runtime_error(
                    "No dx given to MultipleImageHelper, "
                    "and images do not all have the same scale.");
            }
        } else {
            _pimpl->dx = dx;
        }

        if (pad_factor <= 0.) pad_factor = sbp::oversample_x;
        // NB: don't need floor, since rhs is positive, so floor is superfluous.
        _pimpl->Nk = goodFFTSize(int(pad_factor*_pimpl->Ninitial));

        double dx2 = _pimpl->dx*_pimpl->dx;
        double dx3 = _pimpl->dx*dx2;

        // fill data from images, shifting to center the image in the table
        _pimpl->vx.resize(images.size());
        _pimpl->vk.resize(images.size());
        _pimpl->flux.resize(images.size());
        _pimpl->xflux.resize(images.size());
        _pimpl->yflux.resize(images.size());
        for (size_t i=0; i<images.size(); ++i) {
            dbg<<"Image "<<i<<std::endl;
            double sum = 0.;
            double sumx = 0.;
            double sumy = 0.;
            _pimpl->vx[i].reset(new XTable(_pimpl->Nk, _pimpl->dx));
            const BaseImage<T>& img = *images[i];
            int xStart = -((img.getXMax()-img.getXMin()+1)/2);
            int y = -((img.getYMax()-img.getYMin()+1)/2);
            dbg<<"xStart = "<<xStart<<", yStart = "<<y<<std::endl;
            for (int iy = img.getYMin(); iy<= img.getYMax(); iy++, y++) {
                int x = xStart;
                for (int ix = img.getXMin(); ix<= img.getXMax(); ix++, x++) {
                    double value = img(ix,iy);
                    _pimpl->vx[i]->xSet(x, y, value);
                    sum += value;
                    sumx += value*x;
                    sumy += value*y;
                    xxdbg<<"ix,iy,x,y = "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
                    xxdbg<<"value = "<<value<<", sums = "<<sum<<','<<sumx<<','<<sumy<<std::endl;
                }
            }
            _pimpl->flux[i] = sum * dx2;
            _pimpl->xflux[i] = sumx * dx3;
            _pimpl->yflux[i] = sumy * dx3;
            dbg<<"flux = "<<_pimpl->flux[i]<<
                ", xflux = "<<_pimpl->xflux[i]<<", yflux = "<<_pimpl->yflux[i]<<std::endl;
        }
    }

    template <class T>
    MultipleImageHelper::MultipleImageHelper(
        const BaseImage<T>& image, double dx, double pad_factor) :
        _pimpl(new MultipleImageHelperImpl)
    {
        dbg<<"Start MultipleImageHelper constructor for one image\n";
        dbg<<"image bounds = "<<image.getBounds()<<std::endl;
        _pimpl->Ninitial= std::max( image.getYMax()-image.getYMin()+1,
                                    image.getXMax()-image.getXMin()+1 );
        _pimpl->Ninitial = _pimpl->Ninitial + _pimpl->Ninitial%2;
        dbg<<"Ninitial = "<<_pimpl->Ninitial<<std::endl;
        assert(_pimpl->Ninitial%2==0);
        assert(_pimpl->Ninitial>=2);

        if (dx<=0.) _pimpl->dx = image.getScale();
        else _pimpl->dx = dx;
        dbg<<"image scale = "<<image.getScale()<<std::endl;
        dbg<<"dx = "<<_pimpl->dx<<std::endl;

        dbg<<"pad_factor = "<<pad_factor<<std::endl;
        if (pad_factor <= 0.) pad_factor = sbp::oversample_x;
        dbg<<"pad_factor => "<<pad_factor<<std::endl;
        // NB: don't need floor, since rhs is positive, so floor is superfluous.
        _pimpl->Nk = goodFFTSize(int(pad_factor*_pimpl->Ninitial));
        dbg<<"Nk = "<<_pimpl->Nk<<std::endl;

        double dx2 = _pimpl->dx*_pimpl->dx;
        double dx3 = _pimpl->dx*dx2;

        // fill data from images, shifting to center the image in the table
        _pimpl->vx.resize(1);
        _pimpl->vk.resize(1);
        _pimpl->flux.resize(1);
        _pimpl->xflux.resize(1);
        _pimpl->yflux.resize(1);
        double sum = 0.;
        double sumx = 0.;
        double sumy = 0.;
        _pimpl->vx[0].reset(new XTable(_pimpl->Nk, _pimpl->dx));
        int xStart = -((image.getXMax()-image.getXMin()+1)/2);
        int y = -((image.getYMax()-image.getYMin()+1)/2);
        dbg<<"xStart = "<<xStart<<", yStart = "<<y<<std::endl;
        for (int iy = image.getYMin(); iy<= image.getYMax(); iy++, y++) {
            int x = xStart;
            for (int ix = image.getXMin(); ix<= image.getXMax(); ix++, x++) {
                double value = image(ix,iy);
                _pimpl->vx[0]->xSet(x, y, value);
                sum += value;
                sumx += value*x;
                sumy += value*y;
                xxdbg<<"ix,iy,x,y = "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
                xxdbg<<"value = "<<value<<", sums = "<<sum<<','<<sumx<<','<<sumy<<std::endl;
            }
        }
        _pimpl->flux[0] = sum * dx2;
        _pimpl->xflux[0] = sumx * dx3;
        _pimpl->yflux[0] = sumy * dx3;
        dbg<<"flux = "<<_pimpl->flux[0]<<
            ", xflux = "<<_pimpl->xflux[0]<<", yflux = "<<_pimpl->yflux[0]<<std::endl;
        dbg<<"vx[0].size = "<<_pimpl->vx[0]->getN()<<", scale = "<<_pimpl->vx[0]->getDx()<<std::endl;
    }

    boost::shared_ptr<KTable> MultipleImageHelper::getKTable(int i) const 
    {
        if (!_pimpl->vk[i].get()) _pimpl->vk[i] = _pimpl->vx[i]->transform();
        return _pimpl->vk[i];
    }

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<T>& image, 
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor) : 
        _multi(image,dx,pad_factor), _wts(1,1.), _xInterp(xInterp), _kInterp(kInterp),
        _readyToShoot(false)
    { initialize(); }

    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const MultipleImageHelper& multi, const std::vector<double>& weights,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp) :
        _multi(multi), _wts(weights), _xInterp(xInterp), _kInterp(kInterp), _readyToShoot(false) 
    {
        assert(weights.size() == multi.size());
        initialize(); 
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::initialize()
    {
        dbg<<"SBInterpolate initialize\n";
        if (!_xInterp.get()) _xInterp = sbp::defaultXInterpolant2d;
        if (!_kInterp.get()) _kInterp = sbp::defaultKInterpolant2d;

        dbg<<"N = "<<_multi.getNin()<<", xrange = "<<_xInterp->xrange();
        dbg<<", scale = "<<_multi.getScale()<<std::endl;

        if (_multi.size() == 1 && _wts[0] == 1.) {
            _xtab = _multi.getXTable(0);
        } else {
            _xtab.reset(new XTable(*_multi.getXTable(0)));
            *_xtab *= _wts[0];
            for (size_t i=1; i<_multi.size(); i++)
                _xtab->accumulate(*_multi.getXTable(i), _wts[i]);
        }
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;

        // Calculate stepK:
        // 
        // The amount of flux missed in a circle of radius pi/stepk should miss at 
        // most alias_threshold of the flux.
        //
        // We add the size of the image and the size of the interpolant in quadrature.
        // (Note: Since this isn't a radial profile, R isn't really a radius, but rather 
        //        the size of the square box that is enclosing all the flux.)
        double R = _multi.getNin()/2. * _multi.getScale();
        // Add xInterp range in quadrature just like convolution:
        double R2 = _xInterp->xrange() * _multi.getScale();
        dbg<<"R(image) = "<<R<<", R(interpolant) = "<<R2<<std::endl;
        R = sqrt(R*R + R2*R2);
        dbg<<"=> R = "<<R<<std::endl;
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;

        // Calculate maxk:
        //
        // Set maxK to the value where the FT is down to maxk_threshold
        //
        // Notice that interpolant other than sinc may make max frequency higher than
        // the Nyquist frequency of the initial image.
        //
        // Also, since we used kvalue_accuracy for the threshold of _xInterp
        // (at least for the default quintic interpolant) rather than maxk_threshold,
        // this will probably be larger than we really need.
        // We could modify the urange method of Interpolant to take a threshold value
        // at that point, rather than just use the constructor's value, but it's 
        // probably not worth it.  It will probably be very rare that the final maxK
        // value of the FFT will be due to an SBInterpolatedImage.  Usually, this will
        // be convolved by a PSF that will have a smaller maxK.
        _maxk = _xInterp->urange() * 2.*M_PI / _multi.getScale(); 
        dbg<<"maxk = "<<_maxk<<std::endl;

        dbg<<"flux = "<<getFlux()<<std::endl;
    }

    SBInterpolatedImage::SBInterpolatedImageImpl::~SBInterpolatedImageImpl() {}

    double SBInterpolatedImage::SBInterpolatedImageImpl::getFlux() const 
    {
        double flux = 0.;
        for (size_t i=0; i<_multi.size(); ++i) flux += _wts[i] * _multi.getFlux(i);
        dbg<<"flux = "<<flux<<std::endl;
        return flux;
    }

    Position<double> SBInterpolatedImage::SBInterpolatedImageImpl::centroid() const 
    {
        double x = 0., y=0.;
        for (size_t i=0; i<_multi.size(); ++i) {
            x += _wts[i] * _multi.getXFlux(i);
            y += _wts[i] * _multi.getYFlux(i);
        }
        double flux = getFlux();
        x /= flux;  y /= flux;
        return Position<double>(x,y);
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkK() const 
    {
        // Conduct FFT
        if (_ktab.get()) return;
        if (_multi.size() == 1 && _wts[0] == 1.) {
            _ktab = _multi.getKTable(0);
        } else {
            _ktab.reset(new KTable(*_multi.getKTable(0)));
            *_ktab *= _wts[0];
            for (size_t i=1; i<_multi.size(); i++)
                _ktab->accumulate(*_multi.getKTable(i), _wts[i]);
        }
        dbg<<"Built ktab\n";
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKGrid(KTable& kt) const 
    {
        // This override of base class is to permit potential efficiency gain from
        // separable interpolant kernel.  If so, the KTable interpolation routine
        // will go faster if we make y iteration the inner loop.
        if (dynamic_cast<const InterpolantXY*> (_kInterp.get())) {
            int N = kt.getN();
            double dk = kt.getDk();
            // Only need ix>=0 because it's Hermitian:
            for (int ix = 0; ix <= N/2; ix++) {
                for (int iy = -N/2; iy < N/2; iy++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValue(k));
                }
            }
        } else {
            // Otherwise just use the normal routine to fill the grid:
            SBProfileImpl::fillKGrid(kt);
        }
    }

    // Same deal: reverse axis order if we have separable interpolant in X domain
    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXGrid(XTable& xt) const 
    {
        if ( dynamic_cast<const InterpolantXY*> (_xInterp.get())) {
            int N = xt.getN();
            double dx = xt.getDx();
            for (int ix = -N/2; ix < N/2; ix++) {
                for (int iy = -N/2; iy < N/2; iy++) {
                    Position<double> x(ix*dx,iy*dx);
                    xt.xSet(ix,iy,xValue(x));
                }
            }
        } else {
            // Otherwise just use the normal routine to fill the grid:
            SBProfileImpl::fillXGrid(xt);
        }
    }

    // One more time: for images now
    // Returns total flux
    template <typename T>
    double SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<T>& I, double dx, double gain) const 
    {
        if ( dynamic_cast<const InterpolantXY*> (_xInterp.get())) {
            double sum=0.;
            for (int ix = I.getXMin(); ix <= I.getXMax(); ix++) {
                for (int iy = I.getYMin(); iy <= I.getYMax(); iy++) {
                    Position<double> x(ix*dx,iy*dx);
                    T val = gain * xValue(x);
                    sum += val;
                    I(ix,iy) += val;
                }
            }
            I.setScale(dx);
            return sum;
        } else {
            // Otherwise just use the normal routine to fill the grid:
            // Note that we need to call doFillXImage, not fillXImage here,
            // to avoid the virtual function resolution.
            return SBProfileImpl::doFillXImage(I,dx,gain);
        }
    }

    double SBInterpolatedImage::SBInterpolatedImageImpl::xValue(const Position<double>& p) const 
    { return _xtab->interpolate(p.x, p.y, *_xInterp); }

    std::complex<double> SBInterpolatedImage::SBInterpolatedImageImpl::kValue(
        const Position<double>& p) const 
    {
        const double TWOPI = 2.*M_PI;

        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*_multi.getScale()/TWOPI;
        if (std::abs(ux) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*_multi.getScale()/TWOPI;
        if (std::abs(uy) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = _xInterp->uval(ux, uy);

        checkK();
        return xKernelTransform * _ktab->interpolate(p.x, p.y, *_kInterp);
    }

    // We provide an option to update the stepk value by directly calculating what
    // size region around the center encloses (1-alias_threshold) of the total flux.
    // This can be useful if you make the image bigger than you need to, just to be
    // safe, but then want to use as large a stepk value as possible.
    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateStepK() const
    {
        dbg<<"Start SBInterpolatedImage calculateStepK()\n";
        dbg<<"Current value of stepk = "<<_stepk<<std::endl;
        dbg<<"Find box that encloses "<<1.-sbp::alias_threshold<<" of the flux.\n";
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;
        int N = _xtab->getN();
        double dx = _xtab->getDx();
        double dx2 = dx*dx;
        Position<double> cen = centroid();
        dbg<<"centroid = "<<cen<<std::endl;
        int ixCen = int(floor(cen.x / dx + 0.5));
        int iyCen = int(floor(cen.y / dx + 0.5));
        dbg<<"center ix,iy = "<<ixCen<<','<<iyCen<<std::endl;
        double fluxTot = getFlux()/dx2;
        dbg<<"fluxTot = "<<fluxTot<<std::endl;
        double flux = (*_xtab).xval(ixCen,iyCen);
        double thresh = (1.-sbp::alias_threshold) * fluxTot;
        dbg<<"thresh = "<<thresh<<std::endl;

        // d1 = 0 means that we haven't yet found the d that enclosed enough flux.
        // When we find a flux > thresh, we set d1 = d.
        // However, since the function can have negative regions, we need to keep 
        // going to make sure an oscillation doesn't bring us back below thresh.
        // When this happens, we set d1 to 0 again and look for a larger value that 
        // enclosed enough flux again.
        int d1 = 0; 
        for (int d=1; (ixCen-d >= -N/2 || ixCen+d < N/2 ||
                       iyCen-d >= -N/2 || iyCen+d < N/2); ++d) {
            dbg<<"d = "<<d<<std::endl;
            dbg<<"d1 = "<<d1<<std::endl;
            dbg<<"flux = "<<flux<<std::endl;
            // Add the left side of box:
            int ix = ixCen - d;
            if (ix >= -N/2) {
                for(int iy = std::max(iyCen-d,-N/2); iy <= std::min(iyCen+d,N/2-1); ++iy) {
                    xxdbg<<"Add xval("<<ix<<','<<iy<<") = "<<_xtab->xval(ix,iy)<<std::endl;
                    flux += _xtab->xval(ix,iy);
                }
            }
            // Add the right size of box:
            ix = ixCen + d;
            if (ix < N/2) {
                for(int iy = std::max(iyCen-d,-N/2); iy <= std::min(iyCen+d,N/2-1); ++iy) {
                    xxdbg<<"Add xval("<<ix<<','<<iy<<") = "<<_xtab->xval(ix,iy)<<std::endl;
                    flux += _xtab->xval(ix,iy);
                }
            }
            // Add the bottom side of box:
            int iy = iyCen - d;
            if (iy >= -N/2) {
                for(int ix = std::max(ixCen-d+1,-N/2); ix <= std::min(ixCen+d-1,N/2-1); ++ix) {
                    xxdbg<<"Add xval("<<ix<<','<<iy<<") = "<<_xtab->xval(ix,iy)<<std::endl;
                    flux += _xtab->xval(ix,iy);
                }
            }
            // Add the top side of box:
            iy = iyCen + d;
            if (iy < N/2) {
                for(int ix = std::max(ixCen-d+1,-N/2); ix <= std::min(ixCen+d-1,N/2-1); ++ix) {
                    xxdbg<<"Add xval("<<ix<<','<<iy<<") = "<<_xtab->xval(ix,iy)<<std::endl;
                    flux += _xtab->xval(ix,iy);
                }
            }
            if (flux < thresh) {
                d1 = 0; // Mark that we haven't gotten to a good enclosing radius yet.
            } else {
                if (d1 == 0) d1 = d; // Mark this radius as a good one.
            }
        }
        dbg<<"Done: flux = "<<flux<<std::endl;
        // Should have added up to the total flux.
        assert( std::abs(flux - fluxTot) < 1.e-6 );
        if (d1 == 0) {
            dbg<<"No smaller radius found.  Keep current value of stepk\n";
            return;
        }
        // (Note: Since this isn't a radial profile, R isn't really a radius, but rather 
        //        the size of the square box that is enclosing (1-alias_thresh) of the flux.)
        double R = (d1+0.5) * dx;
        dbg<<"d = "<<d1<<" => R = "<<R<<std::endl;
        // Add xInterp range in quadrature just like convolution:
        double R2 = _xInterp->xrange() * _multi.getScale();
        dbg<<"R(image) = "<<R<<", R(interpolant) = "<<R2<<std::endl;
        R = sqrt(R*R + R2*R2);
        dbg<<"=> R = "<<R<<std::endl;
        _stepk = M_PI / R;
        dbg<<"stepk = "<<_stepk<<std::endl;
    }

    // Helper struct to compare two pairs by their first value.
    struct PairSorter 
    { 
        bool operator()(const std::pair<double,double>& p1,
                        const std::pair<double,double>& p2) const
        { return p1.first < p2.first; }
    };

    // Helper struct to find a pair whose second value is > thresh.
    struct AboveThresh
    {
        AboveThresh(double thresh) : _thresh(thresh) {}
        bool operator()(const std::pair<double,double>& p) const
        { return p.second > _thresh; }
        double _thresh;
    };
    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateMaxK() const
    {
        dbg<<"Start SBInterpolatedImage calculateMaxK()\n";
        dbg<<"Current value of maxk = "<<_maxk<<std::endl;
        dbg<<"Find the smallest k such that all values outside of this are less than "
            <<sbp::maxk_threshold<<std::endl;
        checkK();
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
        
        int N = _ktab->getN();
        double dk = _ktab->getDk();
        double dk2 = dk*dk;

        // Among the elements with kval > thresh, find the one with the maximum ksq
        double thresh = sbp::maxk_threshold * getFlux();
        thresh *= thresh; // Since values will be |kval|^2.
        double maxk_ksq = 0.;
        double maxk_norm_kval = 0.;
        for(int ix=-N/2, j=0; ix<N/2; ++ix) for(int iy=-N/2; iy<N/2; ++iy, ++j) {
            double norm_kval = std::norm(_ktab->kval(ix,iy));
            if (norm_kval > thresh) {
                double ksq = (ix*ix + iy*iy) * dk2;
                if (ksq  > maxk_ksq) {
                    maxk_ksq = ksq;
                    maxk_norm_kval = norm_kval;
                }
            }
        }
        dbg<<"Found |kval|^2 = "<<maxk_norm_kval<<" at ksq = "<<maxk_ksq<<std::endl;

        // Check if we want to use the new value.  (Only if smaller than the current value.)
        double new_maxk = sqrt(maxk_ksq);
        dbg<<"new_maxk = "<<new_maxk<<std::endl;
        if (new_maxk < _maxk) {
            dbg<<"New value is smaller, so update\n";
            _maxk = new_maxk;
        } else {
            dbg<<"New value is not smaller, so keep the current value.\n";
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkReadyToShoot() const 
    {
        if (_readyToShoot) return;

        dbg<<"SBInterpolatedImage not ready to shoot.  Build _pt:\n";

        // Build the sets holding cumulative fluxes of all Pixels
        _positiveFlux = 0.;
        _negativeFlux = 0.;
        _pt.clear();
        for (int iy=-_multi.getNin()/2; iy<_multi.getNin()/2; iy++) {
            double y = iy*_multi.getScale();
            for (int ix=-_multi.getNin()/2; ix<_multi.getNin()/2; ix++) {
                double flux = _xtab->xval(ix,iy) * _multi.getScale()*_multi.getScale();
                if (flux==0.) continue;
                double x=ix*_multi.getScale();
                if (flux > 0.) {
                    _positiveFlux += flux;
                } else {
                    _negativeFlux += -flux;
                }
                _pt.push_back(Pixel(x,y,flux));
            }
        }
        _pt.buildTree();
        dbg<<"Built tree\n";

        // The above just computes the positive and negative flux for the main image.
        // This is convolved by the interpolant, so we need to correct these values
        // in the same way that SBConvolve does:
        double p1 = _positiveFlux;
        double n1 = _negativeFlux;
        dbg<<"positiveFlux = "<<p1<<", negativeFlux = "<<n1<<std::endl;
        double p2 = _xInterp->getPositiveFlux();
        double n2 = _xInterp->getNegativeFlux();
        dbg<<"Interpolant has positiveFlux = "<<p2<<", negativeFlux = "<<n2<<std::endl;
        _positiveFlux = p1*p2 + n1*n2;
        _negativeFlux = p1*n2 + n1*p2;
        dbg<<"positiveFlux => "<<_positiveFlux<<", negativeFlux => "<<_negativeFlux<<std::endl;

        _readyToShoot = true;
    }

    // Photon-shooting 
    boost::shared_ptr<PhotonArray> SBInterpolatedImage::SBInterpolatedImageImpl::shoot(
        int N, UniformDeviate ud) const
    {
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

        boost::shared_ptr<PhotonArray> result(new PhotonArray(N));
        if (N<=0 || _pt.empty()) return result;
        double totalAbsFlux = _positiveFlux + _negativeFlux;
        double fluxPerPhoton = totalAbsFlux / N;
        dbg<<"posFlux = "<<_positiveFlux<<", negFlux = "<<_negativeFlux<<std::endl;
        dbg<<"totFlux = "<<_positiveFlux-_negativeFlux<<", totAbsFlux = "<<totalAbsFlux<<std::endl;
        dbg<<"fluxPerPhoton = "<<fluxPerPhoton<<std::endl;
        for (int i=0; i<N; i++) {
            double unitRandom = ud();
            const Pixel* p = _pt.find(unitRandom);
            result->setPhoton(i, p->x, p->y, 
                              p->isPositive ? fluxPerPhoton : -fluxPerPhoton);
        }
        dbg<<"result->getTotalFlux = "<<result->getTotalFlux()<<std::endl;

        // Last step is to convolve with the interpolation kernel. 
        // Can skip if using a 2d delta function
        const InterpolantXY* xyPtr = dynamic_cast<const InterpolantXY*> (_xInterp.get());
        if ( !(xyPtr && dynamic_cast<const Delta*> (xyPtr->get1d()))) {
            boost::shared_ptr<PhotonArray> pa_interp = _xInterp->shoot(N, ud);
            pa_interp->scaleXY(_xtab->getDx());
            result->convolve(*pa_interp, ud);
        }

        dbg<<"InterpolatedImage Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    // instantiate template functions for expected image types
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<float> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<double> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<int> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<short> > >& images,
        double dx, double pad_factor);

    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<float>& image, double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<double>& image, double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<int>& image, double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<short>& image, double dx, double pad_factor);

    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor);

    template double SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<float>& I, double dx, double gain) const;
    template double SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<double>& I, double dx, double gain) const;
}

