
#include <algorithm>

//#define DEBUGLOGGING

#include "SBInterpolatedImage.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif


namespace galsim {

    const double TWOPI = 2.*M_PI;

    template <class T>
    MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<T> > >& images,
        double dx, double padFactor) :
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

        if (padFactor <= 0.) padFactor = sbp::oversample_x;
        _pimpl->Nk = goodFFTSize(int(std::floor(padFactor*_pimpl->Ninitial)));

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
                    xdbg<<"ix,iy,x,y = "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
                    xdbg<<"value = "<<value<<", sums = "<<sum<<','<<sumx<<','<<sumy<<std::endl;
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
        const BaseImage<T>& image, double dx, double padFactor) :
        _pimpl(new MultipleImageHelperImpl)
    {
        _pimpl->Ninitial= std::max( image.getYMax()-image.getYMin()+1,
                                    image.getXMax()-image.getXMin()+1 );
        _pimpl->Ninitial = _pimpl->Ninitial + _pimpl->Ninitial%2;
        assert(_pimpl->Ninitial%2==0);
        assert(_pimpl->Ninitial>=2);

        if (dx<=0.) _pimpl->dx = image.getScale();
        else _pimpl->dx = dx;

        if (padFactor <= 0.) padFactor = sbp::oversample_x;
        _pimpl->Nk = goodFFTSize(int(std::floor(padFactor*_pimpl->Ninitial)));

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
                xdbg<<"ix,iy,x,y = "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
                xdbg<<"value = "<<value<<", sums = "<<sum<<','<<sumx<<','<<sumy<<std::endl;
            }
        }
        _pimpl->flux[0] = sum * dx2;
        _pimpl->xflux[0] = sumx * dx3;
        _pimpl->yflux[0] = sumy * dx3;
        dbg<<"flux = "<<_pimpl->flux[0]<<
            ", xflux = "<<_pimpl->xflux[0]<<", yflux = "<<_pimpl->yflux[0]<<std::endl;
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
        double dx, double padFactor) : 
        _multi(image,dx,padFactor), _wts(1,1.), _xInterp(xInterp), _kInterp(kInterp),
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
        if (!_xInterp.get()) _xInterp = sbp::defaultXInterpolant2d;
        if (!_kInterp.get()) _kInterp = sbp::defaultKInterpolant2d;

        _max_size = (_multi.getNin()+2.*_xInterp->xrange())*_multi.getScale();

        if (_multi.size() == 1 && _wts[0] == 1.) {
            _xtab = _multi.getXTable(0);
        } else {
            _xtab.reset(new XTable(*_multi.getXTable(0)));
            *_xtab *= _wts[0];
            for (size_t i=1; i<_multi.size(); i++)
                _xtab->accumulate(*_multi.getXTable(i), _wts[i]);
        }
    }

    SBInterpolatedImage::SBInterpolatedImageImpl::~SBInterpolatedImageImpl() {}

    double SBInterpolatedImage::SBInterpolatedImageImpl::getFlux() const 
    {
        double flux = 0.;
        for (size_t i=0; i<_multi.size(); ++i) flux += _wts[i] * _multi.getFlux(i);
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
        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*_multi.getScale()/TWOPI;
        if (std::abs(ux) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*_multi.getScale()/TWOPI;
        if (std::abs(uy) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = _xInterp->uval(ux, uy);

        checkK();
        return xKernelTransform * _ktab->interpolate(p.x, p.y, *_kInterp);
    }

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBInterpolatedImage::SBInterpolatedImageImpl::maxK() const 
    {
        // Notice that interpolant other than sinc may make max frequency higher than
        // the Nyquist frequency of the initial image
        
        // Also, since we used kvalue_accuracy for the threshold of _xInterp
        // (at least for the default quintic interpolant) rather than maxk_threshold,
        // this will probably be larger than we really need.
        // We could modify the urange method of Interpolant to take a threshold value
        // at that point, rather than just use the constructor's value, but it's 
        // probably not worth it.  It will probably be very rare that the final maxK
        // value of the FFT will be due to an SBInterpolatedImage.  Usually, this will
        // be convolved by a PSF that will have a smaller maxK.
        return _xInterp->urange() * 2.*M_PI / _multi.getScale(); 
    }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBInterpolatedImage::SBInterpolatedImageImpl::stepK() const 
    {
        // In this case, R = original image extent + kernel footprint, which we
        // have already stored as _max_size.
        return M_PI / _max_size;
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::checkReadyToShoot() const 
    {
        if (_readyToShoot) return;

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
            Pixel* p = _pt.find(unitRandom);
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
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<float> > >& images,
        double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<double> > >& images,
        double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<int> > >& images,
        double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<short> > >& images,
        double dx, double padFactor);

    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<float>& image, double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<double>& image, double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<int>& image, double dx, double padFactor);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<short>& image, double dx, double padFactor);

    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<int>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<short>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double padFactor);

    template double SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<float>& I, double dx, double gain) const;
    template double SBInterpolatedImage::SBInterpolatedImageImpl::fillXImage(
        ImageView<double>& I, double dx, double gain) const;
}

