
#include <algorithm>

#include "SBInterpolatedImage.h"

namespace galsim {

    const double TWOPI = 2.*M_PI;

    // Default k-space interpolant is quintic:
    Quintic defaultKInterpolant1d(sbp::kvalue_accuracy);

    InterpolantXY SBInterpolatedImage::defaultKInterpolant2d(defaultKInterpolant1d);

    SBInterpolatedImage::SBInterpolatedImage(
        int Npix, double dx, const Interpolant2d& i, int Nimages) :  
        
        _Ninitial(Npix+Npix%2), _dx(dx), _Nimages(Nimages),
        _xInterp(&i), _kInterp(&defaultKInterpolant2d),
        _wts(_Nimages, 1.), _fluxes(_Nimages, 1.), 
        _xFluxes(_Nimages, 0.), _yFluxes(_Nimages,0.),
        _xsum(0), _ksum(0), _xsumValid(false), _ksumValid(false),
        _ready(false), _readyToShoot(false) 
    {
        assert(_Ninitial%2==0);
        assert(_Ninitial>=2);
        // Choose the padded size for input array - size 2^N or 3*2^N
        // Make FFT either 2^n or 3x2^n
        _Nk = goodFFTSize(sbp::oversample_x*_Ninitial);
        _dk = TWOPI / (_Nk*_dx);

        // allocate xTables
        for (int i=0; i<_Nimages; i++) 
            _vx.push_back(new XTable(_Nk, _dx));
        _max_size = (_Ninitial+2.*_xInterp->xrange())*_dx;
    }

    template <typename T>
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& img, const Interpolant2d& i, double dx, double padFactor) : 

        _dx(dx), _Nimages(1),
        _xInterp(&i), _kInterp(&defaultKInterpolant2d),
        _wts(_Nimages, 1.), _fluxes(_Nimages, 1.), 
        _xFluxes(_Nimages, 0.), _yFluxes(_Nimages,0.),
        _xsum(0), _ksum(0), _xsumValid(false), _ksumValid(false),
        _ready(false), _readyToShoot(false) 
    {
        _Ninitial = std::max( img.getYMax()-img.getYMin()+1, img.getXMax()-img.getXMin()+1);
        _Ninitial = _Ninitial + _Ninitial%2;
        assert(_Ninitial%2==0);
        assert(_Ninitial>=2);
        if (_dx<=0.) {
            _dx = img.getScale();
        }
        if (padFactor <= 0.) padFactor = sbp::oversample_x;
        // Choose the padded size for input array - size 2^N or 3*2^N
        // Make FFT either 2^n or 3x2^n
        _Nk = goodFFTSize(static_cast<int> (std::floor(padFactor*_Ninitial)));
        _dk = TWOPI / (_Nk*_dx);

        // allocate xTables
        for (int i=0; i<_Nimages; i++) 
            _vx.push_back(new XTable(_Nk, _dx));
        // fill data from image, shifting to center the image in the table
        int xStart = -((img.getXMax()-img.getXMin()+1)/2);
        int yTab = -((img.getYMax()-img.getYMin()+1)/2);
        for (int iy = img.getYMin(); iy<= img.getYMax(); iy++, yTab++) {
            int xTab = xStart;
            for (int ix = img.getXMin(); ix<= img.getXMax(); ix++, xTab++) 
                _vx.front()->xSet(xTab, yTab, img(ix,iy));
        }
        _max_size = (_Ninitial+2*_xInterp->xrange())*_dx;
    }

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs):
        _Ninitial(rhs._Ninitial), _dx(rhs._dx), _Nk(rhs._Nk), _Nimages(rhs._Nimages),
        _xInterp(rhs._xInterp), _kInterp(rhs._kInterp),
        _wts(rhs._wts), _fluxes(rhs._fluxes), _xFluxes(rhs._xFluxes), _yFluxes(rhs._yFluxes),
        _xsum(0), _ksum(0), _xsumValid(false), _ksumValid(false), 
        _ready(rhs._ready), _readyToShoot(false), _max_size(rhs._max_size)
    {
        // copy tables
        for (int i=0; i<_Nimages; i++) {
            _vx.push_back(new XTable(*rhs._vx[i]));
            if (_ready) _vk.push_back(new KTable(*rhs._vk[i]));
        }
    }

    SBInterpolatedImage::~SBInterpolatedImage() 
    {
        for (size_t i=0; i<_vx.size(); i++) if (_vx[i]) { delete _vx[i]; _vx[i]=0; }
        for (size_t i=0; i<_vk.size(); i++) if (_vk[i]) { delete _vk[i]; _vk[i]=0; }
        if (_xsum) { delete _xsum; _xsum=0; }
        if (_ksum) { delete _ksum; _ksum=0; }
    }

    double SBInterpolatedImage::getFlux() const 
    {
        checkReady();
        return _wts * _fluxes;
    }

    void SBInterpolatedImage::setFlux(double flux) 
    {
        checkReady();
        double factor = flux/getFlux();
        _wts *= factor;
        if (_xsumValid) *_xsum *= factor;
        if (_ksumValid) *_ksum *= factor;
        _readyToShoot = false;   // Need to rescale all the cumulative fluxes
    }

    Position<double> SBInterpolatedImage::centroid() const 
    {
        checkReady();
        double wtsfluxes = _wts * _fluxes;
        return Position<double>((_wts * _xFluxes) / wtsfluxes, (_wts * _yFluxes) / wtsfluxes);
    }

    void SBInterpolatedImage::setPixel(double value, int ix, int iy, int iz) 
    {
        if (iz < 0 || iz>=_Nimages)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel image number " << iz << " out of bounds";
        if (ix < -_Ninitial/2 || ix >= _Ninitial / 2)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel x coordinate " << ix << " out of bounds";
        if (iy < -_Ninitial/2 || iy >= _Ninitial / 2)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::setPixel x coordinate " << iy << " out of bounds";

        _ready = false;
        _readyToShoot = false;
        _vx[iz]->xSet(ix, iy, value);
    }

    double SBInterpolatedImage::getPixel(int ix, int iy, int iz) const 
    {
        if (iz < 0 || iz>=_Nimages)
            FormatAndThrow<SBError>() << 
                "SBInterpolatedImage::getPixel image number " << iz << " out of bounds";

        return _vx[iz]->xval(ix, iy);
    }

    void SBInterpolatedImage::setWeights(const tmv::Vector<double>& wts) 
    {
        assert(_wts.size()==_Nimages);
        _wts = wts;
        _xsumValid = false;
        _ksumValid = false;
        _readyToShoot = false;
    }

    void SBInterpolatedImage::checkReady() const 
    {
        if (_ready) return;
        // Flush old kTables if any;
        for (size_t i=0; i<_vk.size(); i++) { delete _vk[i]; _vk[i]=0; }
        _vk.clear();

        for (int i=0; i<_Nimages; i++) {
            // Get sums:
            double sum = 0.;
            double sumx = 0.;
            double sumy = 0.;
            for (int iy=-_Ninitial/2; iy<_Ninitial/2; iy++) {
                for (int ix=-_Ninitial/2; ix<_Ninitial/2; ix++) {
                    double value = _vx[i]->xval(ix, iy);
                    sum += value;
                    sumx += value*ix;
                    sumy += value*iy;
                }
            }
            _fluxes[i] = sum*_dx*_dx;
            _xFluxes[i] = sumx * std::pow(_dx, 3.);
            _yFluxes[i] = sumy * std::pow(_dx, 3.);

            // Conduct FFT
            _vk.push_back( _vx[i]->transform());
        }
        _ready = true;
        _xsumValid = false;
        _ksumValid = false;
        assert(int(_vk.size())==_Nimages);
    }

    void SBInterpolatedImage::checkXsum() const 
    {
        checkReady();
        if (_xsumValid) return;
        if (!_xsum) {
            _xsum = new XTable(*_vx[0]);
            *_xsum *= _wts[0];
        } else {
            _xsum->clear();
            _xsum->accumulate(*_vx[0], _wts[0]);
        }
        for (int i=1; i<_Nimages; i++)
            _xsum->accumulate(*_vx[i], _wts[i]);
        _xsumValid = true;
    }

    void SBInterpolatedImage::checkKsum() const 
    {
        checkReady();
        if (_ksumValid) return;
        if (!_ksum) {
            _ksum = new KTable(*_vk[0]);
            *_ksum *= _wts[0];
        } else {
            _ksum->clear();
            _ksum->accumulate(*_vk[0], _wts[0]);
        }
        for (int i=1; i<_Nimages; i++)
            _ksum->accumulate(*_vk[i], _wts[i]);
        _ksumValid = true;
    }

    void SBInterpolatedImage::fillKGrid(KTable& kt) const 
    {
        // This override of base class is to permit potential efficiency gain from
        // separable interpolant kernel.  If so, the KTable interpolation routine
        // will go faster if we make y iteration the inner loop.
        if (dynamic_cast<const InterpolantXY*> (_kInterp)) {
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
            SBProfile::fillKGrid(kt);
        }
    }

    // Same deal: reverse axis order if we have separable interpolant in X domain
    void SBInterpolatedImage::fillXGrid(XTable& xt) const 
    {
#ifdef DANIELS_TRACING
        cout << "SBInterpolatedImage::fillXGrid called" << endl;
#endif
        if ( dynamic_cast<const InterpolantXY*> (_xInterp)) {
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
            SBProfile::fillXGrid(xt);
        }
    }

    // One more time: for images now
    // Returns total flux
    template <typename T>
    double SBInterpolatedImage::fillXImage(ImageView<T>& I, double dx) const 
    {
        if ( dynamic_cast<const InterpolantXY*> (_xInterp)) {
            double sum=0.;
            for (int ix = I.getXMin(); ix <= I.getXMax(); ix++) {
                for (int iy = I.getYMin(); iy <= I.getYMax(); iy++) {
                    Position<double> x(ix*dx,iy*dx);
                    T val = xValue(x);
                    sum += val;
                    I(ix,iy) = val;
                }
            }
            I.setScale(dx);
            return sum;
        } else {
            // Otherwise just use the normal routine to fill the grid:
            // Note that we need to call doFillXImage, not fillXImage here,
            // to avoid the virtual function resolution.
            return SBProfile::doFillXImage(I,dx);
        }
    }

#ifndef OLD_WAY
    double SBInterpolatedImage::xValue(const Position<double>& p) const 
    {
        checkXsum();
        return _xsum->interpolate(p.x, p.y, *_xInterp);
    }

    std::complex<double> SBInterpolatedImage::kValue(const Position<double>& p) const 
    {
        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*_dx/TWOPI;
        if (std::abs(ux) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*_dx/TWOPI;
        if (std::abs(uy) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = _xInterp->uval(ux, uy);

        checkKsum();
        return xKernelTransform * _ksum->interpolate(p.x, p.y, *_kInterp);
    }

#else
    double SBInterpolatedImage::xValue(const Position<double>& p) const 
    {
        // Interpolate WITHOUT wrapping the image.
        int ixMin = static_cast<int> ( std::ceil(p.x/_dx - _xInterp->xrange()));
        ixMin = std::max(ixMin, -_Ninitial/2);
        int ixMax = static_cast<int> ( std::floor(p.x/_dx + _xInterp->xrange()));
        ixMax = std::min(ixMax, _Ninitial/2-1);
        int iyMin = static_cast<int> ( std::ceil(p.y/_dx - _xInterp->xrange()));
        iyMin = std::max(iyMin, -_Ninitial/2);
        int iyMax = static_cast<int> ( std::floor(p.y/_dx + _xInterp->xrange()));
        iyMax = std::min(iyMax, _Ninitial/2-1);

        if (ixMax < ixMin || iyMax < iyMin) return 0.;  // kernel does not overlap data
        int npts = (ixMax - ixMin+1)*(iyMax-iyMin+1);
        tmv::Vector<double> kernel(npts,0.);
        tmv::Matrix<double> data(_Nimages, npts, 0.);
        int ipt = 0;
        for (int iy = iyMin; iy <= iyMax; iy++) {
            double deltaY = p.y/_dx - iy;
            for (int ix = ixMin; ix <= ixMax; ix++, ipt++) {
                double deltaX = p.x/_dx - ix;
                kernel[ipt] = _xInterp->xval(deltaX, deltaY);
                for (int iz=0; iz<_Nimages; iz++) {
                    data(iz, ipt) = _vx[iz]->xval(ix, iy);
                }
            }
        }
        return _wts * data * kernel;
    }

    std::complex<double> SBInterpolatedImage::kValue(const Position<double>& p) const 
    {
        checkReady();
        // Interpolate in k space, first apply kInterp kernel to wrapped
        // k-space data, then multiply by FT of xInterp kernel.

        // Don't bother if the desired k value is cut off by the x interpolant:
        double ux = p.x*_dx/TWOPI;
        if (std::abs(ux) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double uy = p.y*_dx/TWOPI;
        if (std::abs(uy) > _xInterp->urange()) return std::complex<double>(0.,0.);
        double xKernelTransform = _xInterp->uval(ux, uy);

        // Range of k points within kernel
        int ixMin = static_cast<int> (std::ceil(p.x/_dk - _kInterp->xrange()));
        int ixMax = static_cast<int> (std::floor(p.x/_dk + _kInterp->xrange()));
        int iyMin = static_cast<int> (std::ceil(p.y/_dk - _kInterp->xrange()));
        int iyMax = static_cast<int> (std::floor(p.y/_dk + _kInterp->xrange()));

        int ixLast = std::min(ixMax, ixMin+_Nk-1);
        int iyLast = std::min(iyMax, iyMin+_Nk-1);
        int npts = (ixLast-ixMin+1) * (iyLast-iyMin+1);
        tmv::Vector<double> kernel(npts, 0.);
        tmv::Matrix<std::complex<double> > data(_Nimages, npts, std::complex<double>(0.,0.));

        int ipt = 0;
        for (int iy = iyMin; iy <= iyLast; iy++) {
            for (int ix = ixMin; ix <= ixLast; ix++) {
                // sum kernel values for all aliases of this frequency
                double sumk = 0.;
                int iyy=iy;
                while (iyy <= iyMax) {
                    double deltaY = p.y/_dk - iyy;
                    int ixx = ix;
                    while (ixx <= ixMax) {
                        double deltaX = p.x/_dk - ixx;
                        sumk += _kInterp->xval(deltaX, deltaY);
                        ixx += _Nk;
                    }
                    iyy += _Nk;
                }
                // Shift ix,iy into un-aliased zone to get k value
                iyy = iy % _Nk;  
                if(iyy>=_Nk/2) iyy-=_Nk; 
                if(iyy<-_Nk/2) iyy+=_Nk;
                int ixx = ix % _Nk;  
                if(ixx>=_Nk/2) ixx-=_Nk; 
                if(ixx<-_Nk/2) ixx+=_Nk;
                for (int iz=0; iz<_Nimages; iz++) 
                    data(iz, ipt) = _vk[iz]->kval(ixx, iyy);
                kernel[ipt] = sumk;
                ipt++;
            }
        }
        return xKernelTransform*(_wts * data * kernel);
    }

#endif

    // Set maxK to the value where the FT is down to maxk_threshold
    double SBInterpolatedImage::maxK() const 
    {
        // Notice that interpolant other than sinc may make max frequency higher than
        // the Nyquist frequency of the initial image
        
        // Also, since we used kvalue_accuracy for the threshold of _xInterp
        // (at least for the default quintic interpolant) rather than maxk_threshold,
        // this will probably be larger than we really need.
        // We could modify the urange method of Interpolant to take a threshold value
        // at that point, rather than just use the constructor's value, but it's 
        // probably not worth it.  It will probably be very rare that the final maxK
        // value of the FFT will be due to and SBInterpolatedImage.  Usually, this will
        // be convolved by a PSF that will have a smaller maxK.
        return _xInterp->urange() * 2.*M_PI / _dx; 
    }

    // The amount of flux missed in a circle of radius pi/stepk should miss at 
    // most alias_threshold of the flux.
    double SBInterpolatedImage::stepK() const 
    {
        // In this case, R = original image extent + kernel footprint, which we
        // have already stored as _max_size.
        return M_PI / _max_size;
    }


    void SBInterpolatedImage::checkReadyToShoot() const 
    {
        if (_readyToShoot) return;

        // Build the sets holding cumulative fluxes of all Pixels
        checkXsum();
        _positiveFlux = 0.;
        _negativeFlux = 0.;
        _pt.clear();
        for (int iy=-_Ninitial/2; iy<_Ninitial/2; iy++) {
            double y = iy*_dx;
            for (int ix=-_Ninitial/2; ix<_Ninitial/2; ix++) {
                double flux = _xsum->xval(ix,iy) * _dx*_dx;
                if (flux==0.) continue;
                double x=ix*_dx;
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
    PhotonArray SBInterpolatedImage::shoot(int N, UniformDeviate& ud) const
    {
        assert(N>=0);
        checkReadyToShoot();
        /* The pixel coordinates are stored by cumulative absolute flux in 
         * a C++ standard-libary set, so the inversion is done with a binary
         * search tree.  There are no doubt speed gains available from sorting the 
         * pixels by flux, and somehow weighting the tree search to the elements holding
         * the most flux.  But I'm doing it the simplest way right now.
         */
        assert(N>=0);

        PhotonArray result(N);
        if (N<=0 || _pt.empty()) return result;
        double totalAbsFlux = _positiveFlux + _negativeFlux;
        double fluxPerPhoton = totalAbsFlux / N;
        for (int i=0; i<N; i++) {
            double unitRandom = ud();
            Pixel* p = _pt.find(unitRandom);
            result.setPhoton(i, p->x, p->y, 
                             p->isPositive ? fluxPerPhoton : -fluxPerPhoton);
        }

        // Last step is to convolve with the interpolation kernel. 
        // Can skip if using a 2d delta function
        const InterpolantXY* xyPtr = dynamic_cast<const InterpolantXY*> (_xInterp);
        if ( !(xyPtr && dynamic_cast<const Delta*> (xyPtr->get1d())))
             result.convolve(_xInterp->shoot(N, ud));

        return result;
    }

    // instantiate template functions for expected image types
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<float>& img, const Interpolant2d& i, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& img, const Interpolant2d& i, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<short>& img, const Interpolant2d& i, double dx, double padFactor);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<int>& img, const Interpolant2d& i, double dx, double padFactor);
}

