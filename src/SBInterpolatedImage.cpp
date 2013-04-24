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

//#define DEBUGLOGGING

#include <algorithm>
#include "SBInterpolatedImage.h"
#include "SBInterpolatedImageImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
std::ostream* dbgout = &std::cout;
int verbose_level = 2;
#endif

namespace galsim {

    template <typename T> 
    SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<T>& image,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        double dx, double pad_factor, boost::shared_ptr<Image<T> > pad_image,
        boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBInterpolatedImageImpl(image,xInterp,kInterp,dx,pad_factor,pad_image,
                                              gsparams)) {}

    SBInterpolatedImage::SBInterpolatedImage(
        const MultipleImageHelper& multi, const std::vector<double>& weights,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBInterpolatedImageImpl(multi,weights,xInterp,kInterp,gsparams)) {}

    SBInterpolatedImage::SBInterpolatedImage(const SBInterpolatedImage& rhs) : SBProfile(rhs) {}

    SBInterpolatedImage::~SBInterpolatedImage() {}

    void SBInterpolatedImage::calculateStepK() const 
    { 
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateStepK(); 
    }

    void SBInterpolatedImage::calculateMaxK() const 
    {
        assert(dynamic_cast<const SBInterpolatedImageImpl*>(_pimpl.get()));
        return static_cast<const SBInterpolatedImageImpl&>(*_pimpl).calculateMaxK(); 
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
        _pimpl->Ninitial = _pimpl->Ninitial + (_pimpl->Ninitial+1)%2;
        assert(_pimpl->Ninitial%2==1);
        assert(_pimpl->Ninitial>=3);

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

        if (pad_factor <= 0.) pad_factor = sbp::default_pad_factor;
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
            for (int iy = img.getYMin(); iy<= img.getYMax(); ++iy, ++y) {
                int x = xStart;
                for (int ix = img.getXMin(); ix<= img.getXMax(); ++ix, ++x) {
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
    MultipleImageHelper::MultipleImageHelper(const BaseImage<T>& image, double dx, 
        double pad_factor, boost::shared_ptr<Image<T> > pad_image) :
        _pimpl(new MultipleImageHelperImpl)
    {
        dbg<<"Start MultipleImageHelper constructor for one image\n";
        dbg<<"image bounds = "<<image.getBounds()<<std::endl;

        if (dx<=0.) _pimpl->dx = image.getScale();
        else _pimpl->dx = dx;
        dbg<<"image scale = "<<image.getScale()<<std::endl;
        dbg<<"dx = "<<_pimpl->dx<<std::endl;

        dbg<<"pad_factor = "<<pad_factor<<std::endl;
        _pimpl->Ninitial = std::max( image.getYMax()-image.getYMin()+1,
                                    image.getXMax()-image.getXMin()+1 );
        _pimpl->Ninitial = _pimpl->Ninitial + (_pimpl->Ninitial+1)%2;
        _pimpl->Nk = image.getPaddedSize(pad_factor);
        dbg<<"Ninitial = "<<_pimpl->Ninitial<<std::endl;
        assert(_pimpl->Ninitial%2==1);
        assert(_pimpl->Ninitial>=3);
        dbg<<"Nk = "<<_pimpl->Nk<<std::endl;

        double dx2 = _pimpl->dx*_pimpl->dx;
        double dx3 = _pimpl->dx*dx2;

        // fill data from images, shifting to center the image in the table
        _pimpl->vx.resize(1);
        _pimpl->vk.resize(1);
        _pimpl->flux.resize(1);
        _pimpl->xflux.resize(1);
        _pimpl->yflux.resize(1);
        _pimpl->vx[0].reset(new XTable(_pimpl->Nk, _pimpl->dx));

        if (pad_image.get()) {
            // Start by copying the pad_image
            // make sure images are same size (but don't worry if bounds are not same)
            if ((1+pad_image->getXMax()-pad_image->getXMin() != _pimpl->Nk) ||
                (1+pad_image->getYMax()-pad_image->getYMin() != _pimpl->Nk)) {
                char err_buff[500];
                sprintf(err_buff,"Supplied image of noise for padding is wrong size: "
                        "received %d by %d, expected %d by %d\n",
                        1+pad_image->getXMax()-pad_image->getXMin(),
                        1+pad_image->getYMax()-pad_image->getYMin(),
                        _pimpl->Nk,_pimpl->Nk);
                throw std::runtime_error(err_buff);
            }
            dbg<<"Copying pad_image\n";
            int xStart = -((pad_image->getXMax()-pad_image->getXMin()+1)/2);
            int y = -((pad_image->getYMax()-pad_image->getYMin()+1)/2);
            dbg<<"xStart = "<<xStart<<", yStart = "<<y<<std::endl;
            for (int iy = pad_image->getYMin(); iy<=pad_image->getYMax(); ++iy, ++y) {
                int x = xStart;
                for (int ix = pad_image->getXMin(); ix<=pad_image->getXMax(); ++ix, ++x) {
                    _pimpl->vx[0]->xSet(x, y, (*pad_image)(ix,iy));
                }
            }
        }

        // Copy the given image to the center
        int xStart = -((image.getXMax()-image.getXMin()+1)/2);
        int y = -((image.getYMax()-image.getYMin()+1)/2);
        dbg<<"xStart = "<<xStart<<", yStart = "<<y<<std::endl;
        for (int iy = image.getYMin(); iy<= image.getYMax(); iy++, y++) {
            int x = xStart;
            for (int ix = image.getXMin(); ix<= image.getXMax(); ix++, x++) {
                _pimpl->vx[0]->xSet(x, y, image(ix,iy));
            }
        }

        // Accumulate the flux and centroid on a square region of size Ninitial x Ninitial
        // (This isn't precisely the same as what was in the original image, since the
        // padding can have some flux, so we do it this way to be consistent with how
        // we use it later.)
        double sum = 0.;
        double sumx = 0.;
        double sumy = 0.;
        const int Nino2 = _pimpl->Ninitial/2;
        for (int y = -Nino2; y<=Nino2; ++y) {
            for (int x = -Nino2; x<=Nino2; ++x) {
                double value = _pimpl->vx[0]->xval(x, y);
                sum += value;
                sumx += value*x;
                sumy += value*y;
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
        double dx, double pad_factor, boost::shared_ptr<Image<T> > pad_image,
        boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams),
        _multi(image,dx,pad_factor,pad_image), _wts(1,1.), _xInterp(xInterp), _kInterp(kInterp),
        _readyToShoot(false)
    { initialize(); }

    SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const MultipleImageHelper& multi, const std::vector<double>& weights,
        boost::shared_ptr<Interpolant2d> xInterp, boost::shared_ptr<Interpolant2d> kInterp,
        boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams),
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
            for (size_t i=1; i<_multi.size(); ++i) 
                _xtab->accumulate(*_multi.getXTable(i), _wts[i]);
        }
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;

        // Calculate stepK:
        // 
        // The amount of flux missed in a circle of radius pi/stepk should be at 
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
        _uscale = _multi.getScale() / (2.*M_PI);
        _maxk = _maxk1 = _xInterp->urange()/_uscale;
        dbg<<"maxk = "<<_maxk<<std::endl;

        _flux = calculateFlux();
        dbg<<"flux = "<<getFlux()<<std::endl;
    }

    SBInterpolatedImage::SBInterpolatedImageImpl::~SBInterpolatedImageImpl() {}

    double SBInterpolatedImage::SBInterpolatedImageImpl::calculateFlux() const 
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
            for (size_t i=1; i<_multi.size(); ++i)
                _ktab->accumulate(*_multi.getKTable(i), _wts[i]);
        }
        dbg<<"Built ktab\n";
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
    }

    double SBInterpolatedImage::SBInterpolatedImageImpl::xValue(const Position<double>& p) const 
    { return _xtab->interpolate(p.x, p.y, *_xInterp); }

    std::complex<double> SBInterpolatedImage::SBInterpolatedImageImpl::kValue(
        const Position<double>& k) const 
    {
        // Don't bother if the desired k value is cut off by the x interpolant:
        if (std::abs(k.x) > _maxk1 || std::abs(k.y) > _maxk1) return std::complex<double>(0.,0.);
        checkK();
        double xKernelTransform = _xInterp->uval(k.x*_uscale, k.y*_uscale);
        return xKernelTransform * _ktab->interpolate(k.x, k.y, *_kInterp);
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXValue(
        tmv::MatrixView<double> val,
        double x0, double dx, int ix_zero,
        double y0, double dy, int iy_zero) const
    {
        dbg<<"SBInterpolatedImage fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();

        if (dynamic_cast<const InterpolantXY*> (_xInterp.get())) {
            // If the interpolant is separable, the XTable interpolation routine
            // will go faster if we make y iteration the inner loop.
            typedef tmv::VIt<double,tmv::Unknown,tmv::NonConj> RMIt;
            for (int i=0;i<m;++i,x0+=dx) {
                double y = y0;
                RMIt valit = val.row(i).begin();
                for (int j=0;j<n;++j,y+=dy) *valit++ = _xtab->interpolate(x0, y, *_xInterp); 
            }
        } else {
            // Otherwise, just do the values in storage order
            typedef tmv::VIt<double,1,tmv::NonConj> CMIt;
            for (int j=0;j<n;++j,y0+=dy) {
                double x = x0;
                CMIt valit = val.col(j).begin();
                for (int i=0;i<m;++i,x+=dx) *valit++ = _xtab->interpolate(x, y0, *_xInterp); 
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, int ix_zero,
        double y0, double dy, int iy_zero) const
    {
        dbg<<"SBInterpolatedImage fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        assert(val.stepi() == 1);
        const int m = val.colsize();
        const int n = val.rowsize();
        checkK();

        // Assign zeros for range that has |u| > maxu
        int i1 = std::max( int((-_maxk1-x0)/dx) , 0 );
        int i2 = std::min( int((_maxk1-x0)/dx)+1 , m );
        int j1 = std::max( int((-_maxk1-y0)/dy) , 0 );
        int j2 = std::min( int((_maxk1-y0)/dy)+1 , n );
        val.colRange(0,j1).setZero();
        val.subMatrix(0,i1,j1,j2).setZero();
        val.subMatrix(i2,m,j1,j2).setZero();
        val.colRange(j2,n).setZero();
        x0 += i1*dx;
        y0 += j1*dy;
        xdbg<<"_maxk1 = "<<_maxk1<<std::endl;
        xdbg<<"i1,i2 = "<<i1<<','<<i2<<std::endl;
        xdbg<<"j1,j2 = "<<j1<<','<<j2<<std::endl;

        // For the rest of the range, calculate ux, uy values
        tmv::Vector<double> ux(i2-i1);
        typedef tmv::VIt<double,1,tmv::NonConj> It;
        It uxit = ux.begin();
        double x = x0;
        for (int i=i1;i<i2;++i,x+=dx) *uxit++ = x * _uscale;
            
        tmv::Vector<double> uy(j2-j1);
        It uyit = uy.begin();
        double y = y0;
        for (int j=j1;j<j2;++j,y+=dy) *uyit++ = y * _uscale;

        const InterpolantXY* kInterpXY = dynamic_cast<const InterpolantXY*>(_kInterp.get());
        if (kInterpXY) {
            // Again, the KTable interpolation routine will go faster if we make y iteration 
            // the inner loop.
            typedef tmv::VIt<std::complex<double>,tmv::Unknown,tmv::NonConj> RMIt;

            const InterpolantXY* xInterpXY = dynamic_cast<const InterpolantXY*>(_xInterp.get());
            if (xInterpXY) {
                // Then the uval's are separable.  Go ahead and pre-calculate them.
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,++uxit) *uxit = xInterpXY->uval1d(*uxit);
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,++uyit) *uyit = xInterpXY->uval1d(*uyit);

                uxit = ux.begin();
                for (int i=i1;i<i2;++i,x0+=dx,++uxit) {
                    double y = y0;
                    uyit = uy.begin();
                    RMIt valit(val.row(i).begin().getP(),val.stepj());
                    for (int j=j1;j<j2;++j,y+=dy) {
                        *valit++ = *uxit * *uyit++ * _ktab->interpolate(x0, y, *kInterpXY);
                    }
                }
            } else {
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,x0+=dx,++uxit) {
                    double y = y0;
                    It uyit = uy.begin();
                    RMIt valit(val.row(i).begin().getP(),val.stepj());
                    for (int j=j1;j<j2;++j,y+=dy) {
                        double xKernelTransform = _xInterp->uval(*uxit, *uyit++);
                        *valit++ = xKernelTransform * _ktab->interpolate(x0, y, *kInterpXY);
                    }
                }
            }
        } else {
            const InterpolantXY* xInterpXY = dynamic_cast<const InterpolantXY*>(_xInterp.get());
            if (xInterpXY) {
                It uxit = ux.begin();
                for (int i=i1;i<i2;++i,++uxit) *uxit = xInterpXY->uval1d(*uxit);
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,++uyit) *uyit = xInterpXY->uval1d(*uyit);

                typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
                CIt valit(val.linearView().begin().getP(),1);
                uyit = uy.begin();
                for (int j=j1;j<j2;++j,y0+=dy,++uyit) {
                    double x = x0;
                    uxit = ux.begin();
                    for (int i=i1;i<i2;++i,x+=dx) {
                        *valit++ = *uxit++ * *uyit * _ktab->interpolate(x, y0, *_kInterp);
                    }
                }
            } else {
                typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> CIt;
                CIt valit(val.linearView().begin().getP(),1);
                It uyit = uy.begin();
                for (int j=j1;j<j2;++j,y0+=dy,++uyit) {
                    double x = x0;
                    It uxit = ux.begin();
                    for (int i=i1;i<i2;++i,x+=dx) {
                        double xKernelTransform = _xInterp->uval(*uxit++, *uyit);
                        *valit++ = xKernelTransform * _ktab->interpolate(x, y0, *_kInterp);
                    }
                }
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillXValue(
        tmv::MatrixView<double> val,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBInterpolatedImage fillXValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<double,1,tmv::NonConj> It;

        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx) {
                *valit++ = _xtab->interpolate(x, y, *_xInterp); 
            }
        }
    }

    void SBInterpolatedImage::SBInterpolatedImageImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBInterpolatedImage fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        checkK();

        double ux0 = x0 * _uscale;
        double uy0 = y0 * _uscale;
        double dux = dx * _uscale;
        double duy = dy * _uscale;
        double duxy = dxy * _uscale;
        double duyx = dyx * _uscale;

        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy,ux0+=duxy,uy0+=duy) {
            double x = x0;
            double y = y0;
            double ux = ux0;
            double uy = uy0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx,ux+=dux,uy+=duyx) {
                if (std::abs(x) > _maxk1 || std::abs(y) > _maxk1) {
                    *valit++ = 0.;
                } else {
                    double xKernelTransform = _xInterp->uval(ux, uy);
                    *valit++ = xKernelTransform * _ktab->interpolate(x, y, *_kInterp);
                }
            }
        }
    }

    // We provide an option to update the stepk value by directly calculating what
    // size region around the center encloses (1-alias_threshold) of the total flux.
    // This can be useful if you make the image bigger than you need to, just to be
    // safe, but then want to use as large a stepk value as possible.
    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateStepK() const
    {
        dbg<<"Start SBInterpolatedImage calculateStepK()\n";
        dbg<<"Current value of stepk = "<<_stepk<<std::endl;
        dbg<<"Find box that encloses "<<1.-this->gsparams->alias_threshold<<" of the flux.\n";
        dbg<<"xtab size = "<<_xtab->getN()<<", scale = "<<_xtab->getDx()<<std::endl;
        //int N = _xtab->getN();
        double dx = _xtab->getDx();
        double dx2 = dx*dx;
        double fluxTot = getFlux()/dx2;
        dbg<<"fluxTot = "<<fluxTot<<std::endl;
        double flux = (*_xtab).xval(0,0);
        double thresh = (1.-this->gsparams->alias_threshold) * fluxTot;
        dbg<<"thresh = "<<thresh<<std::endl;

        // d1 = 0 means that we haven't yet found the d that enclosed enough flux.
        // When we find a flux > thresh, we set d1 = d.
        // However, since the function can have negative regions, we need to keep 
        // going to make sure an oscillation doesn't bring us back below thresh.
        // When this happens, we set d1 to 0 again and look for a larger value that 
        // enclosed enough flux again.
        int d1 = 0; 
        const int Nino2 = _multi.getNin()/2;
        for (int d=1; d<=Nino2; ++d) {
            xdbg<<"d = "<<d<<std::endl;
            xdbg<<"d1 = "<<d1<<std::endl;
            xdbg<<"flux = "<<flux<<std::endl;
            // Add the left, right, top and bottom sides of box:
            for(int x = -d; x < d; ++x) {
                // Note: All 4 corners are added exactly once by including x=-d but omitting 
                // x=d from the loop.
                flux += _xtab->xval(x,-d);  // bottom
                flux += _xtab->xval(d,x);   // right
                flux += _xtab->xval(-x,d);  // top
                flux += _xtab->xval(-d,-x); // left
            }
            if (flux < thresh) {
                d1 = 0; // Mark that we haven't gotten to a good enclosing radius yet.
            } else {
                if (d1 == 0) d1 = d; // Mark this radius as a good one.
            }
        }
        dbg<<"Done: flux = "<<flux<<std::endl;
        // Should have added up to the total flux.
        assert( std::abs(flux - fluxTot) < 1.e-3 * std::abs(fluxTot) );
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

    // The std library norm function uses abs to get a more accurate value.
    // We don't actually care about the slight accuracy gain, so we use a 
    // fast norm that just does x^2 + y^2
    inline double fast_norm(const std::complex<double>& z)
    { return real(z)*real(z) + imag(z)*imag(z); }

    void SBInterpolatedImage::SBInterpolatedImageImpl::calculateMaxK() const
    {
        dbg<<"Start SBInterpolatedImage calculateMaxK()\n";
        dbg<<"Current value of maxk = "<<_maxk<<std::endl;
        dbg<<"Find the smallest k such that all values outside of this are less than "
            <<this->gsparams->maxk_threshold<<std::endl;
        checkK();
        dbg<<"ktab size = "<<_ktab->getN()<<", scale = "<<_ktab->getDk()<<std::endl;
        
        double dk = _ktab->getDk();

        // Among the elements with kval > thresh, find the one with the maximum ksq
        double thresh = this->gsparams->maxk_threshold * getFlux();
        thresh *= thresh; // Since values will be |kval|^2.
        double maxk_ix = 0.;
        // When we get 5 rows in a row all below thresh, stop.
        int n_below_thresh = 0;
        int N = _ktab->getN();
        // Don't go past the current value of maxk
        if (N/2 * dk > _maxk) 
            N = int(_maxk*2./dk);
        // We take the k value to be maximum of kx and ky.  This is appropriate, because
        // this is how maxK() is eventually used -- it sets the size in k-space for both
        // kx and ky when drawing.  Since kx<0 is just the conjugate of the corresponding
        // point at (-kx,-ky), we only check the right half of the square.  i.e. the 
        // upper-right and lower-right quadrants.
        for(int ix=0; ix<=N/2; ++ix) {
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
        for (int iy=-_multi.getNin()/2; iy<_multi.getNin()/2; ++iy) {
            double y = iy*_multi.getScale();
            for (int ix=-_multi.getNin()/2; ix<_multi.getNin()/2; ++ix) {
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
        for (int i=0; i<N; ++i) {
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
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<float> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<double> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<int32_t>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<int32_t> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImage(
        const BaseImage<int16_t>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<int16_t> > pad_image,
        boost::shared_ptr<GSParams> gsparams);

    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<float> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<double> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<int32_t> > >& images,
        double dx, double pad_factor);
    template MultipleImageHelper::MultipleImageHelper(
        const std::vector<boost::shared_ptr<BaseImage<int16_t> > >& images,
        double dx, double pad_factor);

    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<float>& image, double dx, double pad_factor,
        boost::shared_ptr<Image<float> > pad_image);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<double>& image, double dx, double pad_factor,
        boost::shared_ptr<Image<double> > pad_image);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<int32_t>& image, double dx, double pad_factor,
        boost::shared_ptr<Image<int32_t> > pad_image);
    template MultipleImageHelper::MultipleImageHelper(
        const BaseImage<int16_t>& image, double dx, double pad_factor,
        boost::shared_ptr<Image<int16_t> > pad_image);

    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<float>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<float> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<double>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<double> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<int32_t>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<int32_t> > pad_image,
        boost::shared_ptr<GSParams> gsparams);
    template SBInterpolatedImage::SBInterpolatedImageImpl::SBInterpolatedImageImpl(
        const BaseImage<int16_t>& image, boost::shared_ptr<Interpolant2d> xInterp,
        boost::shared_ptr<Interpolant2d> kInterp, double dx, double pad_factor,
        boost::shared_ptr<Image<int16_t> > pad_image,
        boost::shared_ptr<GSParams> gsparams);

} // namespace galsim
