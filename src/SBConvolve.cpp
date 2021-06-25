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

#include "SBConvolve.h"
#include "SBConvolveImpl.h"
#include "SBTransform.h"

namespace galsim {

    SBConvolve::SBConvolve(const std::list<SBProfile>& plist, bool real_space,
                           const GSParams& gsparams) :
        SBProfile(new SBConvolveImpl(plist,real_space,gsparams)) {}

    SBConvolve::SBConvolve(const SBConvolve& rhs) : SBProfile(rhs) {}

    SBConvolve::~SBConvolve() {}

    std::list<SBProfile> SBConvolve::getObjs() const
    {
        assert(dynamic_cast<const SBConvolveImpl*>(_pimpl.get()));
        return static_cast<const SBConvolveImpl&>(*_pimpl).getObjs();
    }

    bool SBConvolve::isRealSpace() const
    {
        assert(dynamic_cast<const SBConvolveImpl*>(_pimpl.get()));
        return static_cast<const SBConvolveImpl&>(*_pimpl).isRealSpace();
    }

    double SBConvolve::SBConvolveImpl::maxSB() const
    {
        // This one is probably the least accurate of all the estimates of maxSB.
        // The calculation is based on the exact value for Gaussians.
        //     maxSB = flux / 2pi sigma^2
        // When convolving multiple Gaussians together, the sigma^2 values add:
        //     sigma_final^2 = Sum_i sigma_i^2
        // from which we can calculate
        //     maxSB = flux_final / 2pi sigma_final^2
        // or
        //     maxSB = flux_final / Sum_i (flux_i / maxSB_i)
        //
        // For non-Gaussians, this procedure will tend to produce an over-estimate of the
        // true maximum SB.  Non-Gaussian profiles tend to have peakier parts which get smoothed
        // more than the Gaussian does.  So this is likely to be too high, which is acceptable.
        ConstIter sptr = _plist.begin();
        double twopisigmasq = sptr->getFlux() / sptr->maxSB();
        for (++sptr; sptr!=_plist.end(); ++sptr)
            twopisigmasq += std::abs(sptr->getFlux()) / sptr->maxSB();
        return _fluxProduct / twopisigmasq;
    }

    SBConvolve::SBConvolveImpl::SBConvolveImpl(const std::list<SBProfile>& plist, bool real_space,
                                               const GSParams& gsparams) :
        SBProfileImpl(gsparams), _real_space(real_space),
        _x0(0.), _y0(0.), _isStillAxisymmetric(true), _fluxProduct(1.),
        _maxk(0.), _stepk(0.)
    {
        for(ConstIter it=plist.begin(); it!=plist.end(); ++it) add(*it);
    }

    void SBConvolve::SBConvolveImpl::add(const SBProfile& sbp)
    {
        dbg<<"Start SBConvolveImpl::add.  Adding item # "<<_plist.size()+1<<std::endl;

        // Add new terms(s) to the _plist:
        assert(GetImpl(sbp));
        const SBProfileImpl* p = GetImpl(sbp);
        const SBConvolveImpl* sbc = dynamic_cast<const SBConvolveImpl*>(p);
        const SBAutoConvolve::SBAutoConvolveImpl* sbc2 =
            dynamic_cast<const SBAutoConvolve::SBAutoConvolveImpl*>(p);
        const SBAutoCorrelate::SBAutoCorrelateImpl* sbc3 =
            dynamic_cast<const SBAutoCorrelate::SBAutoCorrelateImpl*>(p);
        if (sbc) {
            dbg<<"  (Item is really "<<sbc->_plist.size()<<" items.)"<<std::endl;
            // If sbp is an SBConvolve, copy its list here
            for (ConstIter pptr = sbc->_plist.begin(); pptr!=sbc->_plist.end(); ++pptr) add(*pptr);
        } else if (sbc2) {
            dbg<<"  (Item is really AutoConvolve.)"<<std::endl;
            // If sbp is an SBAutoConvolve, put two of its item here:
            const SBProfile& obj = sbc2->getAdaptee();
            add(obj);
            add(obj);
        } else if (sbc3) {
            dbg<<"  (Item is really AutoCorrelate items.)"<<std::endl;
            // If sbp is an SBAutoCorrelate, put its item and 180 degree rotated verion here:
            const SBProfile& obj = sbc3->getAdaptee();
            add(obj);
            SBProfile temp = obj.transform(-1., 0., 0., -1.);
            add(temp);
        } else {
            if (!sbp.isAnalyticK() && !_real_space)
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!sbp.isAnalyticX() && _real_space)
                throw SBError("Real-space SBConvolve requires members to be analytic in x");
            _plist.push_back(sbp);
            _x0 += sbp.centroid().x;
            _y0 += sbp.centroid().y;
            _isStillAxisymmetric = _isStillAxisymmetric && sbp.isAxisymmetric();
            _fluxProduct *= sbp.getFlux();
        }
    }

    double SBConvolve::SBConvolveImpl::maxK() const
    {
        if (_maxk == 0.) {
            for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
                double it_maxk = it->maxK();
                dbg<<"SBConvolve component has maxK = "<<it_maxk<<std::endl;
                if (_maxk <= 0. || it_maxk < _maxk) _maxk = it_maxk;
            }
            dbg<<"Net maxK = "<<_maxk<<std::endl;
        }
        return _maxk;
    }

    double SBConvolve::SBConvolveImpl::stepK() const
    {
        if (_stepk == 0.) {
            for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
                double it_stepk = it->stepK();
                dbg<<"SBConvolve component has stepK = "<<it_stepk<<std::endl;
                _stepk += 1./(it_stepk*it_stepk);  // Accumulate Sum 1/stepk^2
            }
            _stepk = 1./sqrt(_stepk);  // Convert to (Sum 1/stepk^2)^(-1/2)
            dbg<<"Net stepK = "<<_stepk<<std::endl;
        }
        return _stepk;
    }

    double SBConvolve::SBConvolveImpl::xValue(const Position<double>& pos) const
    {
        // Perform a direct calculation of the convolution at a particular point by
        // doing the real-space integral.
        // Note: This can only really be done one pair at a time, so it is
        // probably rare that this will be more efficient if N > 2.
        // For now, we don't bother implementing this for N > 2.

        if (_plist.size() == 2) {
            const SBProfile& p1 = _plist.front();
            const SBProfile& p2 = _plist.back();
            if (p2.isAxisymmetric())
                return RealSpaceConvolve(p2,p1,pos,_fluxProduct,this->gsparams);
            else
                return RealSpaceConvolve(p1,p2,pos,_fluxProduct,this->gsparams);
        } else if (_plist.empty())
            return 0.;
        else if (_plist.size() == 1)
            return _plist.front().xValue(pos);
        else
            throw SBError("Real-space integration of more than 2 profiles is not implemented.");
    }

    std::complex<double> SBConvolve::SBConvolveImpl::kValue(const Position<double>& k) const
    {
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        std::complex<double> kv = pptr->kValue(k);
        for (++pptr; pptr != _plist.end(); ++pptr) kv *= pptr->kValue(k);
        return kv;
    }

    template <typename T>
    void SBConvolve::SBConvolveImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBConvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKImage(im,kx0,dkx,izero,ky0,dky,jzero);
        if (++pptr != _plist.end()) {
            ImageAlloc<std::complex<T> > im2(im.getBounds());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKImage(im2.view(),kx0,dkx,izero,ky0,dky,jzero);
                im *= im2;
            }
        }
    }

    template <typename T>
    void SBConvolve::SBConvolveImpl::fillKImage(ImageView<std::complex<T> > im,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBConvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx);
        if (++pptr != _plist.end()) {
            ImageAlloc<std::complex<T> > im2(im.getBounds());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKImage(im2.view(),kx0,dkx,dkxy,ky0,dky,dkyx);
                im *= im2;
            }
        }
    }

    double SBConvolve::SBConvolveImpl::getPositiveFlux() const
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        double pResult = pptr->getPositiveFlux();
        double nResult = pptr->getNegativeFlux();
        for (++pptr; pptr!=_plist.end(); ++pptr) {
            double p = pptr->getPositiveFlux();
            double n = pptr->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return pResult;
    }

    // Note duplicated code here, could be caching results for tiny efficiency gain
    double SBConvolve::SBConvolveImpl::getNegativeFlux() const
    {
        if (_plist.empty()) return 0.;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        double pResult = pptr->getPositiveFlux();
        double nResult = pptr->getNegativeFlux();
        for (++pptr; pptr!=_plist.end(); ++pptr) {
            double p = pptr->getPositiveFlux();
            double n = pptr->getNegativeFlux();
            double pNew = p*pResult + n*nResult;
            nResult = p*nResult + n*pResult;
            pResult = pNew;
        }
        return nResult;
    }

    void SBConvolve::SBConvolveImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Convolve shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        if (pptr==_plist.end())
            throw SBError("Cannot shoot() for empty SBConvolve");
        pptr->shoot(photons, ud);
        // It may be necessary to shuffle when convolving because we do
        // do not have a gaurantee that the convolvee's photons are
        // uncorrelated, e.g. they might both have their negative ones
        // at the end.
        // However, this decision is now made by the convolve method.
        for (++pptr; pptr != _plist.end(); ++pptr) {
            PhotonArray temp(N);
            pptr->shoot(temp, ud);
            photons.convolve(temp, ud);
        }
        dbg<<"Convolve Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    //
    // AutoConvolve
    //

    SBAutoConvolve::SBAutoConvolve(const SBProfile& s, bool real_space,
                                   const GSParams& gsparams) :
        SBProfile(new SBAutoConvolveImpl(s, real_space, gsparams)) {}
    SBAutoConvolve::SBAutoConvolve(const SBAutoConvolve& rhs) : SBProfile(rhs) {}
    SBAutoConvolve::~SBAutoConvolve() {}

    SBProfile SBAutoConvolve::getObj() const
    {
        assert(dynamic_cast<const SBAutoConvolveImpl*>(_pimpl.get()));
        return static_cast<const SBAutoConvolveImpl&>(*_pimpl).getObj();
    }

    bool SBAutoConvolve::isRealSpace() const
    {
        assert(dynamic_cast<const SBAutoConvolveImpl*>(_pimpl.get()));
        return static_cast<const SBAutoConvolveImpl&>(*_pimpl).isRealSpace();
    }

    double SBAutoConvolve::SBAutoConvolveImpl::maxSB() const
    {
        // f^2 / (f/sb + f/sb) = f*sb/2
        return _adaptee.getFlux() * _adaptee.maxSB() / 2.;
    }

    SBAutoConvolve::SBAutoConvolveImpl::SBAutoConvolveImpl(const SBProfile& s, bool real_space,
                                                           const GSParams& gsparams) :
        SBProfileImpl(gsparams), _adaptee(s), _real_space(real_space) {}

    double SBAutoConvolve::SBAutoConvolveImpl::xValue(const Position<double>& pos) const
    { return RealSpaceConvolve(_adaptee,_adaptee,pos,getFlux(),this->gsparams); }

    template <typename T>
    struct Square
    { T operator()(T x) { return x*x; } };

    template <typename T>
    void SBAutoConvolve::SBAutoConvolveImpl::fillKImage(ImageView<std::complex<T> > im,
                                                        double kx0, double dkx, int izero,
                                                        double ky0, double dky, int jzero) const
    {
        dbg<<"SBAutoConvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,izero,ky0,dky,jzero);
        transform_pixel(im, Square<std::complex<T> >());
    }

    template <typename T>
    void SBAutoConvolve::SBAutoConvolveImpl::fillKImage(ImageView<std::complex<T> > im,
                                                        double kx0, double dkx, double dkxy,
                                                        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBAutoConvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx);
        transform_pixel(im, Square<std::complex<T> >());
    }

    double SBAutoConvolve::SBAutoConvolveImpl::getPositiveFlux() const
    {
        double p = _adaptee.getPositiveFlux();
        double n = _adaptee.getNegativeFlux();
        return p*p + n*n;
    }

    double SBAutoConvolve::SBAutoConvolveImpl::getNegativeFlux() const
    {
        double p = _adaptee.getPositiveFlux();
        double n = _adaptee.getNegativeFlux();
        return 2.*p*n;
    }

    void SBAutoConvolve::SBAutoConvolveImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"AutoConvolve shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        _adaptee.shoot(photons, ud);
        PhotonArray temp(N);
        _adaptee.shoot(temp, ud);
        photons.convolve(temp, ud);
        dbg<<"AutoConvolve Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }


    //
    // AutoCorrelate
    //

    SBAutoCorrelate::SBAutoCorrelate(const SBProfile& s, bool real_space,
                                     const GSParams& gsparams) :
        SBProfile(new SBAutoCorrelateImpl(s, real_space, gsparams)) {}
    SBAutoCorrelate::SBAutoCorrelate(const SBAutoCorrelate& rhs) : SBProfile(rhs) {}
    SBAutoCorrelate::~SBAutoCorrelate() {}

    SBProfile SBAutoCorrelate::getObj() const
    {
        assert(dynamic_cast<const SBAutoCorrelateImpl*>(_pimpl.get()));
        return static_cast<const SBAutoCorrelateImpl&>(*_pimpl).getObj();
    }

    bool SBAutoCorrelate::isRealSpace() const
    {
        assert(dynamic_cast<const SBAutoCorrelateImpl*>(_pimpl.get()));
        return static_cast<const SBAutoCorrelateImpl&>(*_pimpl).isRealSpace();
    }

    double SBAutoCorrelate::SBAutoCorrelateImpl::maxSB() const
    {
        return _adaptee.getFlux() * _adaptee.maxSB() / 2.;
    }

    SBAutoCorrelate::SBAutoCorrelateImpl::SBAutoCorrelateImpl(
        const SBProfile& s, bool real_space, const GSParams& gsparams) :
        SBProfileImpl(gsparams), _adaptee(s), _real_space(real_space) {}

    double SBAutoCorrelate::SBAutoCorrelateImpl::xValue(const Position<double>& pos) const
    {
        SBProfile temp = _adaptee.transform(-1., 0., 0., -1.);
        return RealSpaceConvolve(_adaptee,temp,pos,getFlux(),this->gsparams);
    }

    template <typename T>
    struct AbsSquare
    { T operator()(T x) { return std::norm(x); } };

    template <typename T>
    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKImage(ImageView<std::complex<T> > im,
                                                          double kx0, double dkx, int izero,
                                                          double ky0, double dky, int jzero) const
    {
        dbg<<"SBAutoCorrelate fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,izero,ky0,dky,jzero);
        transform_pixel(im, AbsSquare<std::complex<T> >());
    }

    template <typename T>
    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKImage(ImageView<std::complex<T> > im,
                                                          double kx0, double dkx, double dkxy,
                                                          double ky0, double dky, double dkyx) const
    {
        dbg<<"SBAutoCorrelate fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx);
        transform_pixel(im, AbsSquare<std::complex<T> >());
    }

    double SBAutoCorrelate::SBAutoCorrelateImpl::getPositiveFlux() const
    {
        double p = _adaptee.getPositiveFlux();
        double n = _adaptee.getNegativeFlux();
        return p*p + n*n;
    }

    double SBAutoCorrelate::SBAutoCorrelateImpl::getNegativeFlux() const
    {
        double p = _adaptee.getPositiveFlux();
        double n = _adaptee.getNegativeFlux();
        return 2.*p*n;
    }

    void SBAutoCorrelate::SBAutoCorrelateImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"AutoCorrelate shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        _adaptee.shoot(photons, ud);
        PhotonArray temp(N);
        _adaptee.shoot(temp, ud);
        // Flip sign of (x,y) in one of the results
        temp.scaleXY(-1.);
        photons.convolve(temp, ud);
        dbg<<"AutoCorrelate Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

}
