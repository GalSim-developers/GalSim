/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBConvolve::SBConvolve(const std::list<SBProfile>& slist, bool real_space,
                           const GSParamsPtr& gsparams) :
        SBProfile(new SBConvolveImpl(slist,real_space,gsparams)) {}

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

    std::string SBConvolve::SBConvolveImpl::repr() const 
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBConvolve([";
        ConstIter sptr = _plist.begin(); 
        oss << sptr->repr();
        for (++sptr; sptr!=_plist.end(); ++sptr) oss << ", " << sptr->repr();
        oss << "], galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBConvolve::SBConvolveImpl::SBConvolveImpl(const std::list<SBProfile>& slist, bool real_space,
                                               const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams ? gsparams : GetImpl(slist.front())->gsparams),
        _real_space(real_space)
    {
        for (ConstIter sptr = slist.begin(); sptr!=slist.end(); ++sptr)
            add(*sptr);
        initialize(); 
    }

    void SBConvolve::SBConvolveImpl::add(const SBProfile& rhs) 
    {
        dbg<<"Start SBConvolveImpl::add.  Adding item # "<<_plist.size()+1<<std::endl;

        // Add new terms(s) to the _plist:
        assert(GetImpl(rhs));
        const SBProfileImpl* p = GetImpl(rhs);
        const SBConvolveImpl* sbc = dynamic_cast<const SBConvolveImpl*>(p);
        const SBAutoConvolve::SBAutoConvolveImpl* sbc2 =
            dynamic_cast<const SBAutoConvolve::SBAutoConvolveImpl*>(p);
        const SBAutoCorrelate::SBAutoCorrelateImpl* sbc3 =
            dynamic_cast<const SBAutoCorrelate::SBAutoCorrelateImpl*>(p);
        if (sbc) {
            dbg<<"  (Item is really "<<sbc->_plist.size()<<" items.)"<<std::endl;
            // If rhs is an SBConvolve, copy its list here
            for (ConstIter pptr = sbc->_plist.begin(); pptr!=sbc->_plist.end(); ++pptr) {
                if (!pptr->isAnalyticK() && !_real_space) 
                    throw SBError("SBConvolve requires members to be analytic in k");
                if (!pptr->isAnalyticX() && _real_space)
                    throw SBError("Real_space SBConvolve requires members to be analytic in x");
                _plist.push_back(*pptr);
            }
        } else if (sbc2) {
            dbg<<"  (Item is really AutoConvolve.)"<<std::endl;
            // If rhs is an SBAutoConvolve, put two of its item here:
            const SBProfile& obj = sbc2->getAdaptee();
            if (!obj.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!obj.isAnalyticX() && _real_space)
                throw SBError("Real_space SBConvolve requires members to be analytic in x");
            _plist.push_back(obj);
            _plist.push_back(obj);
        } else if (sbc3) {
            dbg<<"  (Item is really AutoCorrelate items.)"<<std::endl;
            // If rhs is an SBAutoCorrelate, put its item and 180 degree rotated verion here:
            const SBProfile& obj = sbc3->getAdaptee();
            if (!obj.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!obj.isAnalyticX() && _real_space)
                throw SBError("Real_space SBConvolve requires members to be analytic in x");
            _plist.push_back(obj);
            SBProfile temp = obj.rotate(180. * degrees);
            _plist.push_back(temp);
        } else {
            if (!rhs.isAnalyticK() && !_real_space) 
                throw SBError("SBConvolve requires members to be analytic in k");
            if (!rhs.isAnalyticX() && _real_space)
                throw SBError("Real-space SBConvolve requires members to be analytic in x");
            _plist.push_back(rhs);
        }
    }

    void SBConvolve::SBConvolveImpl::initialize()
    {
        _x0 = _y0 = 0.;
        _fluxProduct = 1.;
        _minMaxK = 0.;
        _isStillAxisymmetric = true;

        _netStepK = 0.;  // Accumulate Sum 1/stepk^2
        for(ConstIter it=_plist.begin(); it!=_plist.end(); ++it) {
            double maxk = it->maxK();
            double stepk = it->stepK();
            dbg<<"SBConvolve component has maxK, stepK = "<<maxk<<" , "<<stepk<<std::endl;
            _fluxProduct *= it->getFlux();
            _x0 += it->centroid().x;
            _y0 += it->centroid().y;
            if ( _minMaxK<=0. || maxk < _minMaxK) _minMaxK = maxk;
            _netStepK += 1./(stepk*stepk);
            _isStillAxisymmetric = _isStillAxisymmetric && it->isAxisymmetric();
        }
        _netStepK = 1./sqrt(_netStepK);  // Convert to (Sum 1/stepk^2)^(-1/2)
        dbg<<"Net maxK, stepK = "<<_minMaxK<<" , "<<_netStepK<<std::endl;
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

    void SBConvolve::SBConvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                double kx0, double dkx, int izero,
                                                double ky0, double dky, int jzero) const
    {
        dbg<<"SBConvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKValue(val,kx0,dkx,izero,ky0,dky,jzero);
        if (++pptr != _plist.end()) {
            tmv::Matrix<std::complex<double> > val2(val.colsize(),val.rowsize());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKValue(val2.view(),kx0,dkx,izero,ky0,dky,jzero);
                val = ElemProd(val,val2);
            }
        }
    }

    void SBConvolve::SBConvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                double kx0, double dkx, double dkxy,
                                                double ky0, double dky, double dkyx) const
    {
        dbg<<"SBConvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKValue(val,kx0,dkx,dkxy,ky0,dky,dkyx);
        if (++pptr != _plist.end()) {
            tmv::Matrix<std::complex<double> > val2(val.colsize(),val.rowsize());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKValue(val2.view(),kx0,dkx,dkxy,ky0,dky,dkyx);
                val = ElemProd(val,val2);
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

    boost::shared_ptr<PhotonArray> SBConvolve::SBConvolveImpl::shoot(int N, UniformDeviate u) const 
    {
        dbg<<"Convolve shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        std::list<SBProfile>::const_iterator pptr = _plist.begin();
        if (pptr==_plist.end())
            throw SBError("Cannot shoot() for empty SBConvolve");
        boost::shared_ptr<PhotonArray> result = pptr->shoot(N, u);
        // It may be necessary to shuffle when convolving because we do
        // do not have a gaurantee that the convolvee's photons are
        // uncorrelated, e.g. they might both have their negative ones
        // at the end.
        // However, this decision is now made by the convolve method.
        for (++pptr; pptr != _plist.end(); ++pptr)
            result->convolve(*pptr->shoot(N, u), u);
        dbg<<"Convolve Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

    //
    // AutoConvolve 
    // 

    SBAutoConvolve::SBAutoConvolve(const SBProfile& s, bool real_space,
                                   const GSParamsPtr& gsparams) :
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

    std::string SBAutoConvolve::SBAutoConvolveImpl::repr() const 
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBAutoConvolve(" << _adaptee.repr() << ", ";
        if (_real_space) oss << "True";
        else oss << "False";
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBAutoConvolve::SBAutoConvolveImpl::SBAutoConvolveImpl(const SBProfile& s, bool real_space,
                                                           const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams ? gsparams : GetImpl(s)->gsparams),
        _adaptee(s), _real_space(real_space) {}

    double SBAutoConvolve::SBAutoConvolveImpl::xValue(const Position<double>& pos) const
    { return RealSpaceConvolve(_adaptee,_adaptee,pos,getFlux(),this->gsparams); }

    void SBAutoConvolve::SBAutoConvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                        double kx0, double dkx, int izero,
                                                        double ky0, double dky, int jzero) const
    {
        dbg<<"SBAutoConvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,izero,ky0,dky,jzero);
        val = ElemProd(val,val);
    }

    void SBAutoConvolve::SBAutoConvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                        double kx0, double dkx, double dkxy,
                                                        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBAutoConvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,dkxy,ky0,dky,dkyx);
        val = ElemProd(val,val);
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

    boost::shared_ptr<PhotonArray> SBAutoConvolve::SBAutoConvolveImpl::shoot(
        int N, UniformDeviate u) const 
    {
        dbg<<"AutoConvolve shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result = _adaptee.shoot(N, u);
        result->convolve(*_adaptee.shoot(N, u), u);
        dbg<<"AutoConvolve Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }


    //
    // AutoCorrelate
    // 

    SBAutoCorrelate::SBAutoCorrelate(const SBProfile& s, bool real_space,
                                     const GSParamsPtr& gsparams) :
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

    std::string SBAutoCorrelate::SBAutoCorrelateImpl::repr() const 
    {
        std::ostringstream oss(" ");
        oss.precision(std::numeric_limits<double>::digits10 + 4);
        oss << "galsim._galsim.SBAutoCorrelate(" << _adaptee.repr() << ", ";
        if (_real_space) oss << "True";
        else oss << "False";
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBAutoCorrelate::SBAutoCorrelateImpl::SBAutoCorrelateImpl(
        const SBProfile& s, bool real_space,
        const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams ? gsparams : GetImpl(s)->gsparams),
        _adaptee(s), _real_space(real_space) {}

    double SBAutoCorrelate::SBAutoCorrelateImpl::xValue(const Position<double>& pos) const
    { 
        SBProfile temp = _adaptee.rotate(180. * degrees);
        return RealSpaceConvolve(_adaptee,temp,pos,getFlux(),this->gsparams);
    }

    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double kx0, double dkx, int izero,
        double ky0, double dky, int jzero) const
    {
        dbg<<"SBAutoCorrelate fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,izero,ky0,dky,jzero);
        val = ElemProd(val,val.conjugate());
    }

    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double kx0, double dkx, double dkxy,
        double ky0, double dky, double dkyx) const
    {
        dbg<<"SBCorrelate fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,dkxy,ky0,dky,dkyx);
        val = ElemProd(val,val.conjugate());
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

    boost::shared_ptr<PhotonArray> SBAutoCorrelate::SBAutoCorrelateImpl::shoot(
        int N, UniformDeviate u) const 
    {
        dbg<<"AutoCorrelate shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        boost::shared_ptr<PhotonArray> result = _adaptee.shoot(N, u);
        boost::shared_ptr<PhotonArray> result2 = _adaptee.shoot(N, u);
        // Flip sign of (x,y) in one of the results
        for (int i=0; i<result2->size(); i++) {
            Position<double> negxy = -Position<double>(result2->getX(i), result2->getY(i));
            result2->setPhoton(i, negxy.x, negxy.y, result2->getFlux(i));
        }
        result->convolve(*result2, u);
        dbg<<"AutoCorrelate Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }

}
