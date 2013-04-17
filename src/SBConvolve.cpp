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

#include "SBConvolve.h"
#include "SBConvolveImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBConvolve::SBConvolve(const std::list<SBProfile>& slist, bool real_space,
                           boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBConvolveImpl(slist,real_space,gsparams)) {}

    SBConvolve::SBConvolve(const SBConvolve& rhs) : SBProfile(rhs) {}

    SBConvolve::~SBConvolve() {}

    SBConvolve::SBConvolveImpl::SBConvolveImpl(const std::list<SBProfile>& slist, bool real_space,
                                               boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams.get() ? gsparams :
                      GetImpl(slist.front())->gsparams),
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
        const SBConvolveImpl *sbc = dynamic_cast<const SBConvolveImpl*>(GetImpl(rhs));
        if (sbc) {  
            dbg<<"  (Item is really "<<sbc->_plist.size()<<" items.)\n";
            // If rhs is an SBConvolve, copy its list here
            for (ConstIter pptr = sbc->_plist.begin(); pptr!=sbc->_plist.end(); ++pptr) {
                if (!pptr->isAnalyticK() && !_real_space) 
                    throw SBError("SBConvolve requires members to be analytic in k");
                if (!pptr->isAnalyticX() && _real_space)
                    throw SBError("Real_space SBConvolve requires members to be analytic in x");
                _plist.push_back(*pptr);
            }
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
                return RealSpaceConvolve(p2,p1,pos,_fluxProduct,this->gsparams.get());
            else 
                return RealSpaceConvolve(p1,p2,pos,_fluxProduct,this->gsparams.get());
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
                                                double x0, double dx, int ix_zero,
                                                double y0, double dy, int iy_zero) const
    {
        dbg<<"SBConvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKValue(val,x0,dx,ix_zero,y0,dy,iy_zero);
        if (++pptr != _plist.end()) {
            tmv::Matrix<std::complex<double> > val2(val.colsize(),val.rowsize());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKValue(val2.view(),x0,dx,ix_zero,y0,dy,iy_zero);
                val = ElemProd(val,val2);
            }
        }
    }

    void SBConvolve::SBConvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                double x0, double dx, double dxy,
                                                double y0, double dy, double dyx) const
    {
        dbg<<"SBConvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->fillKValue(val,x0,dx,dxy,y0,dy,dyx);
        if (++pptr != _plist.end()) {
            tmv::Matrix<std::complex<double> > val2(val.colsize(),val.rowsize());
            for (; pptr != _plist.end(); ++pptr) {
                GetImpl(*pptr)->fillKValue(val2.view(),x0,dx,dxy,y0,dy,dyx);
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
    
    SBAutoConvolve::SBAutoConvolve(const SBProfile& s,
                                   boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBAutoConvolveImpl(s, gsparams)) {}
    SBAutoConvolve::SBAutoConvolve(const SBAutoConvolve& rhs) : SBProfile(rhs) {}
    SBAutoConvolve::~SBAutoConvolve() {}

    SBAutoConvolve::SBAutoConvolveImpl::SBAutoConvolveImpl(const SBProfile& s,
                                                           boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams.get() ? gsparams : GetImpl(s)->gsparams),
        _adaptee(s) {}

    double SBAutoConvolve::SBAutoConvolveImpl::xValue(const Position<double>& pos) const
    { return RealSpaceConvolve(_adaptee,_adaptee,pos,getFlux(),this->gsparams.get()); }

    void SBAutoConvolve::SBAutoConvolveImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, int ix_zero,
        double y0, double dy, int iy_zero) const
    {
        dbg<<"SBAutoConvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,ix_zero,y0,dy,iy_zero);
        val = ElemProd(val,val);
    }

    void SBAutoConvolve::SBAutoConvolveImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBConvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,dxy,y0,dy,dyx);
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
    
    SBAutoCorrelate::SBAutoCorrelate(const SBProfile& s,
                                     boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBAutoCorrelateImpl(s, gsparams)) {}
    SBAutoCorrelate::SBAutoCorrelate(const SBAutoCorrelate& rhs) : SBProfile(rhs) {}
    SBAutoCorrelate::~SBAutoCorrelate() {}

    SBAutoCorrelate::SBAutoCorrelateImpl::SBAutoCorrelateImpl(const SBProfile& s,
                                                              boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams.get() ? gsparams : GetImpl(s)->gsparams),
        _adaptee(s) {}

    double SBAutoCorrelate::SBAutoCorrelateImpl::xValue(const Position<double>& pos) const
    { 
        SBProfile temp = _adaptee;
        temp.applyRotation(180. * degrees);
        return RealSpaceConvolve(_adaptee,temp,pos,getFlux(),this->gsparams.get());
    }

    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, int ix_zero,
        double y0, double dy, int iy_zero) const
    {
        dbg<<"SBAutoCorrelate fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,ix_zero,y0,dy,iy_zero);
        val = ElemProd(val,val.conjugate());
    }

    void SBAutoCorrelate::SBAutoCorrelateImpl::fillKValue(
        tmv::MatrixView<std::complex<double> > val,
        double x0, double dx, double dxy,
        double y0, double dy, double dyx) const
    {
        dbg<<"SBCorrelate fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,dxy,y0,dy,dyx);
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
