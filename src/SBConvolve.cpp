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

    SBConvolve::SBConvolve(const SBProfile& s1, const SBProfile& s2, bool real_space) :
        SBProfile(new SBConvolveImpl(s1,s2,real_space)) {}

    SBConvolve::SBConvolve(const SBProfile& s1, const SBProfile& s2, const SBProfile& s3,
                           bool real_space) :
        SBProfile(new SBConvolveImpl(s1,s2,s3,real_space)) {}

    SBConvolve::SBConvolve(const std::list<SBProfile>& slist, bool real_space) :
        SBProfile(new SBConvolveImpl(slist,real_space)) {}

    SBConvolve::SBConvolve(const SBConvolve& rhs) : SBProfile(rhs) {}

    SBConvolve::~SBConvolve() {}

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
                return RealSpaceConvolve(p2,p1,pos,_fluxProduct);
            else 
                return RealSpaceConvolve(p1,p2,pos,_fluxProduct);
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

    void SBConvolve::SBConvolveImpl::kValue(
        tmv::VectorView<double> kx, tmv::VectorView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    {
        tmv::Vector<double> kx2 = kx;
        tmv::Vector<double> ky2 = ky;
        tmv::Matrix<std::complex<double> > kval2(kval.colsize(),kval.rowsize());

        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->kValue(kx,ky,kval);
        for (++pptr; pptr != _plist.end(); ++pptr) {
            kx = kx2;
            ky = ky2;
            GetImpl(*pptr)->kValue(kx,ky,kval2.view());
            kval = ElemProd(kval,kval2);
        }
    }

    void SBConvolve::SBConvolveImpl::kValue(
        tmv::MatrixView<double> kx, tmv::MatrixView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    {
        tmv::Matrix<double> kx2 = kx;
        tmv::Matrix<double> ky2 = ky;
        tmv::Matrix<std::complex<double> > kval2(kval.colsize(),kval.rowsize());

        ConstIter pptr = _plist.begin();
        assert(pptr != _plist.end());
        GetImpl(*pptr)->kValue(kx,ky,kval);
        for (++pptr; pptr != _plist.end(); ++pptr) {
            kx = kx2;
            ky = ky2;
            GetImpl(*pptr)->kValue(kx,ky,kval2.view());
            kval = ElemProd(kval,kval2);
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
}
