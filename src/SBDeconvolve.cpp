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

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee) :
        SBProfile(new SBDeconvolveImpl(adaptee)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

    SBDeconvolve::SBDeconvolveImpl::SBDeconvolveImpl(const SBProfile& adaptee) :
        _adaptee(adaptee), _thresh(sbp::kvalue_accuracy * _adaptee.getFlux()),
        _thresh_sq(_thresh*_thresh)
    {
        dbg<<"thresh = "<<_thresh<<std::endl;
    }

    SBDeconvolve::SBDeconvolveImpl::~SBDeconvolveImpl() {}

    // xValue() not implemented for SBDeconvolve.
    double SBDeconvolve::SBDeconvolveImpl::xValue(const Position<double>& p) const 
    { throw SBError("SBDeconvolve::xValue() not implemented"); }

    std::complex<double> SBDeconvolve::SBDeconvolveImpl::kValue(const Position<double>& k) const 
    {
        std::complex<double> temp = _adaptee.kValue(k);
        return std::norm(temp) > _thresh_sq ? 1./temp : 0;
    }

    void SBDeconvolve::SBDeconvolveImpl::kValue(
        tmv::VectorView<double> kx, tmv::VectorView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    {
        GetImpl(_adaptee)->kValue(kx,ky,kval);

        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It kvalit(kval.linearView().begin().getP(),1);
        const int ntot = kval.colsize() * kval.rowsize();
        for (int i=0;i<ntot;++i,++kvalit) 
            *kvalit = std::norm(*kvalit) > _thresh_sq ? 1./(*kvalit) : 0;
    }

    void SBDeconvolve::SBDeconvolveImpl::kValue(
        tmv::MatrixView<double> kx, tmv::MatrixView<double> ky,
        tmv::MatrixView<std::complex<double> > kval) const
    {
        GetImpl(_adaptee)->kValue(kx,ky,kval);

        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It kvalit(kval.linearView().begin().getP(),1);
        const int ntot = kval.colsize() * kval.rowsize();
        for (int i=0;i<ntot;++i,++kvalit) 
            *kvalit = std::norm(*kvalit) > _thresh_sq ? 1./(*kvalit) : 0;
    }

    Position<double> SBDeconvolve::SBDeconvolveImpl::centroid() const 
    { return -_adaptee.centroid(); }

    double SBDeconvolve::SBDeconvolveImpl::getFlux() const 
    { return 1./_adaptee.getFlux(); }

    boost::shared_ptr<PhotonArray> SBDeconvolve::SBDeconvolveImpl::shoot(
        int N, UniformDeviate u) const 
    {
        throw SBError("SBDeconvolve::shoot() not implemented");
        return boost::shared_ptr<PhotonArray>();
    }

}
