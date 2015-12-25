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

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee,
                               const GSParamsPtr& gsparams) :
        SBProfile(new SBDeconvolveImpl(adaptee,gsparams)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

    std::string SBDeconvolve::SBDeconvolveImpl::repr() const 
    {
        std::ostringstream oss(" ");
        oss << "galsim._galsim.SBDeconvolve(" << _adaptee.repr();
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBDeconvolve::SBDeconvolveImpl::SBDeconvolveImpl(const SBProfile& adaptee,
                                                     const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams ? gsparams : GetImpl(adaptee)->gsparams),
        _adaptee(adaptee)
    {
        double maxk = maxK();
        _maxksq = maxk*maxk;
        dbg<<"SBDeconvolve constructor: _maxksq = "<<_maxksq<<std::endl;
    }

    // xValue() not implemented for SBDeconvolve.
    double SBDeconvolve::SBDeconvolveImpl::xValue(const Position<double>& p) const 
    { throw SBError("SBDeconvolve::xValue() not implemented (and not possible)"); }

    std::complex<double> SBDeconvolve::SBDeconvolveImpl::kValue(const Position<double>& k) const 
    {
        return (k.x*k.x + k.y*k.y <= _maxksq) ? 1./_adaptee.kValue(k) : 0.;
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double kx0, double dkx, int izero,
                                                    double ky0, double dky, int jzero) const
    {
        dbg<<"SBDeconvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,izero,ky0,dky,jzero);

        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,ky0+=dky) {
            double kx = kx0;
            double kysq = ky0*ky0;
            for (int i=0;i<m;++i,kx+=dkx,++valit) 
                *valit = (kx*kx+kysq <= _maxksq) ? 1./(*valit) : 0.;
        }
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double kx0, double dkx, double dkxy,
                                                    double ky0, double dky, double dkyx) const
    {
        dbg<<"SBDeconvolve fillKValue\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,kx0,dkx,dkxy,ky0,dky,dkyx);

        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It valit = val.linearView().begin();
        for (int j=0;j<n;++j,kx0+=dkxy,ky0+=dky) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0;i<m;++i,kx+=dkx,ky+=dkyx,++valit) 
                *valit = (kx*kx+ky*ky <= _maxksq) ? 1./(*valit) : 0.;
        }
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
