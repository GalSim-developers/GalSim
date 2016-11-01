/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#include "SBFourierSqrt.h"
#include "SBFourierSqrtImpl.h"

namespace galsim {

    SBFourierSqrt::SBFourierSqrt(const SBProfile& adaptee,
                                 const GSParamsPtr& gsparams) :
        SBProfile(new SBFourierSqrtImpl(adaptee,gsparams)) {}

    SBFourierSqrt::SBFourierSqrt(const SBFourierSqrt& rhs) : SBProfile(rhs) {}

    SBProfile SBFourierSqrt::getObj() const
    {
        assert(dynamic_cast<const SBFourierSqrtImpl*>(_pimpl.get()));
        return static_cast<const SBFourierSqrtImpl&>(*_pimpl).getObj();
    }

    SBFourierSqrt::~SBFourierSqrt() {}

    std::string SBFourierSqrt::SBFourierSqrtImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss << "galsim._galsim.SBFourierSqrt(" << _adaptee.serialize();
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBFourierSqrt::SBFourierSqrtImpl::SBFourierSqrtImpl(const SBProfile& adaptee,
                                                        const GSParamsPtr& gsparams) :
        SBProfileImpl(gsparams ? gsparams : GetImpl(adaptee)->gsparams),
        _adaptee(adaptee)
    {
        double maxk = maxK();
        _maxksq = maxk*maxk;
        dbg<<"SBFourierSqrt constructor: _maxksq = "<<_maxksq<<std::endl;
    }

    // xValue() not implemented for SBFourierSqrt.
    double SBFourierSqrt::SBFourierSqrtImpl::xValue(const Position<double>& p) const
    { throw SBError("SBFourierSqrt::xValue() not implemented (and not possible)"); }

    std::complex<double> SBFourierSqrt::SBFourierSqrtImpl::kValue(const Position<double>& k) const
    {
        return (k.x*k.x + k.y*k.y <= _maxksq) ? std::sqrt(_adaptee.kValue(k)) : 0.;
    }

    void SBFourierSqrt::SBFourierSqrtImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                      double kx0, double dkx, int izero,
                                                      double ky0, double dky, int jzero) const
    {
        dbg<<"SBFourierSqrt fillKValue\n";
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
                *valit = (kx*kx+kysq <= _maxksq) ? std::sqrt(*valit) : 0.;
        }
    }

    void SBFourierSqrt::SBFourierSqrtImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                      double kx0, double dkx, double dkxy,
                                                      double ky0, double dky, double dkyx) const
    {
        dbg<<"SBFourierSqrt fillKValue\n";
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
                *valit = (kx*kx+ky*ky <= _maxksq) ? std::sqrt(*valit) : 0.;
        }
    }

    Position<double> SBFourierSqrt::SBFourierSqrtImpl::centroid() const
    {
        return 0.5*_adaptee.centroid();
    }

    double SBFourierSqrt::SBFourierSqrtImpl::getFlux() const
    {
        return std::sqrt(_adaptee.getFlux());
    }

    boost::shared_ptr<PhotonArray> SBFourierSqrt::SBFourierSqrtImpl::shoot(
        int N, UniformDeviate u) const
    {
        throw SBError("SBFourierSqrt::shoot() not implemented");
        return boost::shared_ptr<PhotonArray>();
    }

}
