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

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee,
                               const GSParamsPtr& gsparams) :
        SBProfile(new SBDeconvolveImpl(adaptee,gsparams)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

    std::string SBDeconvolve::SBDeconvolveImpl::serialize() const
    {
        std::ostringstream oss(" ");
        oss << "galsim._galsim.SBDeconvolve(" << _adaptee.serialize();
        oss << ", galsim.GSParams("<<*gsparams<<"))";
        return oss.str();
    }

    SBDeconvolve::SBDeconvolveImpl::SBDeconvolveImpl(const SBProfile& adaptee,
                                                     const GSParamsPtr& _gsparams) :
        SBProfileImpl(_gsparams ? _gsparams : GetImpl(adaptee)->gsparams),
        _adaptee(adaptee)
    {
        double maxk = maxK();
        _maxksq = maxk*maxk;
        double flux = GetImpl(_adaptee)->getFlux();
        _min_acc_kval = flux * gsparams->kvalue_accuracy;
        dbg<<"SBDeconvolve constructor: _maxksq = "<<_maxksq;
        dbg<<", _min_acc_kval = "<<_min_acc_kval<<std::endl;
    }

    // xValue() not implemented for SBDeconvolve.
    double SBDeconvolve::SBDeconvolveImpl::xValue(const Position<double>& p) const
    { throw SBError("SBDeconvolve::xValue() not implemented (and not possible)"); }

    std::complex<double> SBDeconvolve::SBDeconvolveImpl::kValue(const Position<double>& k) const
    {
        double ksq = k.x*k.x + k.y*k.y;
        if (ksq > _maxksq) {
            return 0.;
        } else {
            std::complex<double> kval = _adaptee.kValue(k);
            double abs_kval = std::abs(kval);
            if (abs_kval < _min_acc_kval)
                return 1./_min_acc_kval;
            else
                return 1./kval;
        }
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKImage(ImageView<std::complex<double> > im,
                                                    double kx0, double dkx, int izero,
                                                    double ky0, double dky, int jzero) const
    {
        dbg<<"SBDeconvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,izero,ky0,dky,jzero);

        // Now invert the values, but be careful about not amplifying noise too much.
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<double>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double kysq = ky0*ky0;
            for (int i=0; i<m; ++i,kx+=dkx,++ptr) {
                double ksq = kx*kx + kysq;
                if (ksq > _maxksq) *ptr = 0.;
                else {
                    double abs_kval = std::abs(*ptr);
                    if (abs_kval < _min_acc_kval)
                        *ptr = 1./_min_acc_kval;
                    else
                        *ptr = 1./(*ptr);
                }
            }
        }
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKImage(ImageView<std::complex<double> > im,
                                                    double kx0, double dkx, double dkxy,
                                                    double ky0, double dky, double dkyx) const
    {
        dbg<<"SBDeconvolve fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx);

        // Now invert the values, but be careful about not amplifying noise too much.
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<double>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,++ptr) {
                double ksq = kx*kx + ky*ky;
                if (ksq > _maxksq) *ptr = 0.;
                else {
                    double abs_kval = std::abs(*ptr);
                    if (abs_kval < _min_acc_kval)
                        *ptr = 1./_min_acc_kval;
                    else
                        *ptr = 1./(*ptr);
                }
            }
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
