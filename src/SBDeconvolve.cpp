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

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee, const GSParams& gsparams) :
        SBProfile(new SBDeconvolveImpl(adaptee,gsparams)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

    SBProfile SBDeconvolve::getObj() const
    {
        assert(dynamic_cast<const SBDeconvolveImpl*>(_pimpl.get()));
        return static_cast<const SBDeconvolveImpl&>(*_pimpl).getObj();
    }

    SBDeconvolve::SBDeconvolveImpl::SBDeconvolveImpl(const SBProfile& adaptee,
                                                     const GSParams& gsparams) :
        SBProfileImpl(gsparams), _adaptee(adaptee)
    {
        double maxk = maxK();
        _maxksq = maxk*maxk;
        double flux = GetImpl(_adaptee)->getFlux();
        _min_acc_kval = flux * gsparams.kvalue_accuracy;
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

    template <typename T>
    void SBDeconvolve::SBDeconvolveImpl::fillKImage(ImageView<std::complex<T> > im,
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
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (int j=0; j<n; ++j,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double kysq = ky0*ky0;
            for (int i=0; i<m; ++i,kx+=dkx) {
                double ksq = kx*kx + kysq;
                if (ksq > _maxksq) *ptr++ = T(0);
                else {
                    double abs_kval = std::abs(*ptr);
                    if (abs_kval < _min_acc_kval)
                        *ptr++ = 1./_min_acc_kval;
                    else {
                        std::complex<T> val = *ptr;
                        *ptr++ = 1./(val);
                    }
                }
            }
        }
    }

    template <typename T>
    void SBDeconvolve::SBDeconvolveImpl::fillKImage(ImageView<std::complex<T> > im,
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
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (int j=0; j<n; ++j,kx0+=dkxy,ky0+=dky,ptr+=skip) {
            double kx = kx0;
            double ky = ky0;
            for (int i=0; i<m; ++i,kx+=dkx,ky+=dkyx) {
                double ksq = kx*kx + ky*ky;
                if (ksq > _maxksq) *ptr++ = 0.;
                else {
                    double abs_kval = std::abs(*ptr);
                    if (abs_kval < _min_acc_kval)
                        *ptr++ = 1./_min_acc_kval;
                    else {
                        std::complex<T> val = *ptr;
                        *ptr++ = 1./(val);
                    }
                }
            }
        }
    }

    Position<double> SBDeconvolve::SBDeconvolveImpl::centroid() const
    { return -_adaptee.centroid(); }

    double SBDeconvolve::SBDeconvolveImpl::getFlux() const
    { return 1./_adaptee.getFlux(); }

    double SBDeconvolve::SBDeconvolveImpl::maxSB() const
    {
        // The only way to really give this any meaning is to consider it in the context
        // of being part of a larger convolution with other components.  The calculation
        // of maxSB for Convolve is
        //     maxSB = flux_final / Sum_i (flux_i / maxSB_i)
        //
        // A deconvolution will contribute a -sigma^2 to the sum, so a logical choice for
        // maxSB is to have flux / maxSB = -flux_adaptee / maxSB_adaptee, so its contribution
        // to the Sum_i 2pi sigma^2 is to subtract its adaptee's value of sigma^2.
        //
        // maxSB = -flux * maxSB_adaptee / flux_adaptee
        //       = -maxSB_adaptee / flux_adaptee^2
        //
        return -_adaptee.maxSB() / std::abs(_adaptee.getFlux() * _adaptee.getFlux());
    }
 
    void SBDeconvolve::SBDeconvolveImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        throw SBError("SBDeconvolve::shoot() not implemented");
    }

}
