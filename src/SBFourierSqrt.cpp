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

#include "SBFourierSqrt.h"
#include "SBFourierSqrtImpl.h"

namespace galsim {

    SBFourierSqrt::SBFourierSqrt(const SBProfile& adaptee, const GSParams& gsparams) :
        SBProfile(new SBFourierSqrtImpl(adaptee,gsparams)) {}

    SBFourierSqrt::SBFourierSqrt(const SBFourierSqrt& rhs) : SBProfile(rhs) {}

    SBProfile SBFourierSqrt::getObj() const
    {
        assert(dynamic_cast<const SBFourierSqrtImpl*>(_pimpl.get()));
        return static_cast<const SBFourierSqrtImpl&>(*_pimpl).getObj();
    }

    SBFourierSqrt::~SBFourierSqrt() {}

    SBFourierSqrt::SBFourierSqrtImpl::SBFourierSqrtImpl(const SBProfile& adaptee,
                                                        const GSParams& gsparams) :
        SBProfileImpl(gsparams), _adaptee(adaptee)
    {
        dbg<<"SBFourierSqrt constructor\n";
    }

    // xValue() not implemented for SBFourierSqrt.
    double SBFourierSqrt::SBFourierSqrtImpl::xValue(const Position<double>& p) const
    { throw SBError("SBFourierSqrt::xValue() not implemented (and not possible)"); }

    std::complex<double> SBFourierSqrt::SBFourierSqrtImpl::kValue(const Position<double>& k) const
    {
        return std::sqrt(_adaptee.kValue(k));
    }

    template <typename T>
    void SBFourierSqrt::SBFourierSqrtImpl::fillKImage(ImageView<std::complex<T> > im,
                                                      double kx0, double dkx, int izero,
                                                      double ky0, double dky, int jzero) const
    {
        dbg<<"SBFourierSqrt fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,izero,ky0,dky,jzero);

        // Now sqrt the values
        const int m = im.getNCol();
        int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (; n; --n,ptr+=skip) {
            for (int i=m; i; --i) {
                std::complex<T> val = *ptr;
                *ptr++ = std::sqrt(val);
            }
        }
    }

    template <typename T>
    void SBFourierSqrt::SBFourierSqrtImpl::fillKImage(ImageView<std::complex<T> > im,
                                                      double kx0, double dkx, double dkxy,
                                                      double ky0, double dky, double dkyx) const
    {
        dbg<<"SBFourierSqrt fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        GetImpl(_adaptee)->fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx);

        // Now sqrt the values
        const int m = im.getNCol();
        int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        const int skip = im.getNSkip();
        assert(im.getStep() == 1);

        for (; n; --n,ptr+=skip) {
            for (int i=m; i; --i) {
                std::complex<T> val = *ptr;
                *ptr++ = std::sqrt(val);
            }
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

    double SBFourierSqrt::SBFourierSqrtImpl::maxSB() const
    {
        // In this case, we want the autoconvolution of this object to get back to the
        // maxSB value of the adaptee.
        // flux * maxSB / 2 = maxSB_adaptee
        // maxSB = 2 * maxSB_adaptee / flux
        return 2. * _adaptee.maxSB() / std::abs(getFlux());
    }

    void SBFourierSqrt::SBFourierSqrtImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        throw SBError("SBFourierSqrt::shoot() not implemented");
    }

}
