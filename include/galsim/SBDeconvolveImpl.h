/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBDeconvolveImpl_H
#define GalSim_SBDeconvolveImpl_H

#include "SBProfileImpl.h"
#include "SBDeconvolve.h"

namespace galsim {

    class SBDeconvolve::SBDeconvolveImpl : public SBProfile::SBProfileImpl
    {
    public:
        SBDeconvolveImpl(const SBProfile& adaptee, const GSParamsPtr& gsparams);
        ~SBDeconvolveImpl() {}

        // xValue() not implemented for SBDeconvolve.
        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const { return _adaptee.maxK(); }
        double stepK() const { return _adaptee.stepK(); }

        bool isAxisymmetric() const { return _adaptee.isAxisymmetric(); }

        // Of course, a deconvolution could have hard edges, but since we can't use this
        // in a real-space convolution anyway, just return false here.
        bool hasHardEdges() const { return false; }

        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const;
        double getFlux() const;

        // shoot also not implemented.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate u) const;

        // Overrides for better efficiency
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        SBProfile _adaptee;
        double _maxksq;

        // Copy constructor and op= are undefined.
        SBDeconvolveImpl(const SBDeconvolveImpl& rhs);
        void operator=(const SBDeconvolveImpl& rhs);
    };

}

#endif 
