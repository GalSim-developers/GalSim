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

#ifndef SBDECONVOLVE_IMPL_H
#define SBDECONVOLVE_IMPL_H

#include "SBProfileImpl.h"
#include "SBDeconvolve.h"

namespace galsim {

    class SBDeconvolve::SBDeconvolveImpl : public SBProfile::SBProfileImpl
    {
    public:
        SBDeconvolveImpl(const SBProfile& adaptee, boost::shared_ptr<GSParams> gsparams);
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
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        SBProfile _adaptee;
        double _maxksq;

        // Copy constructor and op= are undefined.
        SBDeconvolveImpl(const SBDeconvolveImpl& rhs);
        void operator=(const SBDeconvolveImpl& rhs);
    };

}

#endif // SBDECONVOLVE_IMPL_H
