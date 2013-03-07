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

#ifndef SBBOX_IMPL_H
#define SBBOX_IMPL_H

#include "SBProfileImpl.h"
#include "SBBox.h"

namespace galsim {

    class SBBox::SBBoxImpl : public SBProfileImpl 
    {
    public:
        SBBoxImpl(double xw, double yw, double flux) :
            _xw(xw), _yw(yw), _flux(flux)
        {
            if (_yw==0.) _yw=_xw; 
            _norm = _flux / (_xw * _yw);
        }

        ~SBBoxImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return false; } 
        bool hasHardEdges() const { return true; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -0.5*_xw;  xmax = 0.5*_xw; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -0.5*_yw;  ymax = 0.5*_yw; }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        double getXWidth() const { return _xw; }
        double getYWidth() const { return _yw; }

        /// @brief Boxcar is trivially sampled by drawing 2 uniform deviates.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        // Override both for efficiency and to put in fractional edge values which
        // don't happen with normal calls to xValue.
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        // Overrides for better efficiency
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _xw;   ///< Boxcar function is `xw` x `yw` across.
        double _yw;   ///< Boxcar function is `xw` x `yw` across.
        double _flux; ///< Flux.
        double _norm; ///< Calculated value: flux / (xw*yw)

        // Sinc function used to describe Boxcar in k space. 
        double sinc(double u) const; 

        // Copy constructor and op= are undefined.
        SBBoxImpl(const SBBoxImpl& rhs);
        void operator=(const SBBoxImpl& rhs);
    };

}

#endif // SBBOX_IMPL_H

