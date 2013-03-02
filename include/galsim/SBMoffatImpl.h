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

#ifndef SBMOFFAT_IMPL_H
#define SBMOFFAT_IMPL_H

#include "SBProfileImpl.h"
#include "SBMoffat.h"

namespace galsim {

    class SBMoffat::SBMoffatImpl : public SBProfileImpl 
    {
    public:
        SBMoffatImpl(double beta, double size, RadiusType rType, double trunc, double flux);

        ~SBMoffatImpl() {}

        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const; 

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return (1.-_fluxFactor) > sbp::maxk_threshold; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_maxR; xmax = _maxR; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_maxR; ymax = _maxR; }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& ) const 
        {
            ymax = sqrt(_maxR_sq - x*x);
            ymin = -ymax;
        }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }


        double getFlux() const { return _flux; }

        /**
         * @brief Moffat photon shooting is done by analytic inversion of cumulative flux 
         * distribution.
         *
         * Will require 2 uniform deviates per photon, plus analytic function (pow and sqrt)
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double getBeta() const { return _beta; }
        double getScaleRadius() const { return _rD; }
        double getFWHM() const { return _FWHM; }
        double getHalfLightRadius() const;

        // Overrides for better efficiency 
        void xValue(tmv::VectorView<double> x, tmv::VectorView<double> y,
                    tmv::MatrixView<double> val) const;
        void xValue(tmv::MatrixView<double> x, tmv::MatrixView<double> y,
                    tmv::MatrixView<double> val) const;
        void kValue(tmv::VectorView<double> kx, tmv::VectorView<double> ky,
                    tmv::MatrixView<std::complex<double> > kval) const;
        void kValue(tmv::MatrixView<double> kx, tmv::MatrixView<double> ky,
                    tmv::MatrixView<std::complex<double> > kval) const;

    private:
        double _beta; ///< Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
        double _flux; ///< Flux.
        double _norm; ///< Normalization. (Including the flux)
        double _rD;   ///< Scale radius for profile `[1 + (r / rD)^2]^beta`.
        double _maxR; ///< Maximum `r`
        double _maxRrD; ///< maxR/rD
        double _FWHM;  ///< Full Width at Half Maximum.
        double _trunc;  ///< Outer truncation radius in same physical units as `_rD`
        double _fluxFactor; ///< Integral of total flux in terms of 'rD' units.
        double _rD_sq;
        double _inv_rD;
        double _inv_rD_sq;
        double _maxRrD_sq;
        double _maxR_sq;
        mutable double _maxK; ///< Maximum k with kValue > 1.e-3

        mutable Table<double,double> _ft;  ///< Lookup table for Fourier transform of Moffat.

        mutable double _re; ///< Stores the half light radius if set or calculated post-setting.

        double (*pow_beta)(double x, double beta);

        /// Setup the FT Table.
        void setupFT() const;

        // Copy constructor and op= are undefined.
        SBMoffatImpl(const SBMoffatImpl& rhs);
        void operator=(const SBMoffatImpl& rhs);
    };

}

#endif // SBMOFFAT_IMPL_H

