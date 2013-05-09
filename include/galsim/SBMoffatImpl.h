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
        SBMoffatImpl(double beta, double size, RadiusType rType, double trunc, double flux,
                     boost::shared_ptr<GSParams> gsparams);
        ~SBMoffatImpl() {}

        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const; 

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return (1.-_fluxFactor) > this->gsparams->maxk_threshold; }
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
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _beta; ///< Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
        double _flux; ///< Flux.
        double _norm; ///< Normalization. (Including the flux)
        double _knorm; ///< Normalization for kValue. (Including the flux)
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

        double (*_pow_beta)(double x, double beta);
        double (SBMoffatImpl::*_kV)(double ksq) const;

        /// Setup the FT Table.
        void setupFT() const;

        // These are the (unnormalized) kValue functions for untruncated Moffats
        double kV_15(double ksq) const;
        double kV_2(double ksq) const;
        double kV_25(double ksq) const;
        double kV_3(double ksq) const;
        double kV_35(double ksq) const;
        double kV_4(double ksq) const;
        double kV_gen(double ksq) const;

        // This does the truncated case.
        double kV_trunc(double ksq) const;

        // pow(x,beta) for special (probably not uncommon) cases.
        static double pow_1(double x, double ) { return x; }
        static double pow_15(double x, double ) { return x * sqrt(x); }
        static double pow_2(double x, double ) { return x*x; }
        static double pow_25(double x, double ) { return x*x * sqrt(x); }
        static double pow_3(double x, double ) { return x*x*x; }
        static double pow_35(double x, double ) { return x*x*x * sqrt(x); }
        static double pow_4(double x, double ) { double xsq=x*x; return xsq*xsq; }
        static double pow_gen(double x, double beta) { return std::pow(x,beta); }

        // Copy constructor and op= are undefined.
        SBMoffatImpl(const SBMoffatImpl& rhs);
        void operator=(const SBMoffatImpl& rhs);
    };

}

#endif // SBMOFFAT_IMPL_H

