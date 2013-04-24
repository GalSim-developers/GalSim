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

#ifndef SBEXPONENTIAL_IMPL_H
#define SBEXPONENTIAL_IMPL_H

#include "SBProfileImpl.h"
#include "SBExponential.h"
#include "LRUCache.h"

namespace galsim {

    /** 
     * @brief Subclass of `SBExponential` which provides the un-normalized radial function.
     *
     * Serves as interface to `OneDimensionalDeviate` used for sampling from this 
     * distribution.
     */
    class ExponentialRadialFunction : public FluxDensity 
    {
    public:
        /**
         * @brief Constructor
         */
        ExponentialRadialFunction() {};
        /**
         * @brief The un-normalized Exponential function
         * @param[in] r radius, in units of scale radius.
         * @returns Exponential function, normalized to unity at origin
         */
        double operator()(double r) const { return std::exp(-r); } 
    };

    /// @brief A private class that stores photon shooting functions for the Exponential profile
    class ExponentialInfo
    {
    public:
        /** 
         * @brief Constructor
         */
        ExponentialInfo(const GSParams* gsparams); 

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~ExponentialInfo() {}

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Sersic profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double maxK() const;
        double stepK() const;

    private:

        ExponentialInfo(const ExponentialInfo& rhs); ///< Hides the copy constructor.
        void operator=(const ExponentialInfo& rhs); ///<Hide assignment operator.

        /// Function class used for photon shooting
        boost::shared_ptr<ExponentialRadialFunction> _radial;  

        /// Class that does numerical photon shooting
        boost::shared_ptr<OneDimensionalDeviate> _sampler;   

        double _maxk; ///< Calculated maxK * r0
        double _stepk; ///< Calculated stepK * r0
    };

    class SBExponential::SBExponentialImpl : public SBProfileImpl
    {
    public:

        SBExponentialImpl(double r0, double flux, boost::shared_ptr<GSParams> gsparams);
        ~SBExponentialImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; splits.push_back(0.); }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        { 
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; 
            if (std::abs(x/_r0) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getScaleRadius() const { return _r0; }

        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

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
        double _flux; ///< Flux.
        double _r0;   ///< Characteristic size of profile `exp[-(r / r0)]`.
        double _r0_sq;
        double _inv_r0;
        double _inv_r0_sq;
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _norm; ///< flux / r0^2 / 2pi
        double _flux_over_2pi; ///< Flux / 2pi

        const boost::shared_ptr<ExponentialInfo> _info;

        // Copy constructor and op= are undefined.
        SBExponentialImpl(const SBExponentialImpl& rhs);
        void operator=(const SBExponentialImpl& rhs);

        static LRUCache<const GSParams*, ExponentialInfo> cache;
    };

}

#endif // SBEXPONENTIAL_IMPL_H

