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

#ifndef SBSERSIC_IMPL_H
#define SBSERSIC_IMPL_H

#include "SBProfileImpl.h"
#include "SBSersic.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Sersic index `n`.
    class SersicInfo 
    {
    public:
        /// @brief Constructor
        SersicInfo(double n, double trunc, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~SersicInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the Sersic function.
         *
         * The input `rsq` should be (r_actual^2 / r0^2).
         * The returned value should then be multiplied by flux * getXNorm() / r0^2.
         */
        double xValue(double rsq) const;

        /**
         * @brief Returns the unnormalized value of the fourier transform.
         *
         * The input `ksq` should be (k_actual^2 * r0^2).
         * The returned value should then be multiplied by flux.
         */
        double kValue(double ksq) const;

        double maxK() const;
        double stepK() const;

        /// @brief The half-light radius in units of r0.
        double getHLR() const;

        /// @brief The fractional flux relative to the untruncated profile.
        double getFluxFraction() const;

        /**
         * @brief The factor by which to multiply the returned value from xValue.
         *
         * Since the returned value needs to be multiplied by flux/r0^2 anyway, we also let
         * the caller of xValue multiply by the normalization, which we calculate for them here.
         */
        double getXNorm() const;

        /// @brief Calculate scale that has the given HLR and truncation radius in physical units.
        double calculateScaleForTruncatedHLR(double re, double trunc) const;

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Sersic profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N  Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:

        SersicInfo(const SersicInfo& rhs); ///< Hide the copy constructor.
        void operator=(const SersicInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _n;       ///< Sersic index.
        double _trunc;   ///< Truncation radius `trunc` in units of r0.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _invn;    ///< 1/n
        double _inv2n;   ///< 1/(2n)
        double _trunc_sq;  ///< trunc^2
        bool _truncated; ///< True if this Sersic profile is truncated.
        double _gamma2n; ///< Gamma(2n) = 1/n * int(exp(-r^1/n)*r,r=0..inf)

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _re;      ///< The HLR in units of r0.
        mutable double _b;       ///< b = re^(1/n)
        mutable double _flux;    ///< Flux relative to the untruncated profile.

        // Parameters for the Hankel transform:
        mutable Table<double,double> _ft;  ///< Lookup table for fourier transform.
        mutable double _kderiv2; ///< Quadratic dependence of F near k=0.
        mutable double _kderiv4; ///< Quartic dependence of F near k=0.
        mutable double _ksq_min; ///< Minimum ksq to use lookup table.
        mutable double _ksq_max; ///< Maximum ksq to use lookup table.
        mutable double _highk_a; ///< Coefficient of 1/k^2 in high-k asymptote
        mutable double _highk_b; ///< Coefficient of 1/k^3 in high-k asymptote

        /// Classes used for photon shooting
        mutable boost::shared_ptr<FluxDensity> _radial;  
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler;   

        // Helper functions used internally:
        void buildFT() const;
        void calculateHLR() const;
        double calculateMissingFluxRadius(double missing_flux_frac) const;
    };

    class SBSersic::SBSersicImpl : public SBProfileImpl
    {
    public:
        SBSersicImpl(double n, double size, RadiusType rType, double flux,
                     double trunc, bool flux_untruncated,
                     const GSParamsPtr& gsparams);

        ~SBSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const 
        {
            splits.push_back(0.);
            if (!_truncated) { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }
            else { xmin = -_trunc; xmax = _trunc; }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            splits.push_back(0.);
            if (!_truncated) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else { ymin = -_trunc; ymax = _trunc; }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const 
        {
            if (!_truncated) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else { ymax = sqrt(_trunc_sq - x*x);  ymin=-ymax; }
            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.); 
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return _truncated; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }

        /// @brief Sersic photon shooting done by rescaling photons from appropriate `SersicInfo`
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        /// @brief Returns the Sersic index n
        double getN() const { return _n; }
        /// @brief Returns the true half-light radius (may be different from the specified value)
        double getHalfLightRadius() const { return _re; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }

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
        double _n;     ///< Sersic index.
        double _flux;  ///< Actual flux (may differ from that specified at the constructor).
        double _r0;    ///< Scale radius specified at the constructor.
        double _re;    ///< Half-light radius specified at the constructor.
        double _trunc;  ///< The truncation radius (if any)
        bool _truncated; ///< True if this Sersic profile is truncated.


        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.
        double _knorm;     ///< Normalization of kValue relative to what SersicInfo returns.
        double _shootnorm; ///< Normalization for photon shooting.

        double _r0_sq;
        double _inv_r0;
        double _inv_r0_sq;
        double _trunc_sq;

        boost::shared_ptr<SersicInfo> _info; ///< Points to info structure for this n,trunc

        // Copy constructor and op= are undefined.
        SBSersicImpl(const SBSersicImpl& rhs);
        void operator=(const SBSersicImpl& rhs);

        static LRUCache<boost::tuple< double, double, GSParamsPtr >, SersicInfo> cache;

    };
}

#endif // SBSERSIC_IMPL_H

