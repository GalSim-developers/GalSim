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

#ifndef GalSim_SBSpergelImpl_H
#define GalSim_SBSpergelImpl_H

#include "SBProfileImpl.h"
#include "SBSpergel.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Spergel index `nu`.
    class SpergelInfo
    {
    public:
        /// @brief Constructor
        SpergelInfo(double nu, double trunc, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~SpergelInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the Spergel function.
         *
         * The input `r` should be (r_actual / r0).
         * The returned value should then be multiplied by flux * getXNorm() / r0^2
         */
        double xValue(double r) const;

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
         * Spergel profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] N  Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:

        SpergelInfo(const SpergelInfo& rhs); ///< Hide the copy constructor.
        void operator=(const SpergelInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _nu;       ///< Spergel index.
        double _trunc;   ///< Truncation radius `trunc` in units of r0.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _gamma_nup1; ///< Gamma(nu+1)
        double _gamma_nup2; ///< Gamma(nu+2)
        double _xnorm0   ;  ///< Normalization at r=0 for nu>0
        bool _truncated;  ///< True if this Spergel profile is truncated.

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _re;      ///< The HLR in units of r0.
        mutable double _flux;    ///< Flux relative to the untruncated profile.

        // Parameters for the Hankel transform:
        mutable Table<double,double> _ft; ///< Lookup table for Fourier transform
        mutable double _a1;               ///< First argument (logk) of lookup table
        mutable double _a1ksq;            ///< converted to ksq
        mutable double _fta1;             ///< First value in lookup table

        // Classes used for photon shooting
        mutable boost::shared_ptr<FluxDensity> _radial;
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler;

        // Helper functions used internally:
        void buildFT() const;
        void calculateHLR() const;
        double calculateFluxRadius(const double& flux_frac) const;
    };

    class SBSpergel::SBSpergelImpl : public SBProfileImpl
    {
    public:
        SBSpergelImpl(double nu, double size, RadiusType rType,
                      double flux, double trunc, bool flux_untruncated,
                      const GSParamsPtr& gsparams);

        ~SBSpergelImpl() {}

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
            else if (std::abs(x) >= _trunc) { ymin = 0; ymax = 0; }
            else { ymax = sqrt(_trunc_sq - x*x);  ymin = -ymax; }

            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.);
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return _truncated; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }

        /// @brief Spergel photon shooting done by rescaling photons from appropriate `SpergelInfo`
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        /// @brief Returns the Spergel index nu
        double getNu() const { return _nu; }
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
        double _nu;      ///< Spergel index
        double _flux;    ///< Actual flux (may differ from that specified at the constructor).
        double _r0;      ///< Scale radius specified at the constructor.
        double _re;      ///< Half-light radius specified at the constructor.
        double _trunc;   ///< The truncation radius (if any)
        bool _truncated; ///< True if this Sersic profile is truncated.

        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.
        double _shootnorm; ///< Normalization for photon shooting.

        double _r0_sq;
        double _inv_r0;
        double _trunc_sq;

        boost::shared_ptr<SpergelInfo> _info; ///< Points to info structure for this nu

        // Copy constructor and op= are undefined.
        SBSpergelImpl(const SBSpergelImpl& rhs);
        void operator=(const SBSpergelImpl& rhs);

        static LRUCache<boost::tuple< double, double, GSParamsPtr >, SpergelInfo> cache;
    };
}

#endif
