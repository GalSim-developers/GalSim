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
        SpergelInfo(double nu, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~SpergelInfo() {}

        double maxK() const;
        double stepK() const;

        /// @brief The half-light radius in units of r0.
        double getHLR() const;

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
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _re;      ///< The HLR in units of r0.
        mutable double _flux;    ///< Flux relative to the untruncated profile.

        /// Classes used for photon shooting
        mutable boost::shared_ptr<FluxDensity> _radial;
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler;

        // Helper functions used internally:
        void calculateHLR() const;
        double calculateMissingFluxRadius(double missing_flux_frac) const;
    };

    class SBSpergel::SBSpergelImpl : public SBProfileImpl
    {
    public:
        SBSpergelImpl(double nu, double size, RadiusType rType, double flux,
                     const GSParamsPtr& gsparams);

        ~SBSpergelImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;


        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF;
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF;
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const
        {
            ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF;
            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.);
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
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

    private:
        double _nu;    ///< Spergel index.
        double _flux;  ///< Actual flux (may differ from that specified at the constructor).
        double _r0;    ///< Scale radius specified at the constructor.

        double _shootnorm; ///< Normalization for photon shooting.

        double _re;
        double _r0_sq;
        double _inv_r0;
        double _flux_over_2pi;
        double _norm;

        boost::shared_ptr<SpergelInfo> _info; ///< Points to info structure for this nu

        // Copy constructor and op= are undefined.
        SBSpergelImpl(const SBSpergelImpl& rhs);
        void operator=(const SBSpergelImpl& rhs);

        static LRUCache<boost::tuple< double, GSParamsPtr >, SpergelInfo> cache;

    };
}

#endif
