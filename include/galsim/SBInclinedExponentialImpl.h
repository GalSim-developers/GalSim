/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBInclinedExponentialImpl_H
#define GalSim_SBInclinedExponentialImpl_H

#include "SBProfileImpl.h"
#include "SBInclinedExponential.h"
#include "LRUCache.h"
#include "Table.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each inclined exponential angle 'i'
    class InclinedExponentialInfo
    {
    public:
        /// @brief Constructor
    	InclinedExponentialInfo(double h_tani_over_r, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~InclinedExponentialInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the Inclined Exponential function.
         *
         * The input 'rx' should be r/r_scale in the direction perpendicular to the minor axis,
         * and the input 'ry' value should be r/h_tani in the direction parallel to the minor axis
         */
        double xValue(double rx, double ry) const;

        /**
         * @brief Returns the unnormalized value of the fourier transform.
         *
         * The input 'kx' should be k*r_scale in the direction perpendicular to the minor axis,
         * and the input 'ky' value should be k*h_tani in the direction parallel to the minor axis
         */
        double kValue(double kx, double ky) const;

        double maxK() const;
        double stepK() const;

        /// @brief The fractional flux relative to the untruncated profile.
        double getFluxFraction() const;

        /**
         * @brief The factor by which to multiply the returned value from xValue.
         *
         * Since the returned value needs to be multiplied by flux/r0^2 anyway, we also let
         * the caller of xValue multiply by the normalization, which we calculate for them here.
         */
        double getXNorm() const;

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Inclined profiles are sampled with a numerical method, using class
         * `UniformDeviate`.
         *
         * @param[in] N  Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:

        InclinedExponentialInfo(const InclinedExponentialInfo& rhs); ///< Hide the copy constructor.
        void operator=(const InclinedExponentialInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _h_tani_over_r;
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _ksq_max;   ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_min;   ///< If ksq > _kq_max, then use kvalue = 0

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.

        // Parameters for the inverse Fourier transform
        /* NYI
        mutable Table2D<double,double> _ift;  ///< Lookup table for inverse Fourier transform.
        */

        // Classes used for photon shooting
        /* NYI
        mutable boost::shared_ptr<FluxDensity> _radial;
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler;
        */

        // Helper functions used internally:
        /* NYI
        void buildIFT() const;
        */
    };

    class SBInclinedExponential::SBInclinedExponentialImpl : public SBProfileImpl
    {
    public:
    	SBInclinedExponentialImpl(double i, double scale_radius, double scale_height, double flux,
                 const GSParamsPtr& gsparams);

        ~SBInclinedExponentialImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            xmin = -integ::MOCK_INF;
            xmax = integ::MOCK_INF;
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            ymin = -integ::MOCK_INF;
            ymax = integ::MOCK_INF;
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            ymin = -integ::MOCK_INF;
            ymax = integ::MOCK_INF;
        }

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }  // Will be in future version though
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }

        /// @brief photon shooting done by rescaling photons from appropriate `InclinedExponentialInfo`
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        /// @brief Returns the inclination angle i in radians
        double getI() const { return _i; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns the scale radius
        double getScaleHeight() const { return _h0; }

        // Overrides for better efficiency
        /* NYI
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        */
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

        std::string serialize() const;

    private:
        double _i;       ///< Inclination angle 'i' in radians
        double _flux;    ///< Actual flux (may differ from that specified at the constructor).
        double _r0;      ///< Scale radius specified at the constructor.
        double _h0;      ///< Scale height specified at the constructor.

        /* NYI
        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.
        double _shootnorm; ///< Normalization for photon shooting.
        */

        double _inv_r0;
        double _h_tani_over_r;
        double _r0_cosi;
        double _inv_r0_cosi;

        boost::shared_ptr<InclinedExponentialInfo> _info; ///< Points to info structure for this n,trunc

        // Copy constructor and op= are undefined.
        SBInclinedExponentialImpl(const SBInclinedExponentialImpl& rhs);
        void operator=(const SBInclinedExponentialImpl& rhs);

        static LRUCache<boost::tuple< double, GSParamsPtr >, InclinedExponentialInfo> cache;

    };
}

#endif
