/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBSersicImpl_H
#define GalSim_SBSersicImpl_H

#include "SBProfileImpl.h"
#include "SBInclinedSersic.h"
#include "SBSersic.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"

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
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

    private:

        SersicInfo(const SersicInfo& rhs); ///< Hide the copy constructor.
        void operator=(const SersicInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _n;       ///< Sersic index.
        double _trunc;   ///< Truncation radius `trunc` in units of r0.
        GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _invn;      ///< 1/n
        double _inv2n;     ///< 1/(2n)
        double _trunc_sq;  ///< trunc^2
        bool _truncated;   ///< True if this Sersic profile is truncated.
        double _gamma2n;   ///< Gamma(2n) = 1/n * int(exp(-r^1/n)*r,r=0..inf)

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _re;      ///< The HLR in units of r0.
        mutable double _b;       ///< b = re^(1/n)
        mutable double _flux;    ///< Flux relative to the untruncated profile.

        // Parameters for the Hankel transform:
        mutable TableBuilder _ft;  ///< Lookup table for Fourier transform.
        mutable double _kderiv2; ///< Quadratic dependence of F near k=0.
        mutable double _kderiv4; ///< Quartic dependence of F near k=0.
        mutable double _ksq_min; ///< Minimum ksq to use lookup table.
        mutable double _ksq_max; ///< Maximum ksq to use lookup table.
        mutable double _highk_a; ///< Coefficient of 1/k^2 in high-k asymptote
        mutable double _highk_b; ///< Coefficient of 1/k^3 in high-k asymptote

        // Classes used for photon shooting
        mutable shared_ptr<FluxDensity> _radial;
        mutable shared_ptr<OneDimensionalDeviate> _sampler;

        // Helper functions used internally:
        void buildFT() const;
        void calculateHLR() const;
        double calculateMissingFluxRadius(double missing_flux_frac) const;
    };

    class SBSersic::SBSersicImpl : public SBProfileImpl
    {
    public:
        SBSersicImpl(double n, double scale_radius, double flux, double trunc,
                     const GSParams& gsparams);

        ~SBSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            if (_trunc==0.) { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }
            else { xmin = -_trunc; xmax = _trunc; }
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
        {
            splits.push_back(0.);
            if (_trunc==0.) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else { ymin = -_trunc; ymax = _trunc; }
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const
        {
            if (_trunc==0.) { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }
            else if (std::abs(x) >= _trunc) { ymin = 0; ymax = 0; }
            else { ymax = sqrt(_trunc_sq - x*x);  ymin = -ymax; }

            if (std::abs(x/_re) < 1.e-2) splits.push_back(0.);
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return _trunc != 0.; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }
        double maxSB() const { return _xnorm; }

        /// @brief Sersic photon shooting done by rescaling photons from appropriate `SersicInfo`
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        /// @brief Returns the Sersic index n
        double getN() const { return _n; }
        /// @brief Returns the true half-light radius (may be different from the specified value)
        double getHalfLightRadius() const { return _re; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns the truncation radius
        double getTrunc() const { return _trunc; }

        // Overrides for better efficiency
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        double _n;       ///< Sersic index.
        double _flux;    ///< Actual flux (may differ from that specified at the constructor).
        double _r0;      ///< Scale radius specified at the constructor.
        double _re;      ///< Half-light radius specified at the constructor.
        double _trunc;   ///< The truncation radius (if any)

        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.
        double _shootnorm; ///< Normalization for photon shooting.

        double _r0_sq;
        double _inv_r0;
        double _inv_r0_sq;
        double _trunc_sq;

        shared_ptr<SersicInfo> _info; ///< Points to info structure for this n,trunc

        void doFillXImage(ImageView<double> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<double> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

        // Copy constructor and op= are undefined.
        SBSersicImpl(const SBSersicImpl& rhs);
        void operator=(const SBSersicImpl& rhs);

        static LRUCache<Tuple<double, double, GSParamsPtr>, SersicInfo> cache;

        friend class SBInclinedSersic;
        friend class SBInclinedSersic::SBInclinedSersicImpl;

    };
}

#endif
