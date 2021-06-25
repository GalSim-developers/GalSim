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

#ifndef GalSim_SBSpergelImpl_H
#define GalSim_SBSpergelImpl_H

#include "SBProfileImpl.h"
#include "SBSpergel.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Spergel index `nu`.
    class SpergelInfo
    {
    public:
        /// @brief Constructor
        SpergelInfo(double nu, const GSParamsPtr& gsparams);

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

        /**
         * @brief The factor by which to multiply the returned value from xValue.
         *
         * Since the returned value needs to be multiplied by flux/r0^2 anyway, we also let
         * the caller of xValue multiply by the normalization, which we calculate for them here.
         */
        double getXNorm() const;

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Spergel profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        double calculateIntegratedFlux(double r) const;
        double calculateFluxRadius(double f) const;

    private:

        SpergelInfo(const SpergelInfo& rhs); ///< Hide the copy constructor.
        void operator=(const SpergelInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _nu;       ///< Spergel index.
        GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _gamma_nup1;  ///< Gamma(nu+1)
        double _gamma_nup2;  ///< Gamma(nu+2)
        double _xnorm0   ;   ///< Normalization at r=0 for nu>0

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _re;      ///< The HLR in units of r0.

        // Classes used for photon shooting
        mutable shared_ptr<FluxDensity> _radial;
        mutable shared_ptr<OneDimensionalDeviate> _sampler;
    };

    class SBSpergel::SBSpergelImpl : public SBProfileImpl
    {
    public:
        SBSpergelImpl(double nu, double scale_radius,
                      double flux, const GSParams& gsparams);

        ~SBSpergelImpl() {}

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
            ymin = -integ::MOCK_INF;
            ymax = integ::MOCK_INF;

            if (std::abs(x/_r0) < 1.e-2) splits.push_back(0.);
        }

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double maxSB() const { return std::abs(_xnorm) * _info->xValue(0.); }

        /// @brief Spergel photon shooting done by rescaling photons from appropriate `SpergelInfo`
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        /// @brief Returns the Spergel index nu
        double getNu() const { return _nu; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns enclosed flux
        double calculateIntegratedFlux(double r) const;
        /// @brief Return flux-enclosing-radius
        double calculateFluxRadius(double f) const;

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
        double _nu;      ///< Spergel index
        double _flux;    ///< Flux
        double _r0;      ///< Scale radius specified at the constructor.

        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.
        double _shootnorm; ///< Normalization for photon shooting.

        double _r0_sq;
        double _inv_r0;
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _k_max;   ///< sqrt(_ksq_max)

        shared_ptr<SpergelInfo> _info; ///< Points to info structure for this nu

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
        SBSpergelImpl(const SBSpergelImpl& rhs);
        void operator=(const SBSpergelImpl& rhs);

        static LRUCache<Tuple<double, GSParamsPtr>, SpergelInfo> cache;
    };
}

#endif
