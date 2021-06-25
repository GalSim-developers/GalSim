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

#ifndef GalSim_SBExponentialImpl_H
#define GalSim_SBExponentialImpl_H

#include "SBProfileImpl.h"
#include "SBExponential.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"

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
        ExponentialInfo(const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~ExponentialInfo() {}

        /**
         * @brief Shoot photons through unit-size, unnormalized profile
         * Sersic profiles are sampled with a numerical method, using class
         * `OneDimensionalDeviate`.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        double maxK() const;
        double stepK() const;

    private:

        ExponentialInfo(const ExponentialInfo& rhs); ///< Hides the copy constructor.
        void operator=(const ExponentialInfo& rhs); ///<Hide assignment operator.

        /// Function class used for photon shooting
        shared_ptr<ExponentialRadialFunction> _radial;

        /// Class that does numerical photon shooting
        shared_ptr<OneDimensionalDeviate> _sampler;

        double _maxk; ///< Calculated maxK * r0
        double _stepk; ///< Calculated stepK * r0
    };

    class SBExponential::SBExponentialImpl : public SBProfileImpl
    {
    public:

        SBExponentialImpl(double r0, double flux, const GSParams& gsparams);

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
        double maxSB() const { return _norm; }

        void shoot(PhotonArray& photons, UniformDeviate ud) const;

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
        double _flux; ///< Flux.
        double _r0;   ///< Characteristic size of profile `exp[-(r / r0)]`.
        double _r0_sq;
        double _inv_r0;
        double _inv_r0_sq;
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _k_max;   ///< sqrt(_ksq_max)
        double _norm; ///< flux / r0^2 / 2pi
        double _flux_over_2pi; ///< Flux / 2pi

        const shared_ptr<ExponentialInfo> _info;

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
        SBExponentialImpl(const SBExponentialImpl& rhs);
        void operator=(const SBExponentialImpl& rhs);

        static LRUCache<GSParamsPtr, ExponentialInfo> cache;
    };

}

#endif
