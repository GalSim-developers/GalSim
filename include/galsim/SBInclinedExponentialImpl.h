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

#ifndef GalSim_SBInclinedExponentialImpl_H
#define GalSim_SBInclinedExponentialImpl_H

#include "SBProfileImpl.h"
#include "SBInclinedExponential.h"

namespace galsim {

    class SBInclinedExponential::SBInclinedExponentialImpl : public SBProfileImpl
    {
    public:
        SBInclinedExponentialImpl(double inclination, double scale_radius, double scale_height,
                                  double flux, const GSParams& gsparams);

        ~SBInclinedExponentialImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& splits) const
        {
            xmin = -integ::MOCK_INF;
            xmax = integ::MOCK_INF;
        }

        void getYRange(double& ymin, double& ymax, std::vector<double>& splits) const
        {
            ymin = -integ::MOCK_INF;
            ymax = integ::MOCK_INF;
        }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& splits) const
        {
            ymin = -integ::MOCK_INF;
            ymax = integ::MOCK_INF;
        }

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }  // May be in future version though
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }

        /// @brief Maximum surface brightness
        double maxSB() const;

        /// @brief photon shooting is not implemented yet.
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        /// @brief Returns the inclination angle in radians
        double getInclination() const { return _inclination; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns the scale height
        double getScaleHeight() const { return _h0; }

        // Overrides for better efficiency
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        double _inclination; ///< Inclination angle
        double _r0;          ///< Scale radius specified at the constructor.
        double _h0;          ///< Scale height specified at the constructor.
        double _flux;        ///< Actual flux (may differ from that specified at the constructor).

        double _inv_r0;
        double _half_pi_h_sini_over_r;
        double _cosi;

        // Some derived values calculated in the constructor:
        double _ksq_max;   ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_min;   ///< If ksq > _kq_max, then use kvalue = 0
        double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        double _stepk;   ///< Sampling in k space necessary to avoid folding.

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
        SBInclinedExponentialImpl(const SBInclinedExponentialImpl& rhs);
        void operator=(const SBInclinedExponentialImpl& rhs);

        // Helper function to get k values
        double kValueHelper(double kx, double ky) const;

        // Helper functor to solve for the proper _maxk
        class SBInclinedExponentialKValueFunctor
        {
            public:
                SBInclinedExponentialKValueFunctor(const SBInclinedExponential::SBInclinedExponentialImpl * p_owner,
            double target_k_value);
            double operator() (double k) const;
            private:
            const SBInclinedExponential::SBInclinedExponentialImpl * _p_owner;
            double _target_k_value;
        };

        friend class SBInclinedExponentialKValueFunctor;
    };
}

#endif
