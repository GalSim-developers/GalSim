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

#ifndef GalSim_SBLinearOpticaletImpl_H
#define GalSim_SBLinearOpticaletImpl_H

#include "SBProfileImpl.h"
#include "SBLinearOpticalet.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each set of radial and
    ///        azimuthal indices
    class LinearOpticaletInfo
    {
    public:
        /// @brief Constructor
        LinearOpticaletInfo(int n1, int m1, int n2, int m2, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~LinearOpticaletInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the LinearOpticalet function.
         *
         * The input `r` should be (r_actual / r0).
         * The returned value should then be divided by r0^2
         */
        double xValue(double r, double phi) const;

        /**
         * @brief Returns the value of the Fourier transform.
         *
         * The input `ksq` should be (k_actual * r0)**2.
         */
        std::complex<double> kValue(double ksq, double phi) const;

        double maxK() const;
        double stepK() const;
    private:

        LinearOpticaletInfo(const LinearOpticaletInfo& rhs); ///< Hide the copy constructor.
        void operator=(const LinearOpticaletInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        int _n1, _m1, _n2, _m2;      ///< LinearOpticalet indices.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.
        mutable double _xnorm;   ///< Real space normalization.

        // Parameters for the Hankel transform:
        mutable Table<double,double> _ftsum; ///< Lookup table for Hankel transform
        mutable Table<double,double> _ftdiff; ///< Lookup table for Hankel transform

        // Helper functions used internally:
        void buildFT() const;
        double calculateFluxRadius(const double& flux_frac) const;
        double Vnm(int n, int m, double r) const;
    };

    class SBLinearOpticalet::SBLinearOpticaletImpl : public SBProfileImpl
    {
    public:
        SBLinearOpticaletImpl(double r0, int n1, int m1, int n2, int m2,
                              const GSParamsPtr& gsparams);

        ~SBLinearOpticaletImpl() {}

        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const;
        // std::complex<double> kValue(const Position<double>& p) const
        // { throw SBError("SBSpergelet::kValue() is not implemented"); }

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); } // TODO: Is this true?

        double getFlux() const
        { return 1.0; }

        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns first radial index
        int getN1() const {return _n1;}
        /// @brief Returns first azimuthal index
        int getM1() const {return _m1;}
        /// @brief Returns second radial index
        int getN2() const {return _n2;}
        /// @brief Returns second azimuthal index
        int getM2() const {return _m2;}

        /// @brief Photon-shooting is not implemented for SBLinearOpticalet, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const
        { throw SBError("SBLinearOpticalet::shoot() is not implemented"); }

        // Overrides for better efficiency
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        double _r0;             ///< Scale radius specified at the constructor.
        int _n1, _m1, _n2, _m2; ///< Radial and azimuthal indices.

        double _xnorm; ///< Normalization of xValue relative to what LinearOpticalet returns.

        double _r0_sq;
        double _inv_r0;
        double _inv_r0_sq;

        boost::shared_ptr<LinearOpticaletInfo> _info; ///< Points to info structure

        // Copy constructor and op= are undefined.
        SBLinearOpticaletImpl(const SBLinearOpticaletImpl& rhs);
        void operator=(const SBLinearOpticaletImpl& rhs);

        static LRUCache<boost::tuple< int, int, int, int, GSParamsPtr >, LinearOpticaletInfo> cache;
    };
}

#endif
