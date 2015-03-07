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

#ifndef GalSim_SBSpergeletImpl_H
#define GalSim_SBSpergeletImpl_H

#include "SBProfileImpl.h"
#include "SBSpergelet.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Spergel index `nu`.
    class SpergeletInfo
    {
    public:
        /// @brief Constructor
        SpergeletInfo(double nu, int j, int q, const GSParamsPtr& gsparams);

        /// @brief Destructor.
        ~SpergeletInfo() {}

        /**
         * @brief Returns the unnormalized value of the fourier transform.
         *
         * The input `ksq` should be (k_actual^2 * r0^2).
         */
        double kValue(double ksq, double phi) const;

        double maxK() const;
        double stepK() const;

    private:

        SpergeletInfo(const SpergeletInfo& rhs); ///< Hide the copy constructor.
        void operator=(const SpergeletInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _nu;       ///< Spergel index.
        int _j, _q;       ///< Spergelet indices.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Some derived values calculated in the constructor:
        double _gamma_nup1;
        double _gamma_nup2;
        double _gamma_nupjp1;
        double _knorm;

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.

        // Helper functions used internally:
        double calculateFluxRadius(const double& flux_frac) const;
    };

    class SBSpergelet::SBSpergeletImpl : public SBProfileImpl
    {
    public:
        SBSpergeletImpl(double nu, double r0, int j, int q, const GSParamsPtr& gsparams);

        ~SBSpergeletImpl() {}

        double xValue(const Position<double>& p) const
        { throw SBError("SBSpergelet::shoot() is not implemented"); }

        double getFlux() const // TODO: This is certainly not true
        { return 1.0; }

        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); } // TODO: Is this true?

        /// @brief Returns the Spergel index nu
        double getNu() const { return _nu; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns radial index
        int getJ() const {return _j;}
        /// @brief Returns azimuthal index
        int getQ() const {return _q;}

        /// @brief Photon-shooting is not implemented for SBSpergelet, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const
        { throw SBError("SBSpergelet::shoot() is not implemented"); }

        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        double _nu;    ///< Spergel index
        double _r0;    ///< Scale radius specified at the constructor.
        int _j, _q;    ///< Radial and azimuthal indices.

        double _xnorm; ///< Normalization of xValue relative to what SpergeletInfo returns.

        double _r0_sq;
        double _inv_r0;

        boost::shared_ptr<SpergeletInfo> _info; ///< Points to info structure for this nu, jq

        // Copy constructor and op= are undefined.
        SBSpergeletImpl(const SBSpergeletImpl& rhs);
        void operator=(const SBSpergeletImpl& rhs);

        static LRUCache<boost::tuple< double, int, int, GSParamsPtr >, SpergeletInfo> cache;
    };
}

#endif
