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

#ifndef GalSim_SBMoffatletImpl_H
#define GalSim_SBMoffatletImpl_H

#include "SBProfileImpl.h"
#include "SBMoffatlet.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Moffat index `beta`.
    class MoffatletInfo
    {
    public:
        /// @brief Constructor
        MoffatletInfo(double beta, int j, int q, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~MoffatletInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the Moffatlet function.
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
        double kValue(double ksq, double phi) const;

        double maxK() const;
        double stepK() const;

        double getHalfLightRadius() const;

    private:

        MoffatletInfo(const MoffatletInfo& rhs); ///< Hide the copy constructor.
        void operator=(const MoffatletInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        double _beta;       ///< Moffat index.
        double _r0;         ///< scale radius
        int _j, _q;         ///< Moffatlet indices.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.

        // Parameters for the Hankel transform:
        mutable Table<double,double> _ft; ///< Lookup table for Fourier transform

        // Helper functions used internally:
        void buildFT() const;
        double calculateFluxRadius(const double& flux_frac) const;
    };

    class SBMoffatlet::SBMoffatletImpl : public SBProfileImpl
    {
    public:
        SBMoffatletImpl(double beta, double r0, int j, int q, const GSParamsPtr& gsparams);

        ~SBMoffatletImpl() {}

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

        /// @brief Returns the Moffat index beta
        double getBeta() const { return _beta; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns radial index
        int getJ() const {return _j;}
        /// @brief Returns azimuthal index
        int getQ() const {return _q;}

        /// @brief Photon-shooting is not implemented for SBMoffatlet, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const
        { throw SBError("SBMoffatlet::shoot() is not implemented"); }

        // Overrides for better efficiency
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        // void fillKValue(tmv::MatrixView<std::complex<double> > val,
        //                 double kx0, double dkx, int izero,
        //                 double ky0, double dky, int jzero) const;
        // void fillKValue(tmv::MatrixView<std::complex<double> > val,
        //                 double kx0, double dkx, double dkxy,
        //                 double ky0, double dky, double dkyx) const;

    private:
        double _beta;  ///< Moffat index
        double _r0;    ///< Scale radius specified at the constructor.
        int _j, _q;    ///< Radial and azimuthal indices.

        double _xnorm; ///< Normalization of xValue relative to what SersicInfo returns.

        double _r0_sq;
        double _inv_r0;

        boost::shared_ptr<MoffatletInfo> _info; ///< Points to info structure for this beta, jq

        // Copy constructor and op= are undefined.
        SBMoffatletImpl(const SBMoffatletImpl& rhs);
        void operator=(const SBMoffatletImpl& rhs);

        static LRUCache<boost::tuple< double, int, int, GSParamsPtr >, MoffatletInfo> cache;
    };
}

#endif
