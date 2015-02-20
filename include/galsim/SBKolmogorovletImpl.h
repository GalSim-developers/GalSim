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

#ifndef GalSim_SBKolmogorovletImpl_H
#define GalSim_SBKolmogorovletImpl_H

#include "SBProfileImpl.h"
#include "SBKolmogorovlet.h"
#include "LRUCache.h"

namespace galsim {

    /// @brief A private class that caches the needed parameters for each Spergel index `nu`.
    class KolmogorovletInfo
    {
    public:
        /// @brief Constructor
        KolmogorovletInfo(int j, int q, const GSParamsPtr& gsparams);

        /// @brief Destructor: deletes photon-shooting classes if necessary
        ~KolmogorovletInfo() {}

        /**
         * @brief Returns the unnormalized real space value of the Kolmogorovlet function.
         *
         * The input `r` should be (r_actual / r0).
         * The returned value should then be multiplied by flux * getXNorm() / r0^2
         */
        //double xValue(double r, double phi) const;

        /**
         * @brief Returns the unnormalized value of the fourier transform.
         *
         * The input `ksq` should be (k_actual^2 * r0^2).
         * The returned value should then be multiplied by flux.
         */
        double kValue(double ksq, double phi) const;

        double maxK() const;
        double stepK() const;

        /**
         * @brief The factor by which to multiply the returned value from xValue.
         *
         * Since the returned value needs to be multiplied by flux/r0^2 anyway, we also let
         * the caller of xValue multiply by the normalization, which we calculate for them here.
         */
        double getXNorm() const;

    private:

        KolmogorovletInfo(const KolmogorovletInfo& rhs); ///< Hide the copy constructor.
        void operator=(const KolmogorovletInfo& rhs); ///<Hide assignment operator.

        // Input variables:
        int _j, _q;       ///< Kolmogorovlet indices.
        const GSParamsPtr _gsparams; ///< The GSParams object.

        // Parameters calculated when they are first needed, and then stored:
        mutable double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        mutable double _stepk;   ///< Sampling in k space necessary to avoid folding.

        // Helper functions used internally:
        double calculateFluxRadius(const double& flux_frac) const;
    };

    class SBKolmogorovlet::SBKolmogorovletImpl : public SBProfileImpl
    {
    public:
        SBKolmogorovletImpl(double r0, int j, int q, const GSParamsPtr& gsparams);

        ~SBKolmogorovletImpl() {}

        // @brief xValue (and in general real-space operations) are not implemented for
        // SBKolmogorovlet, will throw an exception.
        double xValue(const Position<double>& p) const
        { throw SBError("SBKolmogorovlet::xValue() is not implemented"); }

        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const
        { return Position<double>(0., 0.); } // TODO: Is this true?

        double getFlux() const;

        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns radial index
        int getJ() const {return _j;}
        /// @brief Returns azimuthal index
        int getQ() const {return _q;}

        /// @brief Photon-shooting is not implemented for SBKolmogorovlet, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const
        { throw SBError("SBKolmogorovlet::shoot() is not implemented"); }

        // Overrides for better efficiency
        // void fillXValue(tmv::MatrixView<double> val,
        //                 double x0, double dx, int ix_zero,
        //                 double y0, double dy, int iy_zero) const;
        // void fillXValue(tmv::MatrixView<double> val,
        //                 double x0, double dx, double dxy,
        //                 double y0, double dy, double dyx) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _r0;    ///< Scale radius specified at the constructor.
        int _j, _q;    ///< Radial and azimuthal indices.

        double _xnorm; ///< Normalization of xValue relative to what SersicInfo returns.

        double _r0_sq;
        double _inv_r0;

        boost::shared_ptr<KolmogorovletInfo> _info; ///< Points to info structure for this nu, jq

        // Copy constructor and op= are undefined.
        SBKolmogorovletImpl(const SBKolmogorovletImpl& rhs);
        void operator=(const SBKolmogorovletImpl& rhs);

        static LRUCache<boost::tuple< int, int, GSParamsPtr >, KolmogorovletInfo> cache;
    };
}

#endif
