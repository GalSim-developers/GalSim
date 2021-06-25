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

#ifndef GalSim_SBInclinedSersicImpl_H
#define GalSim_SBInclinedSersicImpl_H

#include "SBProfileImpl.h"
#include "SBInclinedSersic.h"
#include "SBSersicImpl.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"

namespace galsim {

    class SBInclinedSersic::SBInclinedSersicImpl : public SBProfileImpl
    {
    public:
        SBInclinedSersicImpl(double n, double inclination, double scale_radius,
                double height, double flux, double trunc, const GSParams& gsparams);

        ~SBInclinedSersicImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; } // Actually true, and might need to be changed so if made analytic in real-space,
                                                    // depending on tests of if it's more efficient/accurate
        bool isAnalyticX() const { return false; } // not yet implemented, would require lookup table
        bool isAnalyticK() const { return true; }  // 1d lookup table

        Position<double> centroid() const
        { return Position<double>(0., 0.); }

        /// @brief Returns the true flux (may be different from the specified flux)
        double getFlux() const { return _flux; }

        /// @brief Maximum surface brightness
        double maxSB() const;

        /// @brief photon shooting is not yet implemented
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        /// @brief Returns the Sersic index n
        double getN() const { return _n; }
        /// @brief Returns the inclination angle
        double getInclination() const { return _inclination; }
        /// @brief Returns the true half-light radius (may be different from the specified value)
        double getHalfLightRadius() const { return _re; }
        /// @brief Returns the scale radius
        double getScaleRadius() const { return _r0; }
        /// @brief Returns the scale height
        double getScaleHeight() const { return _h0; }
        /// @brief Returns the truncation radius
        double getTrunc() const { return _trunc; }

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
        double _n;       ///< Sersic index.
        double _inclination; ///< Inclination angle
        double _flux;    ///< Actual flux (may differ from that specified at the constructor).
        double _r0;      ///< Scale radius specified at the constructor.
        double _re;      ///< Half-light radius specified at the constructor.
        double _h0;          ///< Scale height specified at the constructor.
        double _trunc;   ///< The truncation radius (if any)

        double _xnorm;     ///< Normalization of xValue relative to what SersicInfo returns.

        double _inv_r0;
        double _half_pi_h_sini_over_r;
        double _cosi;
        double _r0_sq;
        double _inv_r0_sq;
        double _trunc_sq;

        // Some derived values calculated in the constructor:
        double _ksq_max;   ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_min;   ///< If ksq > _kq_max, then use kvalue = 0
        double _maxk;    ///< Value of k beyond which aliasing can be neglected.
        double _stepk;   ///< Sampling in k space necessary to avoid folding.

        shared_ptr<SersicInfo> _info; ///< Points to info structure for this n,trunc

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
        SBInclinedSersicImpl(const SBInclinedSersicImpl& rhs);
        void operator=(const SBInclinedSersicImpl& rhs);

        // Helper function to get k values
        double kValueHelper(double kx, double ky) const;

        // Helper functor to solve for the proper _maxk
        class SBInclinedSersicKValueFunctor;

        friend class SBInclinedSersicKValueFunctor;

    };
}

#endif
