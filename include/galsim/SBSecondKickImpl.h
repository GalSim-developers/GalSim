/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBSecondKickImpl_H
#define GalSim_SBSecondKickImpl_H

#include "SBProfileImpl.h"
#include "SBSecondKick.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"
#include "SBAiryImpl.h"
#include <boost/move/unique_ptr.hpp>

namespace galsim {

    //
    //
    //
    //SKInfo
    //
    //
    //

    class SKInfo
    {
    public:
        SKInfo(double lam, double r0, double diam, double obscuration, double L0, double kcrit,
               const GSParamsPtr& gsparams);
        ~SKInfo() {}

        double stepK() const { return _stepk; }
        double maxK() const { return _maxk; }
        double getDelta() const { return _delta; }

        double kValue(double) const;
        double kValueRaw(double) const;
        double xValue(double) const;
        double xValueRaw(double) const;
        double xValueExact(double) const;
        double structureFunction(double rho) const;
        double structureFunction2(double rho) const;
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:
        SKInfo(const SKInfo& rhs); ///<Hide the copy constructor
        void operator=(const SKInfo& rhs); ///<Hide the assignment operator

        double _lam; // Wavelength in meters
        double _lam_arcsec; // lam * ARCSEC2RAD / 2pi
        double _r0; // Fried parameter in meters
        double _lam_over_r0; // Wavelength in meters
        double _diam; // in meters
        double _obscuration; // linear fractional circular obscuration
        double _L0; // // Outer scale in units of the Fried parameter, r0.  I.e., L0/r0.
        double _L0_invcuberoot;  // (r0/L0)^(1/3)
        double _L0invsq; // (r0/L0)^2
        double _L053; // (L0/r0)^(5/3)
        double _kcrit;
        double _stepk;
        double _maxk;
        double _knorm;
        double _4_over_diamsq;
        double _delta;

        const GSParamsPtr _gsparams;

        boost::movelib::unique_ptr<AiryInfo> _airy_info;

        TableDD _radial;
        TableDD _kvLUT;
        boost::shared_ptr<OneDimensionalDeviate> _sampler;

        void _buildRadial();
        void _buildKVLUT();
    };

    //
    //
    //
    //SBSecondKickImpl
    //
    //
    //

    class SBSecondKick::SBSecondKickImpl : public SBProfileImpl
    {
    public:
        SBSecondKickImpl(double lam, double r0, double diam, double obscuration, double L0,
                         double kcrit, double flux, double scale, const GSParamsPtr& gsparams);
        ~SBSecondKickImpl() {}

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;
        double getHalfLightRadius() const;
        double getDelta() const;

        Position<double> centroid() const { return Position<double>(0., 0.); }

        double getFlux() const { return _flux-getDelta(); }
        double getLam() const { return _lam; }
        double getR0() const { return _r0; }
        double getDiam() const { return _diam; }
        double getObscuration() const { return _obscuration; }
        double getL0() const { return _L0; }
        double getKCrit() const { return _kcrit; }
        double getScale() const { return _scale; }
        double maxSB() const { return _flux * _info->xValue(0.); }

        /**
         * @brief SBSecondKick photon-shooting is done numerically with `OneDimensionalDeviate`
         * class.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double xValue(const Position<double>& p) const;
        double xValue(double r) const;
        double xValueRaw(double k) const;
        double xValueExact(double k) const;
        std::complex<double> kValue(const Position<double>& p) const;
        double kValue(double k) const;
        double kValueRaw(double k) const;

        double structureFunction(double rho) const;

        std::string serialize() const;

    private:

        double _lam;
        double _r0;
        double _diam;
        double _obscuration;
        double _L0;
        double _kcrit;
        double _flux;
        double _scale;

        boost::shared_ptr<SKInfo> _info;

        // Copy constructor and op= are undefined.
        SBSecondKickImpl(const SBSecondKickImpl& rhs);
        void operator=(const SBSecondKickImpl& rhs);

        static LRUCache<boost::tuple<double,double,double,double,double,double,GSParamsPtr>,SKInfo>
            cache;
    };
}

#endif
