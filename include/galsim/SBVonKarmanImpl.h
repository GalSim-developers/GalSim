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

#ifndef GalSim_SBVonKarmanImpl_H
#define GalSim_SBVonKarmanImpl_H

#include "SBProfileImpl.h"
#include "SBVonKarman.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"

namespace galsim {

    //
    //
    //
    //VonKarmanInfo
    //
    //
    //

    class VonKarmanInfo
    {
    public:
        VonKarmanInfo(double lam, double r0, double L0, bool doDelta, const GSParamsPtr& gsparams);

        ~VonKarmanInfo() {}

        double stepK() const { return _stepk; }
        double maxK() const { return _maxk; }
        double getDeltaAmplitude() const {return _deltaAmplitude; }
        double getHalfLightRadius() const {return _hlr; }

        double kValue(double) const;
        double xValue(double) const;
        double structureFunction(double rho) const;
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:
        VonKarmanInfo(const VonKarmanInfo& rhs); ///<Hide the copy constructor
        void operator=(const VonKarmanInfo& rhs); ///<Hide the assignment operator

        double kValueNoTrunc(double) const;

        double _lam; // Wavelength in meters
        double _r0; // Fried parameter in meters
        double _L0; // Outer scale in meters
        double _r0L0m53; // (r0/L0)^(-5/3)
        double _stepk;
        double _maxk;
        double _deltaAmplitude;
        bool _doDelta;
        double _hlr; // half-light-radius

        // Magic constants that we can compute once and store.
        const static double magic1; // 2 gamma(11/6) / (2^(5/6) pi^(8/3)) * (24/5 gamma(6/5))^(5/6)
        const static double magic2; // gamma(5/6) / 2^(1/6)
        const static double magic3; // magic1 * gamma(-5/6) / 2^(11/6)
        const static double magic4; // gamma(11/6) gamma(5/6) / pi^(8/3) * (24/5 gamma(6/5))^(5/6)
        const GSParamsPtr _gsparams;

        TableDD _radial;
        boost::shared_ptr<OneDimensionalDeviate> _sampler;

        void _buildRadialFunc();
    };

    //
    //
    //
    //SBVonKarmanImpl
    //
    //
    //

    class SBVonKarman::SBVonKarmanImpl : public SBProfileImpl
    {
    public:
        SBVonKarmanImpl(double lam, double r0, double L0, double flux, double scale, bool doDelta,
                        const GSParamsPtr& gsparams);
        ~SBVonKarmanImpl() {}

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;
        double getDeltaAmplitude() const;
        double getHalfLightRadius() const;

        Position<double> centroid() const { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getLam() const { return _lam; }
        double getR0() const { return _r0; }
        double getL0() const { return _L0; }
        double getScale() const { return _scale; }
        bool getDoDelta() const { return _doDelta; }
        double maxSB() const { return _flux * _info->xValue(0.); }

        /**
         * @brief SBVonKarman photon-shooting is done numerically with `OneDimensionalDeviate`
         * class.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double xValue(const Position<double>& p) const;
        double xValue(double r) const;
        std::complex<double> kValue(const Position<double>& p) const;
        double kValue(double k) const;

        double structureFunction(double rho) const;

        std::string serialize() const;

    private:

        double _lam;
        double _r0;
        double _L0;
        double _flux;
        double _scale;
        bool _doDelta;

        boost::shared_ptr<VonKarmanInfo> _info;

        // Copy constructor and op= are undefined.
        SBVonKarmanImpl(const SBVonKarmanImpl& rhs);
        void operator=(const SBVonKarmanImpl& rhs);

        static LRUCache<boost::tuple<double,double,double,bool,GSParamsPtr>,VonKarmanInfo> cache;
    };
}

#endif
