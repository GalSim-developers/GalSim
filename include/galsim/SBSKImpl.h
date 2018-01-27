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

#ifndef GalSim_SBSKImpl_H
#define GalSim_SBSKImpl_H

#include "SBProfileImpl.h"
#include "SBSK.h"
#include "LRUCache.h"
#include "OneDimensionalDeviate.h"
#include "Table.h"
#include "SBAiry.h"

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
        double getHalfLightRadius() const {return _hlr; }

        double kValue(double) const;
        double kValueSlow(double) const;
        double xValue(double) const;
        double xValueSlow(double) const;
        double vkStructureFunction(double rho) const;
        double structureFunction(double rho) const;
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

    private:
        SKInfo(const SKInfo& rhs); ///<Hide the copy constructor
        void operator=(const SKInfo& rhs); ///<Hide the assignment operator

        double _lam; // Wavelength in meters
        double _r0; // Fried parameter in meters
        double _r0m53; // r0^(-5/3)
        double _diam;
        double _obscuration;
        double _L0; // Outer scale in meters
        double _L0invsq; // 1/L0/L0
        double _r0L0m53;
        double _kcrit;
        double _stepk;
        double _maxk;
        double _hlr; // half-light-radius

        // Magic constants that we can compute once and store.
        const static double magic1;
        const static double magic2;
        const static double magic3;
        const static double magic5; // 2 gamma(11/6)^2 / pi^(8/3) (24/5 gamma(6/5))^(5/6)
        const GSParamsPtr _gsparams;

        boost::shared_ptr<SBAiry> _airy;

        TableDD _sfLUT;
        TableDD _radial;
        boost::shared_ptr<OneDimensionalDeviate> _sampler;

        void _buildSFLUT();
        void _buildRadial();
    };

    //
    //
    //
    //SBSKImpl
    //
    //
    //

    class SBSK::SBSKImpl : public SBProfileImpl
    {
    public:
        SBSKImpl(double lam, double r0, double diam, double obscuration, double L0, double kcrit,
                 double flux, double scale, const GSParamsPtr& gsparams);
        ~SBSKImpl() {}

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;
        double getHalfLightRadius() const;

        Position<double> centroid() const { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double getLam() const { return _lam; }
        double getR0() const { return _r0; }
        double getDiam() const { return _diam; }
        double getObscuration() const { return _obscuration; }
        double getL0() const { return _L0; }
        double getKCrit() const { return _kcrit; }
        double getScale() const { return _scale; }
        double maxSB() const { return _flux * _info->xValue(0.); }

        /**
         * @brief SBSK photon-shooting is done numerically with `OneDimensionalDeviate`
         * class.
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double xValue(const Position<double>& p) const;
        double xValue(double r) const;
        double xValueSlow(double k) const;
        std::complex<double> kValue(const Position<double>& p) const;
        double kValue(double k) const;
        double kValueSlow(double k) const;

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
        SBSKImpl(const SBSKImpl& rhs);
        void operator=(const SBSKImpl& rhs);

        static LRUCache<boost::tuple<double,double,double,double,double,double,GSParamsPtr>,SKInfo> cache;
    };
}

#endif
