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
        VonKarmanInfo(double lam, double L0, bool doDelta, const GSParamsPtr& gsparams,
                      double force_stepk);

        ~VonKarmanInfo() {}

        double stepK() const {
            if(_stepk == 0.0) _buildRadialFunc();
            return _stepk;
        }
        double maxK() const { return _maxk; }
        double getDelta() const { return _delta; }
        double getHalfLightRadius() const {
            if (!_radial.finalized()) _buildRadialFunc();
            return _hlr;
        }

        double kValue(double) const;
        double xValue(double) const;
        double structureFunction(double rho) const;
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        double kValueNoTrunc(double) const;
        double rawXValue(double) const;

    private:
        VonKarmanInfo(const VonKarmanInfo& rhs); ///<Hide the copy constructor
        void operator=(const VonKarmanInfo& rhs); ///<Hide the assignment operator

        double _lam; // Wavelength in units of the Fried parameter, r0
        double _L0; // Outer scale in units of the Fried parameter, r0
        double _L0_invcuberoot;  // (r0/L0)^(1/3)
        double _L053; // (r0/L0)^(-5/3)
        mutable double _stepk;
        double _maxk;
        double _delta;
        double _deltaScale;  // 1/(1-_delta)
        double _lam_arcsec;  // _lam * ARCSEC2RAD / 2pi
        bool _doDelta;
        mutable double _hlr; // half-light-radius

        GSParamsPtr _gsparams;

        mutable TableBuilder _radial;
        mutable shared_ptr<OneDimensionalDeviate> _sampler;

        void _buildRadialFunc() const;
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
                        const GSParams& gsparams, double force_stepk);
        ~SBVonKarmanImpl() {}

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;
        double getDelta() const;
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
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        double xValue(const Position<double>& p) const;
        double xValue(double r) const;
        std::complex<double> kValue(const Position<double>& p) const;
        double kValue(double k) const;

        double structureFunction(double rho) const;

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

        double _lam;
        double _r0;
        double _L0;
        double _flux;
        double _scale;
        bool _doDelta;

        shared_ptr<VonKarmanInfo> _info;

        // Copy constructor and op= are undefined.
        SBVonKarmanImpl(const SBVonKarmanImpl& rhs);
        void operator=(const SBVonKarmanImpl& rhs);

        static LRUCache<Tuple<double,double,bool,GSParamsPtr,double>,VonKarmanInfo> cache;
    };

    double vkStructureFunction(double rho, double L0, double L0_invcuberoot, double L053);

}

#endif
