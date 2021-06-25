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

#ifndef GalSim_SBDeltaFunctionImpl_H
#define GalSim_SBDeltaFunctionImpl_H

#include "SBProfileImpl.h"
#include "SBDeltaFunction.h"

namespace galsim {

    // Not quite as high as std::numeric_limits<double>::max() == 1.8e308 so math with this
    // doesn't easily turn into inf.
    const double MOCK_INF = 1.e300;

    class SBDeltaFunction::SBDeltaFunctionImpl : public SBProfileImpl
    {
    public:
        SBDeltaFunctionImpl(double flux, const GSParams& gsparams);

        ~SBDeltaFunctionImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }
        double maxSB() const { return MOCK_INF; }

        /**
         * @brief Shoot photons through this SBDeltaFunction.
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

    private:
        double _flux;

        // Copy constructor and op= are undefined.
        SBDeltaFunctionImpl(const SBDeltaFunctionImpl& rhs);
        void operator=(const SBDeltaFunctionImpl& rhs);
    };
}

#endif
