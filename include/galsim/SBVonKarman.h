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

#ifndef GalSim_SBVonKarman_H
#define GalSim_SBVonKarman_H
/**
 * @file SBVonKarman.h @brief SBProfile for von Karman turbulence PSF.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {
        // How many VonKarman profiles to save in the cache
        const int max_vonKarman_cache = 100;
    }

    class PUBLIC_API SBVonKarman : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam          Wavelength in nm.
         * @param[in] r0           Fried parameter in m (at given wavelength lam).
         * @param[in] L0           Outer scale in m.
         * @param[in] flux         Flux.
         * @param[in] scale        Scale of 'x' in xValue in arcsec.
         * @param[in] doDelta      Whether or not to include delta-function contribution to
                                   encircled energy when computing stepk/maxk/HLR.
         * @param[in] gsparams     GSParams.
         */
        SBVonKarman(double lam, double r0, double L0, double flux,
                    double scale, bool doDelta, const GSParams& gsparams,
                    double force_stepk);

        /// @brief Copy constructor
        SBVonKarman(const SBVonKarman& rhs);

        /// @brief Destructor.
        ~SBVonKarman();

        /// Getters
        double getLam() const;
        double getR0() const;
        double getL0() const;
        double getScale() const;
        bool getDoDelta() const;
        double getDelta() const;
        double getHalfLightRadius() const;

        double structureFunction(double) const;

        friend class VKXIntegrand;

    protected:

        class SBVonKarmanImpl;

    private:
        // op= is undefined
        void operator=(const SBVonKarman& rhs);
    };
}

#endif
