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

#ifndef GalSim_SBSK_H
#define GalSim_SBSK_H
/**
 * @file SBSK.h @brief SBProfile for second kick.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {
        // How many SK profiles to save in the cache
        const int max_SK_cache = 100;
    }

    class SBSK : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam          Wavelength in nm.
         * @param[in] r0           Fried parameter in m (at given wavelength lam).
         * @param[in] diam         Telescope diameter in m.
         * @param[in] obscuration  Fractional linear obscuration.
         * @param[in] L0           Outer scale in m.
         * @param[in] kcrit        Critical turbulence Fourier mode.
         * @param[in] flux         Flux.
         * @param[in] scale        Scale of 'x' in xValue in arcsec.
         * @param[in] gsparams     GSParams.
         */
        SBSK(double lam, double r0, double diam, double obscuration, double L0, double kcrit,
             double flux, double scale, const GSParamsPtr& gsparams);

        /// @brief Copy constructor
        SBSK(const SBSK& rhs);

        /// @brief Destructor.
        ~SBSK();

        /// Getters
        double getLam() const;
        double getR0() const;
        double getDiam() const;
        double getObscuration() const;
        double getL0() const;
        double getKCrit() const;
        double getScale() const;
        double getHalfLightRadius() const;
        /// Alternate versions of x/k Value for testing purposes
        double kValueSlow(double) const;
        double xValueSlow(double) const;

        double structureFunction(double) const;

        friend class SKXIntegrand;

    protected:

        class SBSKImpl;

    private:
        // op= is undefined
        void operator=(const SBSK& rhs);
    };
}

#endif
