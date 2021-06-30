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

#ifndef GalSim_SBSecondKick_H
#define GalSim_SBSecondKick_H
/**
 * @file SBSecondKick.h @brief SBProfile for second kick.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {
        // How many SecondKick profiles to save in the cache
        const int max_SK_cache = 100;
    }

    class PUBLIC_API SBSecondKick : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam_over_r0  lambda/r0, equivalent to the same in SBKolmogorov
         * @param[in] kcrit        Critical turbulence Fourier mode in units of r0.
         * @param[in] flux         Flux.
         * @param[in] gsparams     GSParams.
         */
        SBSecondKick(double lam_over_r0, double kcrit, double flux, const GSParamsPtr& gsparams);

        /// @brief Copy constructor
        SBSecondKick(const SBSecondKick& rhs);

        /// @brief Destructor.
        ~SBSecondKick();

        /// Getters
        double getLamOverR0() const;
        double getKCrit() const;
        double getDelta() const;
        /// Alternate versions of x/k Value for testing purposes
        double kValue(double) const;
        double kValueRaw(double) const;
        double xValue(double) const;
        double xValueRaw(double) const;
        double xValueExact(double) const;

        double structureFunction(double) const;

    protected:

        class SBSecondKickImpl;

    private:
        // op= is undefined
        void operator=(const SBSecondKick& rhs);
    };
}

#endif
