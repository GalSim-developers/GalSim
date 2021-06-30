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

#ifndef GalSim_SBFourierSqrt_H
#define GalSim_SBFourierSqrt_H
/**
 * @file SBFourierSqrt.h @brief SBProfile adapter which computes the square root of its subject in k space.
 */


#include "SBProfile.h"

namespace galsim {

    /**
     * @brief SBProfile adapter which computes the square root of its subject in k space.
     *
     * @param[in] adaptee   SBProfile to compute the Fourier-space square root of.
     * @param[in] gsparams  GSParams object storing constants that control the accuracy of
     *                      image operations and rendering, if different from the default.
     */
    class PUBLIC_API SBFourierSqrt : public SBProfile
    {
    public:
        /// @brief Constructor.
        SBFourierSqrt(const SBProfile& adaptee, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBFourierSqrt(const SBFourierSqrt& rhs);

        /// @brief Get the SBProfile being operated on
        SBProfile getObj() const;

        /// @brief Destructor.
        ~SBFourierSqrt();

    protected:

        class SBFourierSqrtImpl;

    private:
        // op= is undefined
        void operator=(const SBFourierSqrt& rhs);
    };

}

#endif
