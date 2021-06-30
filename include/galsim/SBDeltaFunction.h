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

#ifndef GalSim_SBDeltaFunction_H
#define GalSim_SBDeltaFunction_H
/**
 * @file SBGDeltaFunction.h @brief SBProfile that implements a Delta Function profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Delta Function Surface Brightness Profile
     *
     * The Delta Function Surface Brightness Profile is characterized
     * by a single property, its 'flux'.
     *
     * Note that the DeltaFunction SBP cannot be drawn by itself. Instead,
     * it should be applied as part of a convolution first.
     */
    class PUBLIC_API SBDeltaFunction : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] flux     flux of the Surface Brightness Profile.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBDeltaFunction(double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBDeltaFunction(const SBDeltaFunction& rhs);

        /// @brief Destructor.
        ~SBDeltaFunction();

    protected:
        class SBDeltaFunctionImpl;

    private:
        // op= is undefined
        void operator=(const SBDeltaFunction& rhs);
    };

}

#endif
