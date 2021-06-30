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

#ifndef GalSim_SBGaussian_H
#define GalSim_SBGaussian_H
/**
 * @file SBGaussian.h @brief SBProfile that implements a 2-d Gaussian profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Gaussian Surface Brightness Profile
     *
     * The Gaussian Surface Brightness Profile is characterized by two properties, its `flux`
     * and the characteristic size `sigma` where the radial profile of the circular Gaussian
     * drops off as `exp[-r^2 / (2. * sigma^2)]`.
     * The maxK() and stepK() are for the SBGaussian are chosen to extend to 4 sigma in both
     * real and k domains, or more if needed to reach the `folding_threshold` spec.
     */
    class PUBLIC_API SBGaussian : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] sigma    characteristic size, surface brightness scales as
         *                     `exp[-r^2 / (2. * sigma^2)]`.
         * @param[in] flux     flux of the Surface Brightness Profile.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBGaussian(double sigma, double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBGaussian(const SBGaussian& rhs);

        /// @brief Destructor.
        ~SBGaussian();

        /// @brief Returns the characteristic size sigma of the Gaussian profile.
        double getSigma() const;

    protected:

        class SBGaussianImpl;

    private:
        // op= is undefined
        void operator=(const SBGaussian& rhs);
    };

}

#endif

