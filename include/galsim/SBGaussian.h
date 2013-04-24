// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

#ifndef SBGAUSSIAN_H
#define SBGAUSSIAN_H
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
     * real and k domains, or more if needed to reach the `alias_threshold` spec.
     */
    class SBGaussian : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] sigma  characteristic size, surface brightness scales as 
         *                   `exp[-r^2 / (2. * sigma^2)]`.
         * @param[in] flux   flux of the Surface Brightness Profile (default `flux = 1.`).
         */
        SBGaussian(double sigma, double flux=1.,
                   boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

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

#endif // SBGAUSSIAN_H

