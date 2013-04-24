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

#ifndef SBEXPONENTIAL_H
#define SBEXPONENTIAL_H
/** 
 * @file SBExponential.h @brief SBProfile that implements a 2-d exponential profile.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Exponential Surface Brightness Profile.  
     *
     * Surface brightness profile with I(r) propto exp[-r/r_0] for some scale-length r_0.  This is a
     * special case of the Sersic profile, but is given a separate class since the Fourier transform
     * has closed form and can be generated without lookup tables.
     */
    class SBExponential : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in 
         * SBSersic.
         *
         * @param[in] r0    scale length for the profile that scales as `exp[-(r / r0)]`, NOT the 
         *                  half-light radius `re`.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBExponential(double r0, double flux=1.,
                      boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBExponential(const SBExponential& rhs);

        /// @brief Destructor.
        ~SBExponential();

        /// @brief Returns the scale radius of the Exponential profile.
        double getScaleRadius() const;

    protected:

        class SBExponentialImpl;

    private:
        // op= is undefined
        void operator=(const SBExponential& rhs);
    };

}

#endif // SBEXPONENTIAL_H

