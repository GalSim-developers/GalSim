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

#ifndef SBMOFFAT_H
#define SBMOFFAT_H
/** 
 * @file SBMoffat.h @brief SBProfile that implements a Moffat profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Surface Brightness for the Moffat Profile (an approximate description of ground-based
     * PSFs).
     *
     * The Moffat surface brightness profile is I(R) propto [1 + (r/r_scale)^2]^(-beta).  The
     * SBProfile representation of a Moffat profile also includes an optional truncation beyond a
     * given radius.
     */
    class SBMoffat : public SBProfile 
    {
    public:
        enum  RadiusType
        {
            FWHM,
            HALF_LIGHT_RADIUS,
            SCALE_RADIUS
        };

        /** @brief Constructor.
         *
         * @param[in] beta           Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
         * @param[in] size           Size specification.
         * @param[in] rType          Kind of size being specified (one of FWHM, HALF_LIGHT_RADIUS,
         *                           SCALE_RADIUS).
         * @param[in] trunc          Outer truncation radius in same physical units as size,
         *                           trunc = 0. for no truncation (default `trunc = 0.`). 
         * @param[in] flux           Flux (default `flux = 1.`).
         */
        SBMoffat(double beta, double size, RadiusType rType, double trunc=0., double flux=1.,
                 boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());


        /// @brief Copy constructor.
        SBMoffat(const SBMoffat& rhs);

        /// @brief Destructor.
        ~SBMoffat();

        /// @brief Returns beta of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getBeta() const;

        /// @brief Returns the FWHM of the Moffat profile.
        double getFWHM() const;

        /// @brief Returns the scale radius rD of the Moffat profile `[1 + (r / rD)^2]^beta`.
        double getScaleRadius() const;

        /// @brief Returns the half light radius of the Moffat profile.
        double getHalfLightRadius() const;

    protected:

        class SBMoffatImpl;

    private:
        // op= is undefined
        void operator=(const SBMoffat& rhs);
    };

}

#endif // SBMOFFAT_H

