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

#ifndef SBAIRY_H
#define SBAIRY_H
/** 
 * @file SBAiry.h @brief SBProfile of an Airy function with an optional obscuration.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // How many Airy profiles to save in the cache
        const int max_airy_cache = 100;

    }

    /** 
     * @brief Surface Brightness Profile for the Airy disk (perfect diffraction-limited PSF for a 
     * circular aperture), with central obscuration.
     *
     * maxK() is set at the hard limit for Airy disks, stepK() makes transforms go to at least 
     * 5 lam/D or EE>(1-alias_threshold).  Schroeder (10.1.18) gives limit of EE at large radius.
     * This stepK could probably be relaxed, it makes overly accurate FFTs.
     * Note x & y are in units of lambda/D here.  Integral over area will give unity in this 
     * normalization.
     */
    class SBAiry : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam_over_D   `lam_over_D` = (lambda * focal length) / (telescope diam) if 
         *                         arg is focal plane position, else `lam_over_D` = 
         *                         lambda / (telescope diam) if arg is in radians of field angle.
         * @param[in] obscuration  linear dimension of central obscuration as fraction of pupil
         *                         dimension.
         * @param[in] flux         flux.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         image operations and rendering, if different from the default.
         */
        SBAiry(double lam_over_D, double obscuration, double flux, const GSParamsPtr& gsparams);

        /// @brief Copy constructor
        SBAiry(const SBAiry& rhs);

        /// @brief Destructor.
        ~SBAiry();

        /// @brief Returns lam_over_D param of the SBAiry.
        double getLamOverD() const;

        /// @brief Returns obscuration param of the SBAiry.
        double getObscuration() const;

    protected:

        class SBAiryImpl;

    private:
        // op= is undefined
        void operator=(const SBAiry& rhs);
    };

}

#endif // SBAIRY_H

