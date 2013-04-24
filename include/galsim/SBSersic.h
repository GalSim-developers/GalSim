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

#ifndef SBSERSIC_H
#define SBSERSIC_H
/** 
 * @file SBSersic.h @brief SBProfile that implements a Sersic profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Sersic Surface Brightness Profile.
     *
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic index
     * `n`, its `flux` and the half-light radius `re`.  Given these properties, the surface
     * brightness profile scales as I(r) propto exp[-(r/r_0)^{1/n}].  Currently the code is limited
     * to 0.5<=n<=4.2, with an exception thrown for values outside that range.
     *
     * The SBProfile representation of a Sersic profile also includes an optional truncation beyond
     * a given radius, by the parameter `trunc`.  The resolution of the truncation radius in units
     * of half light radius `re` is limited to 2 decimal places, in order not to overload the Sersic
     * information caching.
     *
     * Another optional parameter, `flux_untruncated`, allows the setting of the flux to the
     * untruncated Sersic, while generating a truncated Sersic.  This facilitates the comparison
     * of truncated and untruncated Sersic, as both the amplitude and the scale parameter
     * `b=r_0^{-1/n}` change when a truncated Sersic is specified to the same flux as the
     * untruncated version with the same Sersic index `n`.  The `flux_untruncated` variable is
     * ignored if `trunc = 0`.
     *
     * Note that when `trunc > 0.` and `flux_untruncated == true`, the actual half-light radius will
     * be different from the specified half-light radius.  The getHalfLightRadius() method will
     * return the true half-light radius.  Similarly, the actual flux will not be the the same as
     * the specified value; the true flux is returned by the getFlux() method.
     *
     * There are several special cases of the Sersic profile that have their own SBProfiles: n=4
     * (SBDeVaucouleurs), n=1 (SBExponential), n=0.5 (SBGaussian).  These special cases use several
     * simplifications in all calculations, whereas for general n, the Fourier transform must be
     * treated numerically.
     */
    class SBSersic : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] n                 Sersic index.
         * @param[in] re                Half-light radius.
         * @param[in] trunc             Outer truncation radius in same physical units as size;
         *                              `trunc = 0.` for no truncation (default `trunc = 0.`).
         * @param[in] flux              Flux (default `flux = 1.`).
         * @param[in] flux_untruncated  If `true`, sets the flux to the untruncated version of the
         *                              Sersic profile with the same index `n` (default
         *                              flux_untruncated = false`).  Ignored if `trunc = 0.`.
         */
        SBSersic(double n, double re, double flux=1.,
                 double trunc=0., bool flux_untruncated=false,
                 boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>());

        /// @brief Copy constructor.
        SBSersic(const SBSersic& rhs);

        /// @brief Destructor.
        ~SBSersic();

        /// @brief Returns the Sersic index `n` of the profile.
        double getN() const;

        /// @brief Returns the half light radius of the Sersic profile.
        double getHalfLightRadius() const;

    protected:

        class SBSersicImpl;

    private:

        // op= is undefined
        void operator=(const SBSersic& rhs);
    };

    /**
     * @brief Surface Brightness for the de Vaucouleurs Profile, a special case of the Sersic with 
     * `n = 4`.
     */
    class SBDeVaucouleurs : public SBSersic 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] re                Half-light radius.
         * @param[in] flux              Flux (default `flux = 1.`).
         * @param[in] trunc             Outer truncation radius in same physical units as size;
         *                               `trunc = 0.` for no truncation (default `trunc = 0.`).
         * @param[in] flux_untruncated  If `true`, sets the flux to the untruncated version of the
         *                              Sersic profile with the same index `n` (default
         *                              flux_untruncated = false`).  Ignored if `trunc = 0.`.
         */
        SBDeVaucouleurs(double re, double flux=1.,
                        double trunc=0., bool flux_untruncated=false,
                        boost::shared_ptr<GSParams> gsparams = boost::shared_ptr<GSParams>()) :
            SBSersic(4., re, flux, trunc, flux_untruncated, gsparams) {}
    };

}

#endif // SBSERSIC_H

