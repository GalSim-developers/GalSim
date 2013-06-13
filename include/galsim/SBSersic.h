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

    namespace sbp {

        // Constrain range of allowed Sersic index n to those for which testing was done
        // Note: If these change, update the comments about the allowed range both below and
        // in galsim/base.py.
        const double minimum_sersic_n = 0.3;   // (Lower bounds has hard limit at ~0.29)
        const double maximum_sersic_n = 7.2;

        // How many Sersic profiles to save in the cache
        const int max_sersic_cache = 100;

    }

    /**
     * @brief Sersic Surface Brightness Profile.
     *
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic index
     * `n`, its `flux`, and the half-light radius `re` (or scale radius `r0`).  Given these
     * properties, the surface brightness profile scales as `I(r) propto exp[-(r/r0)^{1/n}]`, or
     * `I(r) propto exp[-b*(r/re)^{1/n}]`.  The code is limited to 0.3 <= n <= 4.2, with an 
     * exception thrown for values outside that range.
     *
     * The SBProfile representation of a Sersic profile also includes an optional truncation beyond
     * a given radius, by the parameter `trunc`.  The resolution of the truncation radius (in units
     * of half light radius `re`) is limited to 2 decimal places, in order not to overload the 
     * Sersic information caching.
     *
     * When the Sersic profile is specfied by the scale radius with truncation, the normalization is
     * adjusted such that the truncated profile has the specified flux (its half-light radius will
     * differ from an equivalent Sersic without truncation).  Similarly, when the Sersic profile is
     * specified by the half-light radius with truncation, SBSersic generates a profile whose flux
     * and half-light radius is as specified, by adjusting its normalization and scale radius.
     *
     * Another optional parameter, `flux_untruncated = true`, allows the setting of the flux to
     * the untruncated Sersic, while generating a truncated Sersic (i.e., the normalizaton is
     * the same with respect to the untruncated case).  This facilitates the comparison of
     * truncated and untruncated Sersic, as the amplitude (as well as the scale parameter
     * `b=(re/r0)^{1/n}`, if half-light radius is specified) changes when a truncated Sersic is
     * specified in the default setting [`flux_untruncated = false`].  The `flux_untruncated`
     * variable is ignored if `trunc = 0`.
     *
     * Note that when `trunc > 0.` and `flux_untruncated == true`,  the actual flux will not be
     * the same as the specified value; its true flux is returned by the getFlux() method.
     * Similarly for the half-light radius, when the Sersic profile is specified by the half-light
     * radius; the getHalfLightRadius() method will return the true half-light radius.  The scale
     * radius will remain at the same value, if this quantity was used to specify the profile.
     *
     * There are several special cases of the Sersic profile that have their own SBProfiles: n=4
     * (SBDeVaucouleurs), n=1 (SBExponential), n=0.5 (SBGaussian).  These special cases use several
     * simplifications in all calculations, whereas for general n, the Fourier transform must be
     * treated numerically.
     */
    class SBSersic : public SBProfile 
    {
    public:
        enum  RadiusType
        {
            HALF_LIGHT_RADIUS,
            SCALE_RADIUS
        };

        /**
         * @brief Constructor.
         *
         * @param[in] n                 Sersic index.
         * @param[in] size              Size specification.
         * @param[in] rType             Kind of size being specified (HALF_LIGHT_RADIUS or
         *                              SCALE_RADIUS).
         * @param[in] flux              Flux (default `flux = 1.`).
         * @param[in] trunc             Outer truncation radius in same physical units as size;
         *                              `trunc = 0.` for no truncation (default `trunc = 0.`).
         * @param[in] flux_untruncated  If `true`, sets the flux to the untruncated version of the
         *                              Sersic profile with the same index `n` (default
         *                              flux_untruncated = false`).  Ignored if `trunc = 0.`.
         * @param[in] gsparams          GSParams object storing constants that control the accuracy
         *                              of image operations and rendering, if different from the
         *                              default.
         */
        SBSersic(double n, double size, RadiusType rType, double flux,
                 double trunc, bool flux_untruncated, const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBSersic(const SBSersic& rhs);

        /// @brief Destructor.
        ~SBSersic();

        /// @brief Returns the Sersic index `n` of the profile.
        double getN() const;

        /// @brief Returns the scale radius r0 of the Sersic profile `exp[-(r/r_0)^(1/n)]`.
        double getScaleRadius() const;

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
         * @param[in] size              Size specification.
         * @param[in] rType             Kind of size being specified (HALF_LIGHT_RADIUS or
         *                              SCALE_RADIUS).
         * @param[in] flux              Flux (default `flux = 1.`).
         * @param[in] trunc             Outer truncation radius in same physical units as size;
         *                               `trunc = 0.` for no truncation (default `trunc = 0.`).
         * @param[in] flux_untruncated  If `true`, sets the flux to the untruncated version of the
         *                              Sersic profile with the same index `n` (default
         *                              flux_untruncated = false`).  Ignored if `trunc = 0.`.
         * @param[in] gsparams          GSParams object storing constants that control the accuracy
         *                              of image operations and rendering, if different from the
         *                              default.
         */
        SBDeVaucouleurs(double size, RadiusType rType, double flux,
                        double trunc, bool flux_untruncated, const GSParamsPtr& gsparams) :
            SBSersic(4., size, rType, flux, trunc, flux_untruncated, gsparams) {}
    };

}

#endif // SBSERSIC_H

