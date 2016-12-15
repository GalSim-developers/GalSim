/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBInclinedSersic_H
#define GalSim_SBInclinedSersic_H
/**
 * @file SBInclinedSersic.h @brief SBProfile that implements a Sersic profile, inclined to the line
 * of sight.
 */

#include "Angle.h"
#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Inclined Sersic Surface Brightness Profile.
     *
     * The Sersic Surface Brightness Profile is characterized by three properties: its Sersic index
     * `n`, its `flux`, and the half-light radius `re` (or scale radius `r0`).  Given these
     * properties, the surface brightness profile scales as `I(r) propto exp[-(r/r0)^{1/n}]`, or
     * `I(r) propto exp[-b*(r/re)^{1/n}]`.  The code is limited to 0.3 <= n <= 6.2, with an
     * exception thrown for values outside that range.
     *
     * The SBProfile representation of a Sersic profile also includes an optional truncation beyond
     * a given radius, by the parameter `trunc`.  Internal Sersic information are cached according
     * to the `(n, trunc/r0)` pair.  All internal calculations are based on the scale radius `r0`.
     * If the profile is specified by the half-light radius `re`, the corresponding scale radius
     * `r0` is calculated, and vice versa.
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
     * truncated and untruncated Sersic, as the amplitude (as well as the scale radius `r0`,
     * if half-light radius is specified) changes when a truncated Sersic is specified with
     * `flux_untruncated = false`.  The `flux_untruncated` variable is ignored if `trunc = 0`.
     *
     * Note that when `trunc > 0.` and `flux_untruncated == true`, the actual flux will not be
     * the same as the specified value; its true flux is returned by the getFlux() method.
     * Similarly for the half-light radius, when the Sersic profile is specified by the half-light
     * radius; the getHalfLightRadius() method will return the true half-light radius.  The scale
     * radius will remain at the same value, if this quantity was used to specify the profile.
     *
     * There are two special cases of the Sersic profile that have their own SBProfiles: n=1
     * (SBExponential), n=0.5 (SBGaussian).  These special cases use several simplifications in
     * all calculations, whereas for general n, the Fourier transform must be treated numerically.
     *
     * This implementation of the Sersic profile additionally allows inclination relative to the
     * line of sight. The true profile is assumed to follow a Sersic distribution in R, multiplied
     * by sech^2(z/Hs), where Hs is the scale height of the disk and z is the distance along the
     * minor axis. The inclination angle determines how elliptical the profile appears.
     *
     * Note that the position angle is always zero. A profile with a different position angle can be
     * obtained through the rotate() method of the corresponding Python class.
     *
     * If the inclination will always be zero (face-on), the SBSersic class can instead be used
     * as a slightly faster alternative. If no truncation radius will be applied and n=1, the
     * SBInclinedExponential class can be used as a much faster alternative.
     */
    class SBInclinedSersic : public SBProfile
    {
    public:
        enum  RadiusType
        {
            HALF_LIGHT_RADIUS,
            SCALE_RADIUS
        };
        enum  HeightType
        {
            SCALE_H_OVER_R,
            SCALE_HEIGHT
        };

        /**
         * @brief Constructor.
         *
         * @param[in] n                 Sersic index.
         * @param[in] inclination       Inclination of the disk relative to line of sight, where
         *                              0 = face-on and pi/2 = edge-on.
         * @param[in] size              Size specification.
         * @param[in] rType             Kind of size being specified (HALF_LIGHT_RADIUS or
         *                              SCALE_RADIUS).
         * @param[in] height            Height specification.
         * @param[in] rType             Kind of height being specified (SCALE_H_OVER_R or
         *                              SCALE_HEIGHT).
         * @param[in] flux              Flux.
         * @param[in] trunc             Outer truncation radius in same physical units as size;
         *                              `trunc = 0.` for no truncation.
         * @param[in] flux_untruncated  If `true`, sets the flux to the untruncated version of the
         *                              Sersic profile with the same index `n`.  Ignored if
         *                              `trunc = 0.`.
         * @param[in] gsparams          GSParams object storing constants that control the accuracy
         *                              of image operations and rendering, if different from the
         *                              default.
         */
        SBInclinedSersic(double n, Angle inclination, double size, RadiusType rType,
                 double height, HeightType hType, double flux,
                 double trunc, bool flux_untruncated, const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBInclinedSersic(const SBInclinedSersic& rhs);

        /// @brief Destructor.
        ~SBInclinedSersic();

        /// @brief Returns the Sersic index `n` of the profile.
        double getN() const;

        /// @brief Returns the inclination angle of the profile as an Angle instance
        Angle getInclination() const;

        /// @brief Returns the scale radius r0 of the Sersic profile.
        double getScaleRadius() const;

        /// @brief Returns the half light radius of the Sersic profile (if it were face-on).
        double getHalfLightRadius() const;

        /// @brief Returns the scale height h0 of the disk profile
        double getScaleHeight() const;

        /// @brief Returns the truncation radius
        double getTrunc() const;

    protected:

        class SBInclinedSersicImpl;

    private:

        // op= is undefined
        void operator=(const SBInclinedSersic& rhs);
    };
}

#endif
