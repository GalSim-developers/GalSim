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

#ifndef GalSim_SBInclinedExponential_H
#define GalSim_SBInclinedExponential_H
/**
 * @file SBInclinedExponential.h @brief SBProfile that implements an inclined exponential profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Inclined exponential surface brightness profile.
     *
     * Surface brightness profile based on the distribution I(R,z) propto sech^2(z/Hs)*exp(-R/Rs), where
     * Hs is the scale height of the disk, Rs is the scale radius, z is the distance along the minor
     * axis, and R is the distance perpendicular to it. This profile is determined by four parameters:
     * The inclination angle, the scale radius, the scale height, and the flux.
     *
     * Note that the position angle is always zero. A profile with a different position angle can be
     * obtained through the rotate() method of the corresponding Python class.
     */
    class PUBLIC_API SBInclinedExponential : public SBProfile
    {
    public:

        /**
         * @brief Constructor.
         *
         * @param[in] inclination       Inclination angle, where 0 = face-on, pi/2 = edge-on
         * @param[in] scale_radius      Scale radius of the exponential disk.
         * @param[in] scale_height      Scale height of the exponential disk.
         * @param[in] flux              Flux.
         * @param[in] gsparams          GSParams object storing constants that control the accuracy
         *                              of image operations and rendering, if different from the
         *                              default.
         */
        SBInclinedExponential(double inclination, double scale_radius, double scale_height,
                              double flux, const GSParams& gsparams);

        /// @brief Copy constructor.
        SBInclinedExponential(const SBInclinedExponential& rhs);

        /// @brief Destructor.
        ~SBInclinedExponential();

        /// @brief Returns the inclination angle of the profile in radians
        double getInclination() const;

        /// @brief Returns the scale radius r0 of the disk profile
        double getScaleRadius() const;

        /// @brief Returns the scale height h0 of the disk profile
        double getScaleHeight() const;

    protected:

        class SBInclinedExponentialImpl;

    private:

        // op= is undefined
        void operator=(const SBInclinedExponential& rhs);
    };
}

#endif
