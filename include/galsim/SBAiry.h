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

#ifndef GalSim_SBAiry_H
#define GalSim_SBAiry_H
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
     * 5 lam/D or EE>(1-folding_threshold).  Schroeder (10.1.18) gives limit of EE at large radius.
     * This stepK could probably be relaxed, it makes overly accurate FFTs.
     * Note x & y are in units of lambda/D here.  Integral over area will give unity in this
     * normalization.
     */
    class PUBLIC_API SBAiry : public SBProfile
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
        SBAiry(double lam_over_D, double obscuration, double flux, const GSParams& gsparams);

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

#endif

