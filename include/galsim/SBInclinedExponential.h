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

#ifndef GalSim_SBInclinedExponential_H
#define GalSim_SBInclinedExponential_H
/**
 * @file SBInclinedExponential.h @brief SBProfile that implements an inclined exponential profile.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // Constrain range of allowed cached values (h_sini_over_r)
        const double minimum_h_tani_over_r = 0.001; // Somewhat arbitrary cut-off - corresponds to within 0.1 degrees of face-on
        const double maximum_h_tani_over_r = 1000.; // Somewhat arbitrary cut-off - corresponds to within 0.1 degrees of edge-on

        // How many profiles to save in the cache
        const int max_inclined_exponential_cache = 100;

    }

    /**
     * @brief Inclined Exponential surface brightness profile.
     *
     * TODO: Fill in
     */
    class SBInclinedExponential : public SBProfile
    {
    public:

        /**
         * @brief Constructor.
         *
         * @param[in] i                 Inclination angle i in radians, where 0 = face-on, pi/2 = edge-on
         * @param[in] scale_radius      Scale radius of the exponential disk.
         * @param[in] scale_height      Scale height of the exponential disk.
         * @param[in] flux              Flux.
         * @param[in] gsparams          GSParams object storing constants that control the accuracy
         *                              of image operations and rendering, if different from the
         *                              default.
         */
    	SBInclinedExponential(double i, double scale_radius, double scale_height, double flux,
                 const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
    	SBInclinedExponential(const SBInclinedExponential& rhs);

        /// @brief Destructor.
        ~SBInclinedExponential();

        /// @brief Returns the inclination angle 'i' of the profile in radians
        double getI() const;

        /// @brief Returns the scale radius r0 of the disk profile
        double getScaleRadius() const;

        /// @brief Returns the scale height r0 of the disk profile
        double getScaleHeight() const;

        /// @brief Returns the truncation radius
        double getTrunc() const;

    protected:

        class SBInclinedExponentialImpl;

    private:

        // op= is undefined
        void operator=(const SBInclinedExponential& rhs);
    };
}

#endif
