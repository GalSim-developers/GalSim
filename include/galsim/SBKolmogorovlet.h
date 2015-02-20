/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBKolmogorovlet_H
#define GalSim_SBKolmogorovlet_H
/**
 * @file SBKolmogorovlet.h @brief SBProfile that implements a 2-d Kolmogorovlet.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // How many Kolmogorovlet profiles to save in the cache
        // Start with 100
        const int max_kolmogorovlet_cache = 100;

    }

    /**
     * @brief Kolmogorovlet Surface Brightness Profile.
     *
     * FILL THIS IN:
     * Essentially apply the same style of Taylor expansion in Spergelets to
     * the Kolmogorov profile.
     */
    class SBKolmogorovlet : public SBProfile
    {
    public:
        // enum  RadiusType
        // {
        //     HALF_LIGHT_RADIUS,
        //     SCALE_RADIUS
        // };
        /**
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in
         * SBSersic.
         *
         * @param[in] r0       scale radius.
         * @param[in] j        Radial index.
         * @param[in] q        Azimuthal index.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBKolmogorovlet(double r0, int j, int q,
                    const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBKolmogorovlet(const SBKolmogorovlet& rhs);

        /// @brief Destructor.
        ~SBKolmogorovlet();

        /// @brief Returns the scale radius of the Spergel profile.
        double getScaleRadius() const;

        /// @brief Return jmn indices of the Kolmogorovlet.
        int getJ() const;
        int getQ() const;

    protected:

        class SBKolmogorovletImpl;

    private:
        // op= is undefined
        void operator=(const SBKolmogorovlet& rhs);
    };

}

#endif
