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

#ifndef GalSim_SBMoffatlet_H
#define GalSim_SBMoffatlet_H
/**
 * @file SBMoffatlet.h @brief SBProfile that implements a 2-d Moffatlet.
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // Constrain range of allowed Moffat index beta.

        const double minimum_moffat_beta = 1.1;
        const double maximum_moffat_beta = 7.0;

        // How many Moffatlet profiles to save in the cache
        // Start with 100, though it may make sense to increase this depending on what value of
        // jmax we end up settling on.
        const int max_moffatlet_cache = 100;
    }

    /**
     * @brief Moffatlet Surface Brightness Profile.
     *
     * Basis functions in the series expansion of the Moffat profile.  The expansion proceeds
     * analogously to the Spergel series expansion, but in real-space instead of Fourier space.
     * This is natural since a Moffat profile is the Fourier transform of a Spergel profile.
     *
     * Reference:
     *   D. N. Spergel, "ANALYTICAL GALAXY PROFILES FOR PHOTOMETRIC AND LENSING ANALYSIS,""
     *   ASTROPHYS J SUPPL S 191(1), 58-65 (2010) [doi:10.1088/0067-0049/191/1/58].
     */
    class SBMoffatlet : public SBProfile
    {
    public:
        /**
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in
         * SBSersic.
         *
         * @param[in] beta     Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
         * @param[in] r0       scale radius.
         * @param[in] j        Radial index.
         * @param[in] q        Azimuthal index.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBMoffatlet(double beta, double r0, int j, int q,
                    const GSParamsPtr& gsparams);

        /// @brief Copy constructor.
        SBMoffatlet(const SBMoffatlet& rhs);

        /// @brief Destructor.
        ~SBMoffatlet();

        /// @brief Returns the Moffat index `nu` of the profile.
        double getBeta() const;

        /// @brief Returns the scale radius of the Moffat profile.
        double getScaleRadius() const;

        /// @brief Return jq indices of the Moffatlet.
        int getJ() const;
        int getQ() const;

    protected:


        class SBMoffatletImpl;
    private:
        // op= is undefined
        void operator=(const SBMoffatlet& rhs);
    };

}

#endif
