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

#ifndef GalSim_SBKolmogorov_H
#define GalSim_SBKolmogorov_H
/**
 * @file SBKolmogorov.h @brief SBProfile of an Kolmogorov function
 */

#include "SBProfile.h"

namespace galsim {

    namespace sbp {

        // How many Kolmogorov profiles to save in the cache
        const int max_kolmogorov_cache = 100;

    }

    /**
     * @brief Surface Brightness Profile for a Kolmogorov turbulent spectrum.
     *
     */
    class PUBLIC_API SBKolmogorov : public SBProfile
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam_over_r0  lambda / r0 in the physical units adopted (user responsible for
         *                         consistency), where r0 is the Fried parameter.
         *                         The FWHM of the Kolmogorov PSF is ~0.976 lambda/r0
         *                         (e.g., Racine 1996, PASP 699, 108).
         *                         Typical values for the Fried parameter are on the order of
         *                         10 cm for most observatories and up to 20 cm for excellent
         *                         sites. The values are usually quoted at lambda = 500 nm and
         *                         r0 depends on wavelength as [r0 ~ lambda^(6/5)].
         * @param[in] flux         Flux.
         * @param[in] gsparams     GSParams object storing constants that control the accuracy of
         *                         image operations and rendering, if different from the default.
         */
        SBKolmogorov(double lam_over_r0, double flux, const GSParams& gsparams);

        /// @brief Copy constructor
        SBKolmogorov(const SBKolmogorov& rhs);

        /// @brief Destructor.
        ~SBKolmogorov();

        /// @brief Returns lam_over_r0 param of the SBKolmogorov.
        double getLamOverR0() const;

    protected:
        class SBKolmogorovImpl;

    private:
        // op= is undefined
        void operator=(const SBKolmogorov& rhs);
    };

}

#endif

