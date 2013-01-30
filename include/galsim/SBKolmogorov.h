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

#ifndef SBKOLMOGOROV_H
#define SBKOLMOGOROV_H
/** 
 * @file SBKolmogorov.h @brief SBProfile of an Kolmogorov function
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Surface Brightness Profile for a Kolmogorov turbulent spectrum.
     *
     */
    class SBKolmogorov : public SBProfile 
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
         *                         r0 depends on wavelength as [r0 ~ lambda^(-6/5)].
         * @param[in] flux         flux (default `flux = 1.`).
         */
        SBKolmogorov(double lam_over_r0, double flux=1.);

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

#endif // SBKOLMOGOROV_H

