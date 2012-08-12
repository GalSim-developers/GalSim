// -*- c++ -*-
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

