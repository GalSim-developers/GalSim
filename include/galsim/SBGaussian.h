// -*- c++ -*-
#ifndef SBGAUSSIAN_H
#define SBGAUSSIAN_H
/** 
 * @file SBGaussian.h @brief SBProfile that implements a 2-d Gaussian profile.
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief Gaussian Surface Brightness Profile
     *
     * The Gaussian Surface Brightness Profile is characterized by two properties, its `flux`
     * and the characteristic size `sigma` where the radial profile of the circular Gaussian
     * drops off as `exp[-r^2 / (2. * sigma^2)]`.
     * The maxK() and stepK() are for the SBGaussian are chosen to extend to 4 sigma in both 
     * real and k domains, or more if needed to reach the `alias_threshold` spec.
     */
    class SBGaussian : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] sigma  characteristic size, surface brightness scales as 
         *                   `exp[-r^2 / (2. * sigma^2)]`.
         * @param[in] flux   flux of the Surface Brightness Profile (default `flux = 1.`).
         */
        SBGaussian(double sigma, double flux=1.);

        /// @brief Copy constructor.
        SBGaussian(const SBGaussian& rhs);

        /// @brief Destructor.
        ~SBGaussian();

        /// @brief Returns the characteristic size sigma of the Gaussian profile.
        double getSigma() const;

    protected:

        class SBGaussianImpl;

    private:
        // op= is undefined
        void operator=(const SBGaussian& rhs);
    };

}

#endif // SBGAUSSIAN_H

