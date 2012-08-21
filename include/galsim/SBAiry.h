// -*- c++ -*-
#ifndef SBAIRY_H
#define SBAIRY_H
/** 
 * @file SBAiry.h @brief SBProfile of an Airy function with an optional obscuration.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Surface Brightness Profile for the Airy disk (perfect diffraction-limited PSF for a 
     * circular aperture), with central obscuration.
     *
     * maxK() is set at the hard limit for Airy disks, stepK() makes transforms go to at least 
     * 5 lam/D or EE>(1-alias_threshold).  Schroeder (10.1.18) gives limit of EE at large radius.
     * This stepK could probably be relaxed, it makes overly accurate FFTs.
     * Note x & y are in units of lambda/D here.  Integral over area will give unity in this 
     * normalization.
     */
    class SBAiry : public SBProfile 
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param[in] lam_over_D   `lam_over_D` = (lambda * focal length) / (telescope diam) if 
         *                         arg is focal plane position, else `lam_over_D` = 
         *                         lambda / (telescope diam) if arg is in radians of field angle.
         * @param[in] obscuration  linear dimension of central obscuration as fraction of pupil
         *                         dimension (default `obscuration = 0.`).
         * @param[in] flux         flux (default `flux = 1.`).
         */
        SBAiry(double lam_over_D, double obscuration=0., double flux=1.);

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

#endif // SBAIRY_H

