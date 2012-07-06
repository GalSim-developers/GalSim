// -*- c++ -*-
#ifndef SBEXPONENTIAL_H
#define SBEXPONENTIAL_H
/** 
 * @file SBExponential.h @brief SBProfile that implements a 2-d exponential profile.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Exponential Surface Brightness Profile.  
     *
     * Surface brightness profile with I(r) propto exp[-r/r_0] for some scale-length r_0.  This is a
     * special case of the Sersic profile, but is given a separate class since the Fourier transform
     * has closed form and can be generated without lookup tables.
     */
    class SBExponential : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor - note that `r0` is scale length, NOT half-light radius `re` as in 
         * SBSersic.
         *
         * @param[in] r0    scale length for the profile that scales as `exp[-(r / r0)]`, NOT the 
         *                  half-light radius `re`.
         * @param[in] flux  flux (default `flux = 1.`).
         */
        SBExponential(double r0, double flux=1.);

        /// @brief Copy constructor.
        SBExponential(const SBExponential& rhs);

        /// @brief Destructor.
        ~SBExponential();

        /// @brief Returns the scale radius of the Exponential profile.
        double getScaleRadius() const;

    protected:

        class ExponentialRadialFunction;
        class ExponentialInfo;
        class SBExponentialImpl;

        // Static class-wide object that does some calculations applicable to all 
        // SBExponential instantiations.
        static ExponentialInfo _info; 

    private:
        // op= is undefined
        void operator=(const SBExponential& rhs);
    };

}

#endif // SBEXPONENTIAL_H

