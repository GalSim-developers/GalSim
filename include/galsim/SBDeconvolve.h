// -*- c++ -*-
#ifndef SBDECONVOLVE_H
#define SBDECONVOLVE_H
/** 
 * @file SBDeconvolve.h @brief SBProfile adapter which inverts its subject in k space to effect a
 * deconvolution.
 */


#include "SBProfile.h"

namespace galsim {

    /**
     * @brief SBProfile adapter which inverts its subject in k space to effect a deconvolvution.
     *
     * (TODO: Add more docs here!)
     */
    class SBDeconvolve : public SBProfile 
    {
    public:
        /// @brief Constructor.
        SBDeconvolve(const SBProfile& adaptee);

        /// @brief Copy constructor.
        SBDeconvolve(const SBDeconvolve& rhs);

        /// @brief Destructor.
        ~SBDeconvolve();

    protected:

        class SBDeconvolveImpl;

    private:
        // op= is undefined
        void operator=(const SBDeconvolve& rhs);
    };

}

#endif // SBDECONVOLVE_H
