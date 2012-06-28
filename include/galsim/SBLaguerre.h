// -*- c++ -*-
#ifndef SBLAGUERRE_H
#define SBLAGUERRE_H
/** 
 * @file SBLaguerre.h @brief SBProfile that implements a 2-d Gauss-Laguerre profile (aka shapelets)
 */

#include "SBProfile.h"
#include "Laguerre.h"

namespace galsim {

    /// @brief Class for describing Gauss-Laguerre polynomial Surface Brightness Profiles.
    class SBLaguerre : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] bvec   `bvec[n,n]` contains flux information for the `(n, n)` basis function.
         * @param[in] sigma  scale size of Gauss-Laguerre basis set (default `sigma = 1.`).
         */
        SBLaguerre(LVector bvec=LVector(), double sigma=1.);

        /// @brief Copy Constructor. 
        SBLaguerre(const SBLaguerre& rhs);

        /// @brief Destructor. 
        ~SBLaguerre();

    protected:
        class SBLaguerreImpl;

    private:
        // op= is undefined
        void operator=(const SBLaguerre& rhs);
    };
}

#endif // SBLAGUERRE_H

