// -*- c++ -*-
#ifndef SBADD_H
#define SBADD_H
/** 
 * @file SBAdd.h @brief SBProfile adapter that is the sum of 2 or more other SBProfiles.
 */

#include "SBProfile.h"

namespace galsim {

    /** 
     * @brief Sums SBProfiles. 
     */
    class SBAdd : public SBProfile 
    {
    public:

        /** 
         * @brief Constructor, 2 inputs.
         *
         * @param[in] s1 first SBProfile.
         * @param[in] s2 second SBProfile.
         */
        SBAdd(const SBProfile& s1, const SBProfile& s2);

        /** 
         * @brief Constructor, list of inputs.
         *
         * @param[in] slist list of SBProfiles.
         */
        SBAdd(const std::list<SBProfile>& slist);

        /// @brief Copy constructor.
        SBAdd(const SBAdd& rhs);

        /// @brief Destructor.
        ~SBAdd();

    protected:

        class SBAddImpl;

    private:
        // op= is undefined
        void operator=(const SBAdd& rhs);
    };
}

#endif // SBPROFILE_H

