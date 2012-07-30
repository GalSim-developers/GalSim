// -*- c++ -*-
#ifndef SBTRANSFORM_H
#define SBTRANSFORM_H
/** 
 * @file SBTransform.h @brief SBProfile adapter that transforms another SBProfile.
 * Includes shear, dilation, rotation, translation, and flux scaling.
 *
 */

#include "SBProfile.h"

namespace galsim {

    /**
     * @brief An affine transformation of another SBProfile.
     *
     * Origin of original shape will now appear at `_cen`.
     * Flux is NOT conserved in transformation - surface brightness is preserved.
     * We keep track of all distortions in a 2x2 matrix `M = [(A B), (C D)]` = [row1, row2] 
     * plus a 2-element Positon object `cen` for the shift, and a flux scaling,
     * in addition to the scaling implicit in the matrix M = abs(det(M)).
     */
    class SBTransform : public SBProfile
    {
    public:
        /** 
         * @brief General constructor.
         *
         * @param[in] sbin SBProfile being transform
         * @param[in] mA A element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mB B element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mC C element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] mD D element of 2x2 distortion matrix `M = [(A B), (C D)]` = [row1, row2]
         * @param[in] cen 2-element (x, y) Position for the translational shift.
         * @param[in] fluxScaling Amount by which the flux should be multiplied.
         */
        SBTransform(const SBProfile& sbin,
                    double mA, double mB, double mC, double mD,
                    const Position<double>& cen=Position<double>(0.,0.), double fluxScaling=1.);

        /** 
         * @brief Construct from an input CppEllipse 
         *
         * @param[in] sbin SBProfile being transformed
         * @param[in] e  CppEllipse.
         * @param[in] fluxScaling Amount by which the flux should be multiplied.
         */
        SBTransform(const SBProfile& sbin,
                    const CppEllipse& e=CppEllipse(),
                    double fluxScaling=1.);

        /// @brief Copy constructor
        SBTransform(const SBTransform& rhs);

        /// @brief Destructor
        ~SBTransform();

    protected:

        class SBTransformImpl;

        static std::complex<double> _kValueNoPhaseNoDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueNoPhaseWithDet(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& , const Position<double>& );
        static std::complex<double> _kValueWithPhase(
            const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
            const Position<double>& k, const Position<double>& cen);

        static Position<double> _fwd_normal(
            double mA, double mB, double mC, double mD, double x, double y, double )
        { return Position<double>(mA*x + mB*y, mC*x + mD*y); }
        static Position<double> _inv_normal(
            double mA, double mB, double mC, double mD, double x, double y, double invdet)
        { return Position<double>(invdet*(mD*x - mB*y), invdet*(-mC*x + mA*y)); }
        static Position<double> _ident(
            double , double , double , double , double x, double y, double )
        { return Position<double>(x,y); }

    private:
        // op= is undefined
        void operator=(const SBTransform& rhs);
    };

}

#endif // SBTRANSFORM_H

