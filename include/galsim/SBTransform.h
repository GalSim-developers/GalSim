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

#ifndef GalSim_SBTransform_H
#define GalSim_SBTransform_H
/**
 * @file SBTransform.h @brief SBProfile adapter that transforms another SBProfile.
 * Includes shear, dilation, rotation, translation, and flux scaling.
 *
 */

#include "SBProfile.h"

namespace galsim {

    template <typename T>
    PUBLIC_API void ApplyKImagePhases(
        ImageView<std::complex<T> > image, double imscale, const double* jac,
        double cenx, double ceny, double fluxScaling);

    /**
     * @brief An affine transformation of another SBProfile.
     *
     * Origin of original shape will now appear at `_cen`.
     * Flux is NOT conserved in transformation - surface brightness is preserved.
     * We keep track of all distortions in a 2x2 matrix `M = [(A B), (C D)]` = [row1, row2]
     * plus a 2-element Positon object `cen` for the shift, and a flux scaling,
     * in addition to the scaling implicit in the matrix M = abs(det(M)).
     */
    class PUBLIC_API SBTransform : public SBProfile
    {
    public:
        /**
         * @brief General constructor.
         *
         * @param[in] obj         SBProfile being transformed
         * @param[in] jac         4-element array (A,B,C,D) of 2x2 distortion matrix
         *                          `M = [(A B), (C D)]`
         * @param[in] cen         2-element (x, y) Position for the translational shift.
         * @param[in] ampScaling  Amount by which the SB amplitude should be multiplied.
         * @param[in] gsparams    GSParams object storing constants that control the accuracy of
         *                        image operations and rendering, if different from the default.
         */
        SBTransform(const SBProfile& sbin, const double* jac,
                    const Position<double>& cen, double ampScaling, const GSParams& gsparams);

        /// @brief Copy constructor
        SBTransform(const SBTransform& rhs);

        /// @brief Destructor
        ~SBTransform();

        SBProfile getObj() const;

        void getJac(double& mA, double& mB, double& mC, double& mD) const;

        Position<double> getOffset() const;

        double getFluxScaling() const;

    protected:

        class SBTransformImpl;

    private:
        // op= is undefined
        void operator=(const SBTransform& rhs);
    };

}

#endif

