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

#include "CorrelatedNoise.h"

namespace galsim {

    /*
     * Covariance matrix calculation using the input SBProfile, the dimensions of the image for
     * which a covariance matrix is desired (in the form of a Bounds), and a scale dx
     */
    void calculateCovarianceMatrix(ImageView<double>& cov,
        const SBProfile& sbp, const Bounds<int>& bounds, double dx)
    {
        // Calculate the required dimensions of the image for which a covariance matrix is needed
        int idim = 1 + bounds.getXMax() - bounds.getXMin();
        int jdim = 1 + bounds.getYMax() - bounds.getYMin();
        int covdim = idim * jdim;

        int k, ell; // k and l are indices that refer to image pixel separation vectors in the
                    // correlation func.
        double x_k, y_ell; // physical vector separations in the correlation func, dx * k etc.

        for (int i=1; i<=covdim; i++) {
            for (int j=i; j<=covdim; j++) {
                k = ((j - 1) / jdim) - ((i - 1) / idim);  // using integer division rules here
                ell = ((j - 1) % jdim) - ((i - 1) % idim);
                x_k = double(k) * dx;
                y_ell = double(ell) * dx;
                Position<double> p = Position<double>(x_k, y_ell);
                cov.setValue(i, j, sbp.xValue(p));
            }

        }
    }

}
