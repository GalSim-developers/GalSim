/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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

#include "TMV.h"
#include "RealGalaxy.h"

namespace galsim
{

    void ComputeCRGCoefficients(std::complex<double>* coef, std::complex<double>* Sigma,
                                const double* w, const std::complex<double>* kimgs,
                                const std::complex<double>* psf_eff_kimgs,
                                const int nsed, const int nband, const int nkx, const int nky)
    {
        // The basic idea is that the galaxy is modeled as Sum_k sed_k * coef_k, where coef_k is an
        // image in k-space.  Then we have N constraints in the various bandpasses:
        //
        //     PSF_n * obj = kimg_n
        //     Sum_k sed_k PSF_n coef_k = kimg_n
        //
        // And really, the PSF could be different for each SED (if the PSF is chromatic), so we
        // really need to integrate the sed * PSF over the bandpass to get the effective PSF.
        //
        //     Sum_k PSF_eff_kn coef_k = kimg_n
        //
        // This is essentially a matrix equation for each pixel value in the kimg, which we can
        // solve using least-squares solution with QRP decomposition.
        //
        // One last detail, we want to weight by the inverse variance, which means we weight
        // by 1/sqrt(var) on each side of the above equation.  These are input as w.
        //
        // Here is Josh's original python-layer code for doing this:
        /*
        # Solve the weighted linear least squares problem for each Fourier mode.  This is
        # effectively a constrained chromatic deconvolution.  Take advantage of symmetries.
        for ix in range(nkx//2+1):
            for iy in range(nky):
                if (ix == 0 or ix == nkx//2) and iy > nky//2:
                    break # already filled in the rest of this column
                ww = np.diag(w[:, iy, ix])
                A = np.dot(ww, PSF_eff_kimgs[:, :, iy, ix])
                b = np.dot(ww, kimgs[:, iy, ix])
                try:
                    x, resids, rank, singval = np.linalg.lstsq(A, b)
                    # condition number is max singular value over min singular value
                    condnum = np.max(singval) / np.min(singval)
                    # Only bother computing covariance of result if condition number is favorable.
                    if condnum < 1.e12:
                        dx = np.linalg.inv(np.dot(np.conj(A.T), A))
                    else:
                        dx = np.zeros((NSED, NSED), dtype=np.complex128)
                except:
                    x = 0.0
                    dx = np.zeros((NSED, NSED), dtype=np.complex128)

                coef[iy, ix] = x
                Sigma[iy, ix] = dx
                # Save work by filling in conjugates.
                coef[-iy, -ix] = np.conj(x)
                Sigma[-iy, -ix] = np.conj(dx)
        */

        int npix = nkx * nky;
        int nsedsq = nsed * nsed;
        tmv::Matrix<std::complex<double> > A(nband, nsed);
        tmv::Vector<std::complex<double> > b(nband);

        for (int ix=0; ix<nkx/2+1; ++ix) {
            for (int iy=0; iy<nky; ++iy) {
                if ((ix == 0 || ix == nkx/2) && iy > nky/2) {
                    // already filled in the rest of this column
                    break;
                }
                tmv::ConstDiagMatrixView<double> ww =
                    tmv::DiagMatrixViewOf(w + iy*nkx + ix, nband, npix);
                tmv::ConstMatrixView<std::complex<double> > psf =
                    tmv::MatrixViewOf(psf_eff_kimgs + iy*nkx + ix, nband, nsed, npix * nsed, npix);
                tmv::ConstVectorView<std::complex<double> > kimg =
                    tmv::VectorViewOf(kimgs + iy*nkx + ix, nband, npix);
                tmv::VectorView<std::complex<double> > x =
                    tmv::VectorViewOf(coef + iy*nkx*nsed + ix*nsed, nsed, 1);
                tmv::MatrixView<std::complex<double> > dx =
                    tmv::MatrixViewOf(Sigma + iy*nkx*nsedsq + ix*nsedsq, nsed, nsed, nsed, 1);

                A = ww * psf;
                b = ww * kimg;
                try {
                    x = b / A;
                    A.makeInverseATA(dx);
                } catch (tmv::Singular) {
                    A.divideUsing(tmv::QRP);
                    x = b / A;
                    A.makeInverseATA(dx);
                }
                if (ix > 0 && iy > 0) {
                    int ix2 = nkx - ix;
                    int iy2 = nky - iy;
                    if (ix == ix2 && iy == iy2) continue;
                    tmv::VectorView<std::complex<double> > x2 =
                        tmv::VectorViewOf(coef + iy2*nkx*nsed + ix2*nsed, nsed, 1);
                    tmv::MatrixView<std::complex<double> > dx2 =
                        tmv::MatrixViewOf(Sigma + iy2*nkx*nsedsq + ix2*nsedsq, nsed, nsed, nsed, 1);
                    x2 = x.conjugate();
                    dx2 = dx.conjugate();
                }
            }
        }
    }

}
