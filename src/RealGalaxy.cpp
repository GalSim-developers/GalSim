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

#ifdef USE_TMV
#include "TMV.h"
#else
#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
#include "Eigen/Dense"
#endif

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

#ifdef USE_TMV
        typedef tmv::Matrix<std::complex<double> > MatrixXcd;
        typedef tmv::Vector<std::complex<double> > VectorXcd;
#else
        using Eigen::MatrixXcd;
        using Eigen::VectorXcd;
        using Eigen::VectorXd;
#endif
        int npix = nkx * nky;
        int nsedsq = nsed * nsed;
        MatrixXcd A(nband, nsed);
        VectorXcd b(nband);

        for (int ix=0; ix<nkx/2+1; ++ix) {
            for (int iy=0; iy<nky; ++iy) {
                if ((ix == 0 || ix == nkx/2) && iy > nky/2) {
                    // already filled in the rest of this column
                    break;
                }
#ifdef USE_TMV
                tmv::ConstDiagMatrixView<double> ww =
                    tmv::DiagMatrixViewOf(w + iy*nkx + ix, nband, npix);
                tmv::ConstMatrixView<std::complex<double> > psf =
                    tmv::MatrixViewOf(psf_eff_kimgs + iy*nkx + ix, nband, nsed, npix * nsed, npix);
                tmv::ConstVectorView<std::complex<double> > kimg =
                    tmv::VectorViewOf(kimgs + iy*nkx + ix, nband, npix);
                tmv::VectorView<std::complex<double> > x =
                    tmv::VectorViewOf(coef + iy*nkx*nsed + ix*nsed, nsed, 1);
                tmv::MatrixView<std::complex<double> > dxT =
                    tmv::MatrixViewOf(Sigma + iy*nkx*nsedsq + ix*nsedsq, nsed, nsed, 1, nsed);

                A = ww * psf;
                b = ww * kimg;
                try {
                    x = b / A;
                    A.makeInverseATA(dxT);
                } catch (tmv::Singular) {
                    A.divideUsing(tmv::QRP);
                    x = b / A;
                    A.makeInverseATA(dxT);
                }
#else
                using Eigen::Dynamic;
                using Eigen::InnerStride;
                using Eigen::Stride;
                using Eigen::Upper;
                Eigen::Map<const VectorXd,0,InnerStride<> > ww(
                    w+iy*nkx+ix, nband, InnerStride<>(npix));
                Eigen::Map<const MatrixXcd,0,Stride<Dynamic,Dynamic> > psf(
                    psf_eff_kimgs + iy*nkx + ix, nband, nsed,
                    Stride<Dynamic,Dynamic>(npix, npix * nsed));
                Eigen::Map<const VectorXcd,0,InnerStride<> > kimg(
                    kimgs + iy*nkx + ix, nband, InnerStride<>(npix));
                Eigen::Map<VectorXcd> x(coef + iy*nkx*nsed + ix*nsed, nsed);
                Eigen::Map<MatrixXcd> dxT(Sigma + iy*nkx*nsedsq + ix*nsedsq, nsed, nsed);

                A = ww.asDiagonal() * psf;
                b = ww.asDiagonal() * kimg;
                Eigen::HouseholderQR<MatrixXcd> qr = A.householderQr();
                Eigen::Diagonal<const MatrixXcd> Rdiag = qr.matrixQR().diagonal();
                if (Rdiag.array().abs().minCoeff() < 1.e-15*Rdiag.array().abs().maxCoeff()) {
                    // Then (nearly) signular.  Use QRP instead.  (This should be fairly rare.)
                    Eigen::ColPivHouseholderQR<MatrixXcd> qrp = A.colPivHouseholderQr();
                    x = qrp.solve(b);

                    // A = Q R Pt
                    // (AtA)^-1 = (PRtQtQRPt)^-1 = (PRtRPt)^-1 = P R^-1 Rt^-1 Pt
                    const int nzp = qrp.nonzeroPivots();
                    Eigen::TriangularView<const Eigen::Block<const MatrixXcd>, Upper> R =
                        qrp.matrixR().topLeftCorner(nzp,nzp).triangularView<Upper>();
                    dxT.setIdentity();
                    R.adjoint().solveInPlace(dxT.topLeftCorner(nzp,nzp));
                    R.solveInPlace(dxT.topLeftCorner(nzp,nzp));
                    dxT = qrp.colsPermutation() * dxT * qrp.colsPermutation().transpose();
                } else {
                    x = qr.solve(b);
                    // A = Q R
                    // (AtA)^-1 = (RtQtQR)^-1 = (RtR)^-1 = R^-1 Rt^-1
                    Eigen::TriangularView<const Eigen::Block<const MatrixXcd>, Upper> R =
                        qr.matrixQR().topRows(nsed).triangularView<Upper>();
                    dxT.setIdentity();
                    R.adjoint().solveInPlace(dxT);
                    R.solveInPlace(dxT);
                }
#endif


                if (ix > 0 && iy > 0) {
                    int ix2 = nkx - ix;
                    int iy2 = nky - iy;
                    if (ix == ix2 && iy == iy2) continue;
#ifdef USE_TMV
                    tmv::VectorViewOf(coef + iy2*nkx*nsed + ix2*nsed, nsed, 1) = x.conjugate();
                    tmv::MatrixViewOf(Sigma + iy2*nkx*nsedsq + ix2*nsedsq, nsed, nsed, 1, nsed) =
                        dxT.conjugate();
#else
                    Eigen::Map<VectorXcd>(coef + iy2*nkx*nsed + ix2*nsed, nsed) = x.conjugate();
                    Eigen::Map<MatrixXcd>(Sigma + iy2*nkx*nsedsq + ix2*nsedsq, nsed, nsed) =
                        dxT.conjugate();
#endif
                }
            }
        }
    }

}
