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

//#define DEBUGLOGGING

#include <vector>
#include "Std.h"
#include "WCS.h"
#include "math/Horner.h"

namespace galsim {

    void ApplyCD(int n, double* x, double* y, const double* cd)
    {
        // For a 2x2 matrix multiplies two vectors, it's actually best to just do it manually.
        double a = cd[0];
        double b = cd[1];
        double c = cd[2];
        double d = cd[3];

        double u,v;
        for(; n; --n) {
            u = a * *x + b * *y;
            v = c * *x + d * *y;
            *x++ = u;
            *y++ = v;
        }
    }

    void InvertAB(int n, int nab, const double* u, const double* v, const double* ab,
                  double* x, double* y, bool doiter, int nabp, const double* abp)
    {
        // Find (x,y) such that:
        //   u = Horner2D(x,y,A) = [1 x x^2 x^3 ...] A [1 y y^2 y^3 ...]T
        //   v = Horner2D(x,y,B) = [1 x x^2 x^3 ...] B [1 y y^2 y^3 ...]T
        // A = ab[0], B = ab[1] (in python)
        //
        // Use Newton's method.
        // du/dx = [0 1 2x 3x^2 ...] A [1 y y^2 y^3 ...]T
        // du/dy = [1 x x^2 x^3 ...] A [0 1 2y 3y^2 ...]T
        // dv/dx = [0 1 2x 3x^2 ...] B [1 y y^2 y^3 ...]T
        // dv/dy = [1 x x^2 x^3 ...] B [0 1 2y 3y^2 ...]T
        //
        // Rather than adjust the x,y vectors to compute these, we can adjust A,B so that
        // we can continue to use the regular x,y for the Horner2D call.
        // A_dudx = [  A10   A11   A12 ... ]
        //          [ 2A20  2A21  2A22 ... ]
        //          [ 3A30  3A31  3A32 ... ]
        //          [ ...                  ]
        // And, since A is initially upper-left triangular, the last column is all 0's.
        // So we can truncate it by 1 in both directions.  If nab = 4, we obtain
        // A_dudx = [  A10   A11   A12 ]
        //          [ 2A20  2A21    0  ]
        //          [ 3A30    0     0  ]
        // Similarly,
        // A_dudy = [ A01  2A02  3A03 ... ]
        //          [ A11  2A12  3A13 ... ]
        //          [ A21  2A22  3A23 ... ]
        //          [ ...                 ]
        // The dv matrices are equivalent for B.
        //
        // Once we compute J = [ du/dx  du/dy ]
        //                     [ dv/dx  dv/dy ]
        //
        // the Newton step is: [ dx ] = J^-1 [ du ]
        //                     [ dy ]        [ dv ]
        // x <- x + dx
        // y <- y + dy
        //
        // Here is Josh's implementation of all this in Python, which we will reimplement
        // in C++ below.  (With some adjustmet of variable names to match the above notation.)
        /*
        # Assemble horner2d coefs for derivatives
        dudxcoef = (np.arange(nab)[:,None]*ab[0])[1:,:-1]
        dudycoef = (np.arange(nab)*ab[0])[:-1,1:]
        dvdxcoef = (np.arange(nab)[:,None]*ab[1])[1:,:-1]
        dvdycoef = (np.arange(nab)*ab[1])[:-1,1:]

        # Usually converges in ~3 iterations.
        for iter in range(10):  # pragma: no branch
            # Want Jac^-1 . du
            # du
            du = horner2d(x, y, ab[0], triangle=True) - u
            dv = horner2d(x, y, ab[1], triangle=True) - v
            # J
            dudx = horner2d(x, y, dudxcoef, triangle=True)
            dudy = horner2d(x, y, dudycoef, triangle=True)
            dvdx = horner2d(x, y, dvdxcoef, triangle=True)
            dvdy = horner2d(x, y, dvdycoef, triangle=True)
            # J^-1 . du
            det = dudx*dvdy - dudy*dvdx
            duu = -(du*dvdy - dv*dudy)/det
            dvv = -(-du*dvdx + dv*dudx)/det

            x += dx
            y += dy
            # Do at least 3 iterations before spending the time to check for
            # convergence.
            if iter >= 2 and np.max(np.abs(np.array([dx, dy]))) < 1e-12:
                break
        else:  # pragma: no cover
            raise GalSimError("Unable to solve for image_pos.")
        */
        using math::Horner2D;
        dbg<<"Start InvertAB\n";
        xdbg<<"u = "<<u[0]<<std::endl;
        xdbg<<"v = "<<v[0]<<std::endl;
        xdbg<<"x = "<<x[0]<<std::endl;
        xdbg<<"y = "<<y[0]<<std::endl;
        dbg<<"n = "<<n<<std::endl;
        dbg<<"nab = "<<nab<<std::endl;

        // Save these for error message or if we use abp.
        const double* u0 = u;
        const double* v0 = v;
        double* x0 = x;
        double* y0 = y;
        int n0 = n;

        // Do the iteration in batches of at most 256, so we can allocate on the stack
        // and we can keep everything in L1 cache.
        int nblock = std::min(n, 256);
        xdbg<<"nblock = "<<nblock<<std::endl;

        double temp[nblock];
        if (abp) {
            dbg<<"Using abp\n";
            const double* Ap = abp;
            const double* Bp = abp + nabp*nabp;
            while (n) {
                int n1 = std::min(n, 256);
                xdbg<<"n = "<<n<<"  "<<n1<<std::endl;
                xdbg<<"u = "<<u[0]<<std::endl;
                xdbg<<"v = "<<v[0]<<std::endl;
                xdbg<<"x = "<<x[0]<<std::endl;
                xdbg<<"y = "<<y[0]<<std::endl;
                Horner2D(u, v, n1, Ap, nabp, nabp, x, temp);  // x = Horner2D(u, v, Ap)
                Horner2D(u, v, n1, Bp, nabp, nabp, y, temp);  // y = Horner2D(u, v, Bp)
                xdbg<<"x => "<<x[0]<<std::endl;
                xdbg<<"y => "<<y[0]<<std::endl;
                x += n1;
                y += n1;
                u += n1;
                v += n1;
                n -= n1;
            }
            if (!doiter) return;
            u = u0;  // Reset back to the original values.
            v = v0;
            x = x0;
            y = y0;
            n = n0;
        }

        const double* A = ab;
        const double* B = ab + nab*nab;
        double A_dudx[(nab-1)*(nab-1)];
        double A_dudy[(nab-1)*(nab-1)];
        double B_dvdx[(nab-1)*(nab-1)];
        double B_dvdy[(nab-1)*(nab-1)];
        for (int i=1; i<nab; ++i) {
            for (int j=1; j<nab; ++j) {
                int k = (i-1)*(nab-1) + (j-1);
                A_dudx[k] = A[i*nab+(j-1)] * i;
                A_dudy[k] = A[(i-1)*nab+j] * j;
                B_dvdx[k] = B[i*nab+(j-1)] * i;
                B_dvdy[k] = B[(i-1)*nab+j] * j;
            }
        }
#ifdef DEBUGLOGGING
        xdbg<<"A_dudx = ";
        for (int k=0; k<(nab-1)*(nab-1); ++k) xdbg<<A_dudx[k]<<" ";
        xdbg<<"\nA_dudy = ";
        for (int k=0; k<(nab-1)*(nab-1); ++k) xdbg<<A_dudy[k]<<" ";
        xdbg<<"\nB_dvdx = ";
        for (int k=0; k<(nab-1)*(nab-1); ++k) xdbg<<B_dvdx[k]<<" ";
        xdbg<<"\nB_dvdy = ";
        for (int k=0; k<(nab-1)*(nab-1); ++k) xdbg<<B_dvdy[k]<<" ";
        xdbg<<std::endl;
#endif

        // More temporary arrays
        double du[nblock];
        double dv[nblock];
        double dudx[nblock];
        double dudy[nblock];
        double dvdx[nblock];
        double dvdy[nblock];

        const int MAX_ITER = 10;
        bool not_converged = false;

        while (n) {
            dbg<<"n = "<<n<<std::endl;
            int n1 = std::min(n, 256);
            for (int iter=0; iter<10; ++iter) {
                Horner2D(x, y, n1, A, nab, nab, du, temp);  // u' = Horner2D(x, y, A)
                for(int m=0; m<n1; ++m) du[m] -= u[m];      // du = u' - u
                Horner2D(x, y, n1, B, nab, nab, dv, temp);  // v' = Horner2D(x, y, B)
                for(int m=0; m<n1; ++m) dv[m] -= v[m];      // dv = v' - v
                xdbg<<"du,dv = "<<du[0]<<", "<<dv[0]<<std::endl;

                Horner2D(x, y, n1, A_dudx, nab-1, nab-1, dudx, temp);  // -> dudx
                Horner2D(x, y, n1, A_dudy, nab-1, nab-1, dudy, temp);  // -> dudy
                Horner2D(x, y, n1, B_dvdx, nab-1, nab-1, dvdx, temp);  // -> dvdx
                Horner2D(x, y, n1, B_dvdy, nab-1, nab-1, dvdy, temp);  // -> dvdy
                xdbg<<"dudx = "<<dudx[0]<<std::endl;
                xdbg<<"dudy = "<<dudy[0]<<std::endl;
                xdbg<<"dvdx = "<<dvdx[0]<<std::endl;
                xdbg<<"dvdy = "<<dvdy[0]<<std::endl;

                // Newton step [dx dy] = -J^-1 [du dv]
                double max_step = 0.;
                for(int m=0; m<n1; ++m) {
                    double det = dudx[m] * dvdy[m] - dudy[m] * dvdx[m];
                    double dx = -(dvdy[m] * du[m] - dudy[m] * dv[m]) / det;
                    double dy = -(-dvdx[m] * du[m] + dudx[m] * dv[m]) / det;
                    if (m == 0) {
                        xdbg<<"dx,dy = "<<dx<<", "<<dy<<std::endl;
                        xdbg<<"x,y = "<<x[m]<<", "<<y[m]<<std::endl;
                    }
                    x[m] += dx;
                    y[m] += dy;
                    // Note: if |x| or |y| > 1, then the relevant test is a fractional step, not
                    //       the absolute step.  So divide by max(1,|x|) and max(1,|y|).
                    double abs_step = std::max(std::abs(dx/std::max(1.,std::abs(x[m]))),
                                               std::abs(dy/std::max(1.,std::abs(y[m]))));
                    if (abs_step > max_step) {
                        dbg<<"abs_step at m = "<<m<<" = "<<abs_step<<std::endl;
                        max_step = abs_step;
                    }
                }
                dbg<<"iter "<<iter<<": max step = "<<max_step<<std::endl;
                if (max_step < 1.e-12) break;

                if (iter == MAX_ITER-1) {
                    not_converged = true;
                }
            }
            x += n1;
            y += n1;
            u += n1;
            v += n1;
            n -= n1;
        }
        if (not_converged) {
            // Check which solutions are not close to the right answer.
            // Note: this is different than the max_step test above, but it shouldn't matter
            //       much since dudx, dvdy are near unity, so du,dv are nearly equal to dx,dy.
            std::vector<int> bad_indices;
            u = u0;  // Reset back to the original values.
            v = v0;
            x = x0;
            y = y0;
            n = n0;
            int m0 = 0;
            while (n) {
                dbg<<"n = "<<n<<std::endl;
                int n1 = std::min(n, 256);
                Horner2D(x, y, n1, A, nab, nab, du, temp);  // u' = Horner2D(x, y, A)
                Horner2D(x, y, n1, B, nab, nab, dv, temp);  // v' = Horner2D(x, y, B)
                for(int m=0; m<n1; ++m) {
                    du[m] -= u[m];
                    dv[m] -= v[m];
                    double abs_err = std::max(std::abs(du[m])/std::max(1.,std::abs(u[m])),
                                              std::abs(dv[m])/std::max(1.,std::abs(v[m])));
                    if (abs_err > 1.e-12) bad_indices.push_back(m + m0);
                }
                x += n1;
                y += n1;
                u += n1;
                v += n1;
                n -= n1;
                m0 += n1;
            }
            if (bad_indices.size() > 0) {
                std::ostringstream oss;
                oss << "Unable to solve for image_pos (max iter reached) ";
                oss << "for the following indices: [";
                for (size_t i=0; i<bad_indices.size(); ++i) oss << bad_indices[i] << ",";
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }
    }

}
