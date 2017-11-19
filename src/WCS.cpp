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

//#define DEBUGLOGGING

#include "Std.h"
#include "WCS.h"
#include "TMV.h"

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

    void setup_pow(tmv::VectorView<double>& x, tmv::Matrix<double>& xpow)
    {
        xpow.col(0).setAllTo(1.);
        xpow.col(1) = x;
        for (int i=2; i<xpow.rowsize(); ++i)
            xpow.col(i) = ElemProd(xpow.col(i-1), x);
    }

    void ApplyPV(int n, const int m, double* uar, double* var, const double* pvar)
    {
        tmv::ConstMatrixView<double> pvu(pvar, m, m, m, 1, tmv::NonConj);
        tmv::ConstMatrixView<double> pvv(pvar + m*m, m, m, m, 1, tmv::NonConj);
        while (n) {
            // Do this in blocks of at most 256 to avoid blowing up the memory usage when
            // this is run on a large image. It's also a bit faster this way, since there
            // are fewer cache misses.
            const int nn = n >= 256 ? 256 : n;
            tmv::VectorView<double> u(uar, nn, 1, tmv::NonConj);
            tmv::VectorView<double> v(var, nn, 1, tmv::NonConj);

            tmv::Matrix<double> upow(nn,m);
            tmv::Matrix<double> vpow(nn,m);
            setup_pow(u, upow);
            setup_pow(v, vpow);

            // If we only have one input position, then the new values of u,v are
            //
            //     u' = [ 1 u u^2 u^3 ] pvu [ 1 v v^2 v^3 ]^T
            //     v' = [ 1 u u^2 u^3 ] pvv [ 1 v v^2 v^3 ]^T
            //
            // When there are multiple inputs, then upow and vpow are each Nx4 matrices.
            // The values we want are the diagonal of the matrix you would get from the
            // above formulae.  So we use the fact that
            //     diag(AT . B) = sum_rows(A * B)

            tmv::Vector<double> ones(m, 1.);
            tmv::Matrix<double> temp = vpow * pvu.transpose();
            temp = ElemProd(upow, temp);
            u = temp * ones;

            temp = vpow * pvv.transpose();
            temp = ElemProd(upow, temp);
            v = temp * ones;

            uar += nn;
            var += nn;
            n -= nn;
        }
    }

    void InvertPV(double& u, double& v, const double* pvar)
    {
        // Let (u0,v0) be the current value of (u,v).  Then we want to find a new (u,v) such that
        //
        //       [ u0 v0 ] = [ 1 u u^2 u^3 ] pv [ 1 v v^2 v^3 ]^T
        //
        // Start with (u,v) = (u0,v0)
        //
        // Then use Newton-Raphson iteration to improve (u,v).  This is extremely fast
        // for typical PV distortions, since the distortions are generally very small.
        // Newton-Raphson doubles the number of significant digits in each iteration.

        const int MAX_ITER = 10;
        const double TOL = 1.e-6 / 3600.;    // 1.e-6 arcsec.  pv always uses degrees units

        double u0 = u;
        double v0 = v;
        tmv::ConstMatrixView<double> pvu(pvar, 4, 4, 4, 1, tmv::NonConj);
        tmv::ConstMatrixView<double> pvv(pvar + 16, 4, 4, 4, 1, tmv::NonConj);

        // Some temporary vectors/matrices we'll use within the loop below.
        tmv::SmallVector<double,4> upow;
        tmv::SmallVector<double,4> vpow;
        tmv::SmallVector<double,4> pvu_vpow;
        tmv::SmallVector<double,4> pvv_vpow;
        tmv::SmallVector<double,4> dupow;
        tmv::SmallVector<double,4> dvpow;
        tmv::SmallMatrix<double,2,2> j1;
        tmv::SmallVector<double,2> diff;
        tmv::SmallVector<double,2> duv;

        double prev_err = -1.;
        for (int iter=0; iter<MAX_ITER; ++iter) {
            double usq = u*u;
            double vsq = v*v;
            upow << 1., u, usq, usq*u;
            vpow << 1., v, vsq, vsq*v;

            pvu_vpow = pvu * vpow;
            pvv_vpow = pvv * vpow;
            double du = upow * pvu_vpow - u0;
            double dv = upow * pvv_vpow - v0;

            // Check that things are improving...
            double err = std::max(std::abs(du),std::abs(dv));

            if (prev_err >= 0. && err > prev_err)
                throw std::runtime_error("Unable to solve for image_pos (not improving)");
            prev_err = err;

            // If we are below tolerance, return this value
            if (err < TOL) return;
            else {
                dupow << 0., 1., 2.*u, 3.*usq;
                dvpow << 0., 1., 2.*v, 3.*vsq;
                j1 << dupow * pvu_vpow,  upow * pvu * dvpow,
                      dupow * pvv_vpow,  upow * pvv * dvpow;
                diff << du, dv;
                duv = diff / j1;
                u -= duv[0];
                v -= duv[1];
                // If we're hitting the limits of double precision, stop iterating.
                if (std::abs(duv[0]/u) < 1.e-15 || std::abs(duv[1]/v) < 1.e-15) return;
            }
        }

        throw std::runtime_error("Unable to solve for image_pos (max iter reached)");
    }

    void setup_pow(double x, tmv::Vector<double>& xpow)
    {
        xpow[1] = x;
        for (int i=2; i<xpow.size(); ++i) xpow[i] = xpow[i-1] * x;
    }

    void InvertAB(int m, double& x, double& y, const double* abar, const double* abpar)
    {
        dbg<<"start invert_ab: "<<x<<" "<<y<<std::endl;
        double x0 = x;
        double y0 = y;

        tmv::ConstMatrixView<double> abx(abar, m, m, m, 1, tmv::NonConj);
        tmv::ConstMatrixView<double> aby(abar + m*m, m, m, m, 1, tmv::NonConj);
        dbg<<"abx = "<<abx<<std::endl;
        dbg<<"aby = "<<aby<<std::endl;

        tmv::Vector<double> xpow(m,1.);
        tmv::Vector<double> ypow(m,1.);
        tmv::Vector<double> abx_ypow(m);
        tmv::Vector<double> aby_ypow(m);
        tmv::Vector<double> dxpow(m,0.);
        tmv::Vector<double> dypow(m,0.);
        tmv::SmallMatrix<double,2,2> j1;
        tmv::SmallVector<double,2> diff;
        tmv::SmallVector<double,2> dxy;

        if (abpar) {
            setup_pow(x, xpow);
            setup_pow(y, ypow);
            tmv::ConstMatrixView<double> abpx(abpar, m, m, m, 1, tmv::NonConj);
            tmv::ConstMatrixView<double> abpy(abpar + m*m, m, m, m, 1, tmv::NonConj);
            abx_ypow = abpx * ypow;
            aby_ypow = abpy * ypow;
            x += xpow * abx_ypow;
            y += xpow * aby_ypow;
        }

        // We do this iteration even if we have AP and BP matrices, since the inverse
        // transformation is not always very accurate.
        // The assumption here is that the A and B matrices are correct and the AP and BP
        // matrices are estimated from them, and thus are approximate at some level.
        // Of course, in reality the A and B matrices are also approximate, but at least this
        // way the WCS is consistent transforming in the two directions.
        const int MAX_ITER = 10;
        const double TOL = 1.e-6 / 3600.;

        double prev_err = -1.;
        for (int iter=0; iter<MAX_ITER; ++iter) {
            dbg<<iter<<" x,y = "<<x<<" "<<y<<std::endl;
            setup_pow(x, xpow);
            setup_pow(y, ypow);

            abx_ypow = abx * ypow;
            aby_ypow = aby * ypow;
            double dx = xpow * abx_ypow + x - x0;
            double dy = xpow * aby_ypow + y - y0;
            dbg<<"diff = "<<dx<<" "<<dy<<std::endl;

            // Check that things are improving...
            double err = std::max(std::abs(dx),std::abs(dy));
            if (prev_err >= 0. && err > prev_err) 
                throw std::runtime_error("Unable to solve for image_pos (not improving)");
            prev_err = err;
            dbg<<"err = "<<err<<std::endl;

            // If we are below tolerance, return this value
            if (err < TOL) return;
            else {
                for(int i=1; i<m; ++i) dxpow[i] = i * xpow[i-1];
                for(int i=1; i<m; ++i) dypow[i] = i * ypow[i-1];

                j1 << dxpow * abx_ypow,  xpow * abx * dypow,
                      dxpow * aby_ypow,  xpow * aby * dypow;
                j1.diag().addToAll(1.);
                dbg<<"j1 = "<<j1<<std::endl;
                diff << dx, dy;
                dxy = diff / j1;
                dbg<<"dxy = "<<dxy<<std::endl;
                x -= dxy[0];
                y -= dxy[1];
                // If we're hitting the limits of double precision, stop iterating.
                if (std::abs(dxy[0]/x) < 1.e-15 && std::abs(dxy[1]/y) < 1.e-15) return;
            }
        }

        throw std::runtime_error("Unable to solve for image_pos (max iter reached)");
    }
}
