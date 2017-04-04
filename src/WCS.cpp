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

    void setup_pow(double x, double y, tmv::Vector<double>& xpow, tmv::Vector<double>& ypow)
    {
        xpow[1] = x;
        ypow[1] = y;
        for (size_t i=2; i<xpow.size(); ++i) xpow[i] = xpow[i-1] * x;
        for (size_t i=2; i<xpow.size(); ++i) ypow[i] = ypow[i-1] * y;
    }

    void InvertAB(double& x, double& y, const double* abar, int order, const double* abpar)
    {
        dbg<<"start invert_ab: "<<x<<" "<<y<<std::endl;
        dbg<<"order = "<<order<<std::endl;
        double x0 = x;
        double y0 = y;

        const int op1 = order+1;
        tmv::ConstMatrixView<double> abx(abar, op1, op1, op1, 1, tmv::NonConj);
        tmv::ConstMatrixView<double> aby(abar + op1*op1, op1, op1, op1, 1, tmv::NonConj);
        dbg<<"abx = "<<abx<<std::endl;
        dbg<<"aby = "<<aby<<std::endl;

        tmv::Vector<double> xpow(op1,1.);
        tmv::Vector<double> ypow(op1,1.);
        tmv::Vector<double> abx_ypow(op1);
        tmv::Vector<double> aby_ypow(op1);
        tmv::Vector<double> dxpow(op1,0.);
        tmv::Vector<double> dypow(op1,0.);
        tmv::SmallMatrix<double,2,2> j1;
        tmv::SmallVector<double,2> diff;
        tmv::SmallVector<double,2> dxy;

        if (abpar) {
            setup_pow(x, y, xpow, ypow);
            tmv::ConstMatrixView<double> abpx(abpar, op1, op1, op1, 1, tmv::NonConj);
            tmv::ConstMatrixView<double> abpy(abpar + op1*op1, op1, op1, op1, 1, tmv::NonConj);
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
            setup_pow(x, y, xpow, ypow);

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
                for(int i=1; i<=order; ++i) dxpow[i] = i * xpow[i-1];
                for(int i=1; i<=order; ++i) dypow[i] = i * ypow[i-1];

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
