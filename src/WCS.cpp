/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
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
#ifdef USE_TMV
#include "TMV.h"
typedef tmv::Vector<double> VectorXd;
typedef tmv::Matrix<double> MatrixXd;
typedef tmv::VectorView<double> MapVectorXd;
#else
#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
#include "Eigen/Dense"
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<Eigen::VectorXd> MapVectorXd;
#endif

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

    void setup_pow(MapVectorXd& x, MatrixXd& xpow)
    {
#ifdef USE_TMV
        xpow.col(0).setAllTo(1.);
        xpow.col(1) = x;
        for (int i=2; i<xpow.rowsize(); ++i)
            xpow.col(i) = ElemProd(xpow.col(i-1), x);
#else
        xpow.col(0).setConstant(1.);
        xpow.col(1) = x;
        for (int i=2; i<xpow.cols(); ++i)
            xpow.col(i).array() = xpow.col(i-1).array() * x.array();
#endif
    }

    void ApplyPV(int n, const int m, double* uar, double* var, const double* pvar)
    {
#ifdef USE_TMV
        tmv::ConstMatrixView<double> pvuT(pvar, m, m, 1, m, tmv::NonConj);
        tmv::ConstMatrixView<double> pvvT(pvar + m*m, m, m, 1, m, tmv::NonConj);
#else
        Eigen::Map<const Eigen::MatrixXd> pvuT(pvar, m, m);
        Eigen::Map<const Eigen::MatrixXd> pvvT(pvar + m*m, m, m);
#endif
        while (n) {
            // Do this in blocks of at most 256 to avoid blowing up the memory usage when
            // this is run on a large image. It's also a bit faster this way, since there
            // are fewer cache misses.
            const int nn = n >= 256 ? 256 : n;

#ifdef USE_TMV
            MapVectorXd u(uar, nn, 1, tmv::NonConj);
            MapVectorXd v(var, nn, 1, tmv::NonConj);
#else
            MapVectorXd u(uar, nn);
            MapVectorXd v(var, nn);
#endif
            MatrixXd upow(nn, m);
            MatrixXd vpow(nn, m);

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

#ifdef USE_TMV
            VectorXd ones(m, 1.);
#else
            VectorXd ones = Eigen::VectorXd::Ones(m);
#endif
            MatrixXd temp = vpow * pvuT;
#ifdef USE_TMV
            temp = ElemProd(upow, temp);
#else
            temp.array() *= upow.array();
#endif
            u = temp * ones;

            temp = vpow * pvvT;
#ifdef USE_TMV
            temp = ElemProd(upow, temp);
#else
            temp.array() *= upow.array();
#endif
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
#ifdef USE_TMV
        tmv::ConstMatrixView<double> pvuT(pvar, 4, 4, 1, 4, tmv::NonConj);
        tmv::ConstMatrixView<double> pvvT(pvar + 16, 4, 4, 1, 4, tmv::NonConj);
        typedef tmv::SmallVector<double,4> Vector4d;
        typedef tmv::SmallMatrix<double,2,2> Matrix2d;
        typedef tmv::SmallVector<double,2> Vector2d;
#else
        Eigen::Map<const Eigen::Matrix4d> pvuT(pvar);
        Eigen::Map<const Eigen::Matrix4d> pvvT(pvar + 16);
        using Eigen::Vector4d;
        using Eigen::Matrix2d;
        using Eigen::Vector2d;
#endif

        // Some temporary vectors/matrices we'll use within the loop below.
        Vector4d upow;
        Vector4d vpow;
        Vector4d pvu_vpow;
        Vector4d pvv_vpow;
        Vector4d dupow;
        Vector4d dvpow;
        Matrix2d j1;
        Vector2d diff;
        Vector2d duv;

        double prev_err = -1.;
        for (int iter=0; iter<MAX_ITER; ++iter) {
            double usq = u*u;
            double vsq = v*v;
            upow << 1., u, usq, usq*u;
            vpow << 1., v, vsq, vsq*v;

            pvu_vpow = pvuT.transpose() * vpow;
            pvv_vpow = pvvT.transpose() * vpow;
#ifdef USE_TMV
            double du = upow * pvu_vpow - u0;
            double dv = upow * pvv_vpow - v0;
#else
            double du = upow.dot(pvu_vpow) - u0;
            double dv = upow.dot(pvv_vpow) - v0;
#endif

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
#ifdef USE_TMV
                j1 << dupow * pvu_vpow,  upow * (pvuT.transpose() * dvpow),
                      dupow * pvv_vpow,  upow * (pvvT.transpose() * dvpow);
#else
                j1 << dupow.dot(pvu_vpow),  upow.dot(pvuT.transpose() * dvpow),
                      dupow.dot(pvv_vpow),  upow.dot(pvvT.transpose() * dvpow);
#endif
                diff << du, dv;
#ifdef USE_TMV
                duv = diff / j1;
#else
                duv = j1.lu().solve(diff);
#endif
                u -= duv[0];
                v -= duv[1];
                // If we're hitting the limits of double precision, stop iterating.
                if (std::abs(duv[0]/u) < 1.e-15 || std::abs(duv[1]/v) < 1.e-15) return;
            }
        }

        throw std::runtime_error("Unable to solve for image_pos (max iter reached)");
    }

    void setup_pow(double x, VectorXd& xpow)
    {
        xpow[0] = 1.;
        xpow[1] = x;
        for (int i=2; i<xpow.size(); ++i) xpow[i] = xpow[i-1] * x;
    }

    void InvertAB(int m, double& x, double& y, const double* abar, const double* abpar)
    {
        dbg<<"start invert_ab: "<<x<<" "<<y<<std::endl;
        double x0 = x;
        double y0 = y;

#ifdef USE_TMV
        tmv::ConstMatrixView<double> abxT(abar, m, m, 1, m, tmv::NonConj);
        tmv::ConstMatrixView<double> abyT(abar + m*m, m, m, 1, m, tmv::NonConj);
        typedef tmv::SmallMatrix<double,2,2> Matrix2d;
        typedef tmv::SmallVector<double,2> Vector2d;
#else
        Eigen::Map<const Eigen::MatrixXd> abxT(abar, m, m);
        Eigen::Map<const Eigen::MatrixXd> abyT(abar + m*m, m, m);
        using Eigen::Matrix2d;
        using Eigen::Vector2d;
#endif
        dbg<<"abx = "<<abxT.transpose()<<std::endl;
        dbg<<"aby = "<<abyT.transpose()<<std::endl;

        VectorXd xpow(m);
        VectorXd ypow(m);
        VectorXd abx_ypow(m);
        VectorXd aby_ypow(m);
        VectorXd dxpow(m); dxpow.setZero();
        VectorXd dypow(m); dypow.setZero();
        Matrix2d j1;
        Vector2d diff;
        Vector2d dxy;

        if (abpar) {
            setup_pow(x, xpow);
            setup_pow(y, ypow);
#ifdef USE_TMV
            tmv::ConstMatrixView<double> abpxT(abpar, m, m, 1, m, tmv::NonConj);
            tmv::ConstMatrixView<double> abpyT(abpar + m*m, m, m, 1, m, tmv::NonConj);
#else
            Eigen::Map<const Eigen::MatrixXd> abpxT(abpar, m, m);
            Eigen::Map<const Eigen::MatrixXd> abpyT(abpar + m*m, m, m);
#endif
            abx_ypow = abpxT.transpose() * ypow;
            aby_ypow = abpyT.transpose() * ypow;
#ifdef USE_TMV
            x += xpow * abx_ypow;
            y += xpow * aby_ypow;
#else
            x += xpow.dot(abx_ypow);
            y += xpow.dot(aby_ypow);
#endif
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

            abx_ypow = abxT.transpose() * ypow;
            aby_ypow = abyT.transpose() * ypow;
#ifdef USE_TMV
            double dx = xpow * abx_ypow + x - x0;
            double dy = xpow * aby_ypow + y - y0;
#else
            double dx = xpow.dot(abx_ypow) + x - x0;
            double dy = xpow.dot(aby_ypow) + y - y0;
#endif
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

#ifdef USE_TMV
                j1 << dxpow * abx_ypow,  xpow * abxT.transpose() * dypow,
                      dxpow * aby_ypow,  xpow * abyT.transpose() * dypow;
                j1.diag().addToAll(1.);
#else
                j1 << dxpow.dot(abx_ypow),  xpow.dot(abxT.transpose() * dypow),
                      dxpow.dot(aby_ypow),  xpow.dot(abyT.transpose() * dypow);
                j1.diagonal().array() += 1.;
#endif
                dbg<<"j1 = "<<j1<<std::endl;
                diff << dx, dy;
#ifdef USE_TMV
                dxy = diff / j1;
#else
                dxy = j1.lu().solve(diff);
#endif
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
