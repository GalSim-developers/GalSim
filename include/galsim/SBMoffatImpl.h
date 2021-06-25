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

#ifndef GalSim_SBMoffatImpl_H
#define GalSim_SBMoffatImpl_H

#include "SBProfileImpl.h"
#include "SBMoffat.h"
#include "Table.h"
#include "OneDimensionalDeviate.h"

namespace galsim {

    class SBMoffat::SBMoffatImpl : public SBProfileImpl
    {
    public:
        SBMoffatImpl(double beta, double scale_radius, double trunc, double flux,
                     const GSParams& gsparams);

        ~SBMoffatImpl() {}

        double xValue(const Position<double>& p) const;

        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; }
        bool hasHardEdges() const { return (1.-_fluxFactor) > this->gsparams.maxk_threshold; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const
        { xmin = -_maxR; xmax = _maxR; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const
        { ymin = -_maxR; ymax = _maxR; }

        void getYRangeX(double x, double& ymin, double& ymax, std::vector<double>& ) const
        {
            if (std::abs(x) >= _maxR) { ymin = 0; ymax = 0; }
            else { ymax = sqrt(_maxR_sq - x*x); ymin = -ymax; }
        }

        Position<double> centroid() const
        { return Position<double>(0., 0.); }


        double getFlux() const { return _flux; }
        double maxSB() const { return _norm; }

        /**
         * @brief Moffat photon shooting is done by analytic inversion of cumulative flux
         * distribution.
         *
         * Will require 2 uniform deviates per photon, plus analytic function (pow and sqrt)
         *
         * @param[in] photons PhotonArray in which to write the photon information
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         */
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        double getBeta() const { return _beta; }
        double getScaleRadius() const { return _rD; }
        double getTrunc() const { return _trunc; }
        double getHalfLightRadius() const;
        double getFWHM() const;

        // Overrides for better efficiency
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const;
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const;
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const;

    private:
        double _beta; ///< Moffat beta parameter for profile `[1 + (r / rD)^2]^beta`.
        double _flux; ///< Flux.
        double _norm; ///< Normalization. (Including the flux)
        double _knorm; ///< Normalization for kValue. (Including the flux)
        double _rD;   ///< Scale radius for profile `[1 + (r / rD)^2]^beta`.
        double _rD_sq;
        double _inv_rD;
        double _inv_rD_sq;
        double _maxR; ///< Maximum `r`
        double _maxRrD; ///< maxR/rD
        double _trunc;  ///< Outer truncation radius in same physical units as `_rD`
        double _fluxFactor; ///< Integral of total flux in terms of 'rD' units.
        double _maxRrD_sq;
        double _maxR_sq;

        mutable TableBuilder _ft;  ///< Lookup table for Fourier transform of Moffat.

        mutable double _stepk;
        mutable double _maxk; ///< Maximum k with kValue > 1.e-3

        double (*_pow_mbeta)(double x, double beta);
        double (SBMoffatImpl::*_kV)(double ksq) const;

        /// Setup the FT Table.
        void setupFT() const;

        // These are the (unnormalized) kValue functions for untruncated Moffats
        double kV_15(double ksq) const;
        double kV_2(double ksq) const;
        double kV_25(double ksq) const;
        double kV_3(double ksq) const;
        double kV_35(double ksq) const;
        double kV_4(double ksq) const;
        double kV_gen(double ksq) const;

        // This does the truncated case.
        double kV_trunc(double ksq) const;

        // pow(x,beta) for special (probably not uncommon) cases.
        static double pow_1(double x, double );
        static double pow_15(double x, double );
        static double pow_2(double x, double );
        static double pow_25(double x, double );
        static double pow_3(double x, double );
        static double pow_35(double x, double );
        static double pow_4(double x, double );
        static double pow_gen(double x, double beta);

        void doFillXImage(ImageView<double> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<double> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, int izero,
                          double y0, double dy, int jzero) const
        { fillXImage(im,x0,dx,izero,y0,dy,jzero); }
        void doFillXImage(ImageView<float> im,
                          double x0, double dx, double dxy,
                          double y0, double dy, double dyx) const
        { fillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<double> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, int izero,
                          double ky0, double dky, int jzero) const
        { fillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        void doFillKImage(ImageView<std::complex<float> > im,
                          double kx0, double dkx, double dkxy,
                          double ky0, double dky, double dkyx) const
        { fillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

        // Copy constructor and op= are undefined.
        SBMoffatImpl(const SBMoffatImpl& rhs);
        void operator=(const SBMoffatImpl& rhs);
    };

}

#endif
