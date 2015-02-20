/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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

#ifndef GalSim_SBBoxImpl_H
#define GalSim_SBBoxImpl_H

#include "SBProfileImpl.h"
#include "SBBox.h"

namespace galsim {

    class SBBox::SBBoxImpl : public SBProfileImpl 
    {
    public:
        SBBoxImpl(double width, double height, double flux, const GSParamsPtr& gsparams);
        ~SBBoxImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return false; } 
        bool hasHardEdges() const { return true; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_wo2;  xmax = _wo2; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_ho2;  ymax = _ho2; }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        double getWidth() const { return _width; }
        double getHeight() const { return _height; }

        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        // Override both for efficiency and to put in fractional edge values which
        // don't happen with normal calls to xValue.
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        // Overrides for better efficiency
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _width;
        double _height;
        double _flux;
        double _norm; // Calculated value: flux / (width*height)
        double _wo2;
        double _ho2;
        double _wo2pi;
        double _ho2pi;

        // Copy constructor and op= are undefined.
        SBBoxImpl(const SBBoxImpl& rhs);
        void operator=(const SBBoxImpl& rhs);
    };

    class SBTopHat::SBTopHatImpl : public SBProfileImpl 
    {
    public:
        SBTopHatImpl(double radius, double flux, const GSParamsPtr& gsparams);
        ~SBTopHatImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return true; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        void getXRange(double& xmin, double& xmax, std::vector<double>& ) const 
        { xmin = -_r0;  xmax = _r0; }

        void getYRange(double& ymin, double& ymax, std::vector<double>& ) const 
        { ymin = -_r0;  ymax = _r0; }

        void getYRangeX(
            double x, double& ymin, double& ymax, std::vector<double>& ) const
        { 
            ymax = sqrt(_r0*_r0 - x*x);
            ymin = -ymax;
        }

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        double getRadius() const { return _r0; }

        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        // Override both for efficiency and to put in fractional edge values which
        // don't happen with normal calls to xValue.
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        // Overrides for better efficiency
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _r0;
        double _r0sq;
        double _flux;
        double _norm;

        // A helper function that calculates kValue given (k r0)^2
        std::complex<double> kValue2(double kr0sq) const;

        // Copy constructor and op= are undefined.
        SBTopHatImpl(const SBTopHatImpl& rhs);
        void operator=(const SBTopHatImpl& rhs);
    };

}

#endif 

