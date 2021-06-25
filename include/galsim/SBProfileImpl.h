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

#ifndef GalSim_SBProfileImpl_H
#define GalSim_SBProfileImpl_H

#include "SBProfile.h"
#include "integ/Int.h"

namespace galsim {

    class SBProfile::SBProfileImpl
    {
    public:

        // Constructor
        SBProfileImpl(const GSParams& _gsparams);

        // Virtual destructor
        virtual ~SBProfileImpl() {}

        // Pure virtual functions:
        virtual double xValue(const Position<double>& p) const =0;
        virtual std::complex<double> kValue(const Position<double>& k) const =0;

        // Calculate xValues and kValues for a bunch of positions at once.
        // For some profiles, this may be more efficient than repeated calls of xValue(pos)
        // since it affords the opportunity for vectorization of the calculations.
        //
        // For the first two versions, the x,y values for val(ix,iy) are
        //     x = x0 + ix dx
        //     y = y0 + iy dy
        // The izero, jzero values are the indices where x=0, y=0.
        // For some profiles (e.g. axi-symmetric profiles), this affords further opportunities
        // for optimization.  If there is no such index, then izero, jzero = 0, which indicates
        // that all the values need to be used.
        //
        // For the latter two versions, the x,y values for val(ix,iy) are
        //     x = x0 + ix dx + iy dxy
        //     y = y0 + iy dy + ix dyx
        //
        // If these aren't overridden, then the regular xValue or kValue will be called for each
        // position.
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, int izero,
                        double y0, double dy, int jzero) const
        // This is a C++ workaround for the fact that templates can't be virtual.
        { doFillXImage(im,x0,dx,izero,y0,dy,jzero); }
        template <typename T>
        void fillXImage(ImageView<T> im,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const
        { doFillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, int izero,
                        double ky0, double dky, int jzero) const
        { doFillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        template <typename T>
        void fillKImage(ImageView<std::complex<T> > im,
                        double kx0, double dkx, double dkxy,
                        double ky0, double dky, double dkyx) const
        { doFillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

        template <typename T>
        void defaultFillXImage(ImageView<T> im,
                               double x0, double dx, int izero,
                               double y0, double dy, int jzero) const;
        template <typename T>
        void defaultFillXImage(ImageView<T> im,
                               double x0, double dx, double dxy,
                               double y0, double dy, double dyx) const;
        template <typename T>
        void defaultFillKImage(ImageView<std::complex<T> > im,
                               double kx0, double dkx, int izero,
                               double ky0, double dky, int jzero) const;
        template <typename T>
        void defaultFillKImage(ImageView<std::complex<T> > im,
                               double kx0, double dkx, double dkxy,
                               double ky0, double dky, double dkyx) const;

        virtual double maxK() const =0;
        virtual double stepK() const =0;
        virtual bool isAxisymmetric() const =0;
        virtual bool hasHardEdges() const =0;
        virtual bool isAnalyticX() const =0;
        virtual bool isAnalyticK() const =0;
        virtual Position<double> centroid() const = 0;
        virtual double getFlux() const =0;
        virtual double maxSB() const =0;
        virtual void shoot(PhotonArray& photons, UniformDeviate ud) const=0;

        // Functions with default implementations:
        virtual void getXRange(double& xmin, double& xmax, std::vector<double>& /*splits*/) const
        { xmin = -integ::MOCK_INF; xmax = integ::MOCK_INF; }

        virtual void getYRange(double& ymin, double& ymax, std::vector<double>& /*splits*/) const
        { ymin = -integ::MOCK_INF; ymax = integ::MOCK_INF; }

        virtual void getYRangeX(
            double /*x*/, double& ymin, double& ymax, std::vector<double>& splits) const
        { getYRange(ymin,ymax,splits); }

        virtual double getPositiveFlux() const { return getFlux()>0. ? getFlux() : 0.; }

        virtual double getNegativeFlux() const { return getFlux()>0. ? 0. : -getFlux(); }

        // Public so it can be directly used from SBProfile.
        GSParams gsparams;

    protected:

        // A helper function for cases where the profile has f(x,y) = f(|x|,|y|).
        // This includes axisymmetric profiles, but also a few other cases.
        // Only one quadrant has its values computed.  Then these values are copied to the other
        // 3 quadrants.  The input values izero, jzero are the index of x=0, y=0.
        // At least one of these needs to be != 0.
        template <typename T>
        void fillXImageQuadrant(ImageView<T> im,
                                double x0, double dx, int m1,
                                double y0, double dy, int n1) const;
        template <typename T>
        void fillKImageQuadrant(ImageView<std::complex<T> > im,
                                double kx0, double dkx, int m1,
                                double ky0, double dky, int n1) const;

        // These need to be overridden by any class that wants to use its own implementation
        // of fillXImage or fillKImage.
        virtual void doFillXImage(ImageView<double> im,
                                  double x0, double dx, int izero,
                                  double y0, double dy, int jzero) const
        { defaultFillXImage(im,x0,dx,izero,y0,dy,jzero); }
        virtual void doFillXImage(ImageView<double> im,
                                  double x0, double dx, double dxy,
                                  double y0, double dy, double dyx) const
        { defaultFillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        virtual void doFillXImage(ImageView<float> im,
                                  double x0, double dx, int izero,
                                  double y0, double dy, int jzero) const
        { defaultFillXImage(im,x0,dx,izero,y0,dy,jzero); }
        virtual void doFillXImage(ImageView<float> im,
                                  double x0, double dx, double dxy,
                                  double y0, double dy, double dyx) const
        { defaultFillXImage(im,x0,dx,dxy,y0,dy,dyx); }
        virtual void doFillKImage(ImageView<std::complex<double> > im,
                                  double kx0, double dkx, int izero,
                                  double ky0, double dky, int jzero) const
        { defaultFillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        virtual void doFillKImage(ImageView<std::complex<double> > im,
                                  double kx0, double dkx, double dkxy,
                                  double ky0, double dky, double dkyx) const
        { defaultFillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }
        virtual void doFillKImage(ImageView<std::complex<float> > im,
                                  double kx0, double dkx, int izero,
                                  double ky0, double dky, int jzero) const
        { defaultFillKImage(im,kx0,dkx,izero,ky0,dky,jzero); }
        virtual void doFillKImage(ImageView<std::complex<float> > im,
                                  double kx0, double dkx, double dkxy,
                                  double ky0, double dky, double dkyx) const
        { defaultFillKImage(im,kx0,dkx,dkxy,ky0,dky,dkyx); }

    private:
        // Copy constructor and op= are undefined.
        SBProfileImpl(const SBProfileImpl& rhs);
        void operator=(const SBProfileImpl& rhs);
    };

    // Some helper functions that some Profiles use to speed up the calculations.

    // Get the range i1 <= i < i2 where (kx0 + i dkx)^2 + ky^2 <= ksqmax
    // Note: this calculates kysq along the way, so it is a reference.
    void GetKValueRange1d(int& i1, int& i2, int m, double kmax, double ksqmax,
                          double kx0, double dkx, double ky, double& kysq);

    // Get the range i1 <= i < i2 where (kx0 + i dkx)^2 + (ky0 + i dky)^2 <= ksqmax
    void GetKValueRange2d(int& i1, int& i2, int m, double kmax, double ksqmax,
                          double kx0, double dkx, double ky0, double dky);

}

#endif
