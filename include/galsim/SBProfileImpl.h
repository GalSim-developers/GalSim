// -*- c++ -*-
/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */

#ifndef SBPROFILE_IMPL_H
#define SBPROFILE_IMPL_H

#include "SBProfile.h"
#include "FFT.h"
#include "integ/Int.h"
#include "TMV.h"

namespace galsim {

    class SBProfile::SBProfileImpl
    {
    public:

        // Constructor 
        SBProfileImpl(boost::shared_ptr<GSParams> _gsparams);

        // Virtual destructor
        virtual ~SBProfileImpl() {}

        // Pure virtual functions:
        virtual double xValue(const Position<double>& p) const =0;
        virtual std::complex<double> kValue(const Position<double>& k) const =0; 

        // Caclulate xValues and kValues for a bunch of positions at once.
        // For some profiles, this may be more efficient than repeated calls of xValue(pos)
        // since it affords the opportunity for vectorization of the calculations.
        //
        // For the first two versions, the x,y values for val(ix,iy) are
        //     x = x0 + ix dx 
        //     y = y0 + iy dy
        // The ix_zero, iy_zero values are the indices where x=0, y=0.
        // For some profiles (e.g. axi-symmetric profiles), this affords further opportunities
        // for optimization.  If there is no such index, then ix_zero, iy_zero = 0, which indicates 
        // that all the values need to be used.
        //
        // For the latter two versions, the x,y values for val(ix,iy) are
        //     x = x0 + ix dx + iy dxy
        //     y = y0 + iy dy + ix dyx
        //
        // If these aren't overridden, then the regular xValue or kValue will be called for each 
        // position.
        virtual void fillXValue(tmv::MatrixView<double> val,
                                double x0, double dx, int ix_zero,
                                double y0, double dy, int iy_zero) const;
        virtual void fillXValue(tmv::MatrixView<double> val,
                                double x0, double dx, double dxy,
                                double y0, double dy, double dyx) const;
        virtual void fillKValue(tmv::MatrixView<std::complex<double> > val,
                                double x0, double dx, int ix_zero,
                                double y0, double dy, int iy_zero) const;
        virtual void fillKValue(tmv::MatrixView<std::complex<double> > val,
                                double x0, double dx, double dxy,
                                double y0, double dy, double dyx) const;

        virtual double maxK() const =0; 
        virtual double stepK() const =0;
        virtual bool isAxisymmetric() const =0;
        virtual bool hasHardEdges() const =0;
        virtual bool isAnalyticX() const =0; 
        virtual bool isAnalyticK() const =0; 
        virtual Position<double> centroid() const = 0;
        virtual double getFlux() const =0; 
        virtual boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const=0;

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

        // Utility for drawing into Image data structures.
        // returns flux integral
        template <typename T>
        double fillXImage(ImageView<T>& image, double gain) const;

        // Utility for drawing a k grid into FFT data structures 
        void fillKGrid(KTable& kt) const;

        // Utility for drawing an x grid into FFT data structures 
        void fillXGrid(XTable& xt) const;

        // Public so it can be directly used from SBProfile.
        boost::shared_ptr<GSParams> gsparams;

    protected:

        // A helper function for cases where the profile has f(x,y) = f(|x|,|y|).
        // This includes axisymmetric profiles, but also a few other cases.
        // Only one quadrant has its values computed.  Then these values are copied to the other
        // 3 quadrants.  The input values ix_zero, iy_zero are the index of x=0, y=0.
        // At least one of these needs to be != 0.
        void fillXValueQuadrant(tmv::MatrixView<double> val,
                                double x0, double dx, int nx1,
                                double y0, double dy, int ny1) const;
        void fillKValueQuadrant(tmv::MatrixView<std::complex<double> > val,
                                double x0, double dx, int nx1,
                                double y0, double dy, int ny1) const;

    private:
        // Copy constructor and op= are undefined.
        SBProfileImpl(const SBProfileImpl& rhs);
        void operator=(const SBProfileImpl& rhs);

        // Default GSParams to use when input is None
        static boost::shared_ptr<GSParams> default_gsparams;
    };

}

#endif // SBPROFILE_IMPL_H

