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

#include "SBProfile.h"
#include "integ/Int.h"
#include "Solve.h"

// To time the real-space convolution integrals...
//#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif

#include <numeric>

namespace galsim {

    class ConvolveFunc :
        public std::binary_function<double,double,double>
    {
    public:
        ConvolveFunc(const SBProfile& p1, const SBProfile& p2, const Position<double>& pos) :
            _p1(p1), _p2(p2), _pos(pos) {}

        double operator()(double x, double y) const
        {
            xdbg<<"Convolve function for pos = "<<_pos<<" at x,y = "<<x<<','<<y<<std::endl;
            double v1 = _p1.xValue(Position<double>(x,y));
            double v2 = _p2.xValue(Position<double>(_pos.x-x,_pos.y-y));
            xdbg<<"Value = "<<v1<<" * "<<v2<<" = "<<v1*v2<<std::endl;
            return v1*v2;
        }
    private:
        const SBProfile& _p1;
        const SBProfile& _p2;
        const Position<double>& _pos;
    };

    class YRegion :
        public std::unary_function<double, integ::IntRegion<double> >
    {
    public:
        YRegion(const SBProfile& p1, const SBProfile& p2, const Position<double>& pos) :
            _p1(p1), _p2(p2), _pos(pos) {}

        integ::IntRegion<double> operator()(double x) const
        {
            xxdbg<<"Get IntRegion for pos = "<<_pos<<" at x = "<<x<<std::endl;
            // First figure out each profiles y region separately.
            double ymin1,ymax1;
            splits1.clear();
            _p1.getYRangeX(x,ymin1,ymax1,splits1);
            double ymin2,ymax2;
            splits2.clear();
            _p2.getYRangeX(_pos.x-x,ymin2,ymax2,splits2);

            // Then take the overlap relevant for the calculation:
            //     _p1.xValue(x,y) * _p2.xValue(x0-x,y0-y)
            xxdbg<<"p1's y range = "<<ymin1<<" ... "<<ymax1<<std::endl;
            xxdbg<<"p2's y range = "<<ymin2<<" ... "<<ymax2<<std::endl;
            double ymin = std::max(ymin1, _pos.y-ymax2);
            double ymax = std::min(ymax1, _pos.y-ymin2);
            xxdbg<<"Y region for x = "<<x<<" = "<<ymin<<" ... "<<ymax<<std::endl;
            if (ymax < ymin) ymax = ymin;
#ifdef DEBUGLOGGING
            std::ostream* integ_dbgout = Debugger::instance().do_level(3) ?
                &Debugger::instance().get_dbgout() : 0;
            integ::IntRegion<double> reg(ymin,ymax,integ_dbgout);
#else
            integ::IntRegion<double> reg(ymin,ymax);
#endif
            for(size_t k=0;k<splits1.size();++k) {
                double s = splits1[k];
                if (s > ymin && s < ymax) reg.addSplit(s);
            }
            for(size_t k=0;k<splits2.size();++k) {
                double s = _pos.y-splits2[k];
                if (s > ymin && s < ymax) reg.addSplit(s);
            }
            return reg;
        }
    private:
        const SBProfile& _p1;
        const SBProfile& _p2;
        const Position<double>& _pos;
        mutable std::vector<double> splits1, splits2;
    };

    // This class finds the overlap between the ymin/ymax values of two profiles.
    // For overlaps of one profile's min with the other's max, this informs how to
    // adjust the xmin/xmax values to avoid the region where the integral is trivially 0.
    // This is important, because the abrupt shift from a bunch of 0's to not is
    // hard for the integrator.  So it helps to figure this out in advance.
    // The other use of this it to see where the two ymin's or the two ymax's cross
    // each other.  This also leads to an abrupt bend in the function being integrated, so
    // it's easier if we put a split point there at the start.
    // The four cases are distinguished by a "mode" variable.
    // mode = 1 and 2 are for finding where the ranges are disjoint.
    // mode = 3 and 4 are for finding the bends.
    struct OverlapFinder
    {
        OverlapFinder(const SBProfile& p1, const SBProfile& p2, const Position<double>& pos,
                      int mode) :
            _p1(p1), _p2(p2), _pos(pos), _mode(mode)
        { assert(_mode >= 1 && _mode <= 4); }
        double operator()(double x) const
        {
            double ymin1, ymax1, ymin2, ymax2;
            splits.clear();
            _p1.getYRangeX(x,ymin1,ymax1,splits);
            _p2.getYRangeX(_pos.x-x,ymin2,ymax2,splits);
            // Note: the real ymin,ymax for p2 are _pos.y-ymax2 and _pos.y-ymin2
            ymin2 = _pos.y - ymin2;
            ymax2 = _pos.y - ymax2;
            std::swap(ymin2,ymax2);
            return
                _mode == 1 ? ymax2 - ymin1 :
                _mode == 2 ? ymax1 - ymin2 :
                _mode == 3 ? ymax2 - ymax1 :
                /*_mode == 4*/ ymin2 - ymin1;
        }

    private:
        const SBProfile& _p1;
        const SBProfile& _p2;
        const Position<double>& _pos;
        int _mode;
        mutable std::vector<double> splits;
    };

    // We pull out this segment, since we do it twice.  Once with which = true, and once
    // with which = false.
    static void UpdateXRange(const OverlapFinder& func, double& xmin, double& xmax,
                             const std::vector<double>& splits)
    {
        xdbg<<"Start UpdateXRange given xmin,xmax = "<<xmin<<','<<xmax<<std::endl;
        // Find the overlap at x = xmin:
        double yrangea = func(xmin);
        xxdbg<<"yrange at x = xmin = "<<yrangea<<std::endl;

        // Find the overlap at x = xmax:
        double yrangeb = func(xmax);
        xxdbg<<"yrange at x = xmax = "<<yrangeb<<std::endl;

        if (yrangea < 0. && yrangeb < 0.) {
            xxdbg<<"Both ends are disjoint.  Check the splits.\n";
            std::vector<double> use_splits = splits;
            if (use_splits.size() == 0) {
                xxdbg<<"No splits provided.  Use the middle instead.\n";
                use_splits.push_back( (xmin+xmax)/2. );
            }
            for (size_t k=0;k<use_splits.size();++k) {
                double xmid = use_splits[k];
                double yrangec = func(xmid);
                xxdbg<<"yrange at x = "<<xmid<<" = "<<yrangec<<std::endl;
                if (yrangec > 0.) {
                    xxdbg<<"Found a non-disjoint split\n";
                    xxdbg<<"Separately adjust both xmin and xmax by finding zero crossings.\n";
                    Solve<OverlapFinder> solver1(func,xmin,xmid);
                    solver1.setMethod(Brent);
                    double root = solver1.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    xmin = root;
                    Solve<OverlapFinder> solver2(func,xmid,xmax);
                    solver2.setMethod(Brent);
                    root = solver2.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    xmax = root;
                    return;
                }
            }
            xdbg<<"All split locations are also disjoint, so set xmin = xmax.\n";
            xmin = xmax;
        } else if (yrangea > 0. && yrangeb > 0.) {
            xdbg<<"Neither end is disjoint.  Integrate the full range\n";
        } else {
            xxdbg<<"One end is disjoint.  Find the zero crossing.\n";
            Solve<OverlapFinder> solver(func,xmin,xmax);
            solver.setMethod(Brent);
            double root = solver.root();
            xdbg<<"Found root at "<<root<<std::endl;
            if (yrangea < 0.) xmin = root;
            else xmax = root;
        }
    }

    static void AddSplitsAtBends(const OverlapFinder& func, double xmin, double xmax,
                                 std::vector<double>& splits)
    {
        xdbg<<"Start AddSplitsAtBends given xmin,xmax = "<<xmin<<','<<xmax<<std::endl;
        // Find the overlap at x = xmin:
        double yrangea = func(xmin);
        xxdbg<<"yrange at x = xmin = "<<yrangea<<std::endl;

        // Find the overlap at x = xmax:
        double yrangeb = func(xmax);
        xxdbg<<"yrange at x = xmax = "<<yrangeb<<std::endl;

        if (yrangea * yrangeb > 0.) {
            xxdbg<<"Both ends are the same sign.  Check the splits.\n";
            std::vector<double> use_splits = splits;
            if (use_splits.size() == 0) {
                xxdbg<<"No splits provided.  Use the middle instead.\n";
                use_splits.push_back( (xmin+xmax)/2. );
            }
            for (size_t k=0;k<use_splits.size();++k) {
                double xmid = use_splits[k];
                double yrangec = func(xmid);
                xxdbg<<"yrange at x = "<<xmid<<" = "<<yrangec<<std::endl;
                if (yrangea * yrangec < 0.) {
                    xxdbg<<"Found split with the opposite sign\n";
                    xxdbg<<"Find crossings on both sides:\n";
                    Solve<OverlapFinder> solver1(func,xmin,xmid);
                    solver1.setMethod(Brent);
                    double root = solver1.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    splits.push_back(root);
                    Solve<OverlapFinder> solver2(func,xmid,xmax);
                    solver2.setMethod(Brent);
                    root = solver2.root();
                    xdbg<<"Found root at "<<root<<std::endl;
                    splits.push_back(root);
                    return;
                }
            }
            xdbg<<"All split locations have the same sign, so don't add any new splits\n";
        } else {
            xxdbg<<"Ends have opposite signs.  Look for zero crossings.\n";
            Solve<OverlapFinder> solver(func,xmin,xmax);
            solver.setMethod(Brent);
            double root = solver.root();
            xdbg<<"Found root at "<<root<<std::endl;
            splits.push_back(root);
        }
    }

    double RealSpaceConvolve(
        const SBProfile& p1, const SBProfile& p2, const Position<double>& pos, double flux,
        const GSParams& gsparams)
    {
        // Coming in, if only one of them is axisymmetric, it should be p1.
        // This cuts down on some of the logic below.
        // Furthermore, the calculation of xmin, xmax isn't optimal if both are
        // axisymmetric.  But that involves a bit of geometry to get the right cuts,
        // so I didn't bother, since I don't think we'll be doing that too often.
        // So p2 is always taken to be a rectangle rather than possibly a circle.
        assert(p1.isAxisymmetric() || !p2.isAxisymmetric());

        dbg<<"Start RealSpaceConvolve for pos = "<<pos<<std::endl;
        double xmin1, xmax1, xmin2, xmax2;
        std::vector<double> xsplits1, xsplits2;
        p1.getXRange(xmin1,xmax1,xsplits1);
        p2.getXRange(xmin2,xmax2,xsplits2);
        dbg<<"p1 X range = "<<xmin1<<"  "<<xmax1<<std::endl;
        dbg<<"p2 X range = "<<xmin2<<"  "<<xmax2<<std::endl;

        // Check for early exit
        if (pos.x < xmin1 + xmin2 || pos.x > xmax1 + xmax2) {
            dbg<<"x is outside range, so trivially 0\n";
            return 0;
        }

        double ymin1, ymax1, ymin2, ymax2;
        std::vector<double> ysplits1, ysplits2;
        p1.getYRange(ymin1,ymax1,ysplits1);
        p2.getYRange(ymin2,ymax2,ysplits2);
        dbg<<"p1 Y range = "<<ymin1<<"  "<<ymax1<<std::endl;
        dbg<<"p2 Y range = "<<ymin2<<"  "<<ymax2<<std::endl;
        // Second check for early exit
        if (pos.y < ymin1 + ymin2 || pos.y > ymax1 + ymax2) {
            dbg<<"y is outside range, so trivially 0\n";
            return 0;
        }

        double xmin = std::max(xmin1, pos.x - xmax2);
        double xmax = std::min(xmax1, pos.x - xmin2);
        xdbg<<"xmin..xmax = "<<xmin<<" ... "<<xmax<<std::endl;

        // Consolidate the splits from each profile in to a single list to use.
        std::vector<double> xsplits;
        for(size_t k=0;k<xsplits1.size();++k) {
            double s = xsplits1[k];
            xdbg<<"p1 has split at "<<s<<std::endl;
            if (s > xmin && s < xmax) xsplits.push_back(s);
        }
        for(size_t k=0;k<xsplits2.size();++k) {
            double s = pos.x-xsplits2[k];
            xdbg<<"p2 has split at "<<xsplits2[k]<<", which is really (pox.x-s) "<<s<<std::endl;
            if (s > xmin && s < xmax) xsplits.push_back(s);
        }

        // If either profile is infinite, then we don't need to worry about any boundary
        // overlaps, so can skip this section.
        if ( (xmin1 == -integ::MOCK_INF || xmax2 == integ::MOCK_INF) &&
             (xmax1 == integ::MOCK_INF || xmin2 == -integ::MOCK_INF) ) {

            // Update the xmin and xmax values if the top of one profile crosses through
            // the bottom of the other.  Then part of the nominal range will in fact
            // be disjoint.  This leads to a bunch of 0's for the inner integral which
            // makes it harder for the outer integral to converge.
            OverlapFinder func1(p1,p2,pos,1);
            UpdateXRange(func1,xmin,xmax,xsplits);
            OverlapFinder func2(p1,p2,pos,2);
            UpdateXRange(func2,xmin,xmax,xsplits);

            // Third check for early exit
            if (xmin >= xmax) {
                xdbg<<"p1 and p2 are disjoint, so trivially 0\n";
                return 0.;
            }

            // Also check for where the two tops or the two bottoms might cross.
            // Then we don't have zero's, but the curve being integrated over gets a bend,
            // which also makes it hard for the outer integral to converge, so we
            // want to add split points at those bends.
            OverlapFinder func3(p1,p2,pos,3);
            AddSplitsAtBends(func3,xmin,xmax,xsplits);
            OverlapFinder func4(p1,p2,pos,4);
            AddSplitsAtBends(func4,xmin,xmax,xsplits);
        }

        ConvolveFunc conv(p1,p2,pos);

#ifdef DEBUGLOGGING
        std::ostream* integ_dbgout = Debugger::instance().do_level(3) ?
            &Debugger::instance().get_dbgout() : 0;
        integ::IntRegion<double> xreg(xmin,xmax,integ_dbgout);
        if (integ_dbgout) xreg.useFXMap();
        dbg<<"xreg = "<<xmin<<" ... "<<xmax<<std::endl;
#else
        integ::IntRegion<double> xreg(xmin,xmax);
#endif

        // Need to re-check validity of splits, since xmin,xmax may have changed.
        for(size_t k=0;k<xsplits.size();++k) {
            double s = xsplits[k];
            if (s > xmin && s < xmax) xreg.addSplit(s);
        }

        YRegion yreg(p1,p2,pos);


#ifdef TIMING
        timeval tp;
        gettimeofday(&tp,0);
        double t1 = tp.tv_sec + tp.tv_usec/1.e6;
#endif

        double result = integ::int2d(conv, xreg, yreg,
                                     gsparams.realspace_relerr,
                                     gsparams.realspace_abserr * flux);

#ifdef TIMING
        gettimeofday(&tp,0);
        double t2 = tp.tv_sec + tp.tv_usec/1.e6;
        dbg<<"Time for ("<<pos.x<<','<<pos.y<<") = "<<t2-t1<<std::endl;
#endif

        dbg<<"Found result = "<<result<<std::endl;
        return result;
    }

}
