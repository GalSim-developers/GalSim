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

#include "SBTransform.h"
#include "SBTransformImpl.h"
#include "fmath/fmath.hpp"  // Use their compiler checks for the right SSE include.

namespace galsim {

    SBTransform::SBTransform(const SBProfile& adaptee, const double* jac,
                             const Position<double>& cen, double ampScaling,
                             const GSParams& gsparams) :
        SBProfile(new SBTransformImpl(adaptee,jac,cen,ampScaling,gsparams)) {}

    SBTransform::SBTransform(const SBTransform& rhs) : SBProfile(rhs) {}

    SBTransform::~SBTransform() {}

    SBProfile SBTransform::getObj() const
    {
        assert(dynamic_cast<const SBTransformImpl*>(_pimpl.get()));
        return static_cast<const SBTransformImpl&>(*_pimpl).getObj();
    }

    void SBTransform::getJac(double& mA, double& mB, double& mC, double& mD) const
    {
        assert(dynamic_cast<const SBTransformImpl*>(_pimpl.get()));
        return static_cast<const SBTransformImpl&>(*_pimpl).getJac(mA,mB,mC,mD);
    }

    Position<double> SBTransform::getOffset() const
    {
        assert(dynamic_cast<const SBTransformImpl*>(_pimpl.get()));
        return static_cast<const SBTransformImpl&>(*_pimpl).getOffset();
    }

    double SBTransform::getFluxScaling() const
    {
        assert(dynamic_cast<const SBTransformImpl*>(_pimpl.get()));
        return static_cast<const SBTransformImpl&>(*_pimpl).getFluxScaling();
    }


    SBTransform::SBTransformImpl::SBTransformImpl(
        const SBProfile& adaptee, const double* jac,
        const Position<double>& cen, double ampScaling,
        const GSParams& gsparams) :
        SBProfileImpl(gsparams),
        _adaptee(adaptee), _cen(cen), _ampScaling(ampScaling),
        _maxk(0.), _stepk(0.), _xmin(0.), _xmax(0.), _ymin(0.), _ymax(0.),
        _kValue(0), _kValueNoPhase(0)
    {
        bool unit = !jac;
        if (jac) {
            _mA = jac[0];
            _mB = jac[1];
            _mC = jac[2];
            _mD = jac[3];
        } else {
            _mA = 1.;
            _mB = 0.;
            _mC = 0.;
            _mD = 1.;
        }
        dbg<<"Start TransformImpl\n";
        dbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", ampScaling = "<<_ampScaling<<std::endl;

        // First check if our adaptee is really another SBTransform:
        assert(GetImpl(_adaptee));
        const SBTransformImpl* sbt = dynamic_cast<const SBTransformImpl*>(GetImpl(_adaptee));
        dbg<<"sbt = "<<sbt<<std::endl;
        if (sbt) {
            dbg<<"wrapping another transformation.\n";
            // We are transforming something that's already a transformation.
            dbg<<"this transformation = "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_ampScaling<<std::endl;
            dbg<<"adaptee transformation = "<<
                sbt->_mA<<','<<sbt->_mB<<','<<sbt->_mC<<','<<sbt->_mD<<','<<
                sbt->_cen<<','<<sbt->_ampScaling<<std::endl;
            dbg<<"adaptee getFlux = "<<_adaptee.getFlux()<<std::endl;
            // We are transforming something that's already a transformation.
            // So just compound the affine transformaions
            // New matrix is product (M_this) * (M_old)
            double mA = _mA; double mB=_mB; double mC=_mC; double mD=_mD;
            _cen += Position<double>(mA*sbt->_cen.x + mB*sbt->_cen.y,
                                     mC*sbt->_cen.x + mD*sbt->_cen.y);
            _mA = mA*sbt->_mA + mB*sbt->_mC;
            _mB = mA*sbt->_mB + mB*sbt->_mD;
            _mC = mC*sbt->_mA + mD*sbt->_mC;
            _mD = mC*sbt->_mB + mD*sbt->_mD;
            unit = false;
            _ampScaling *= sbt->_ampScaling;
            dbg<<"this transformation => "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_ampScaling<<std::endl;
            _adaptee = sbt->_adaptee;
        } else {
            dbg<<"wrapping a non-transformation.\n";
            dbg<<"this transformation = "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_ampScaling<<std::endl;
        }
        _zeroCen = _cen.x == 0. && _cen.y == 0.;

        // It will be reasonably common to have an identity matrix (for just
        // a flux scaling and/or shift) for (A,B,C,D).  If so, we can use simpler
        // versions of fwd and inv:
        if (unit) {
            dbg<<"Using identity functions for fwd and inv\n";
            _fwd = &SBTransform::SBTransformImpl::_ident;
            _inv = &SBTransform::SBTransformImpl::_ident;

            _absdet = _invdet = 1.;
            _fluxScaling = _ampScaling;
        } else {
            dbg<<"Using normal fwd and inv\n";
            _fwd = &SBTransform::SBTransformImpl::_fwd_normal;
            _inv = &SBTransform::SBTransformImpl::_inv_normal;

            // Calculate some derived quantities:
            double det = _mA*_mD-_mB*_mC;
            assert(det != 0);  // Checked in python layer
            _absdet = std::abs(det);
            _invdet = 1./det;
            // The scale factor for the flux is absdet * ampScaling.
            _fluxScaling = _absdet * _ampScaling;
        }

        xdbg<<"Transformation init\n";
        xdbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        xdbg<<"_cen = "<<_cen<<std::endl;
        xdbg<<"_invdet = "<<_invdet<<std::endl;
        xdbg<<"_absdet = "<<_absdet<<std::endl;
        xdbg<<"_ampScaling = "<<_ampScaling<<std::endl;
        xdbg<<"_fluxScaling -> "<<_fluxScaling<<std::endl;
    }

    double SBTransform::SBTransformImpl::maxK() const
    {
        // The adaptee's maxk can be slow (e.g. high-n Sersic), so delay this calculation
        // until we actually need it.
        if (_maxk == 0.) {
            stepK(); // Make sure _major, _minor are set.
            _maxk = _adaptee.maxK() / _minor;
        }
        return _maxk;
    }

    double SBTransform::SBTransformImpl::stepK() const
    {
        if (_stepk == 0.) {
            double h1 = hypot( _mA+_mD, _mB-_mC);
            double h2 = hypot( _mA-_mD, _mB+_mC);
            _major = 0.5*std::abs(h1+h2);
            _minor = 0.5*std::abs(h1-h2);
            if (_major < _minor) std::swap(_major,_minor);

            _stepk = _adaptee.stepK() / _major;

            // If we have a shift, we need to further modify stepk
            //     stepk = Pi/R
            // R <- R + |shift|
            // stepk <- Pi/(Pi/stepk + |shift|)
            if (_cen.x != 0. || _cen.y != 0.) {
                double shift = sqrt( _cen.x*_cen.x + _cen.y*_cen.y );
                dbg<<"stepk from adaptee = "<<_stepk<<std::endl;
                _stepk = M_PI / (M_PI/_stepk + shift);
                dbg<<"shift = "<<shift<<", stepk -> "<<_stepk<<std::endl;
            }
        }
        return _stepk;
    }

    void SBTransform::SBTransformImpl::setupRanges() const
    {
        if (_xmin != 0. || _xmax != 0.) return;

        // Calculate the values for getXRange and getYRange:
        if (_adaptee.isAxisymmetric()) {
            // The original is a circle, so first get its radius.
            _adaptee.getXRange(_xmin,_xmax,_xsplits);
            if (_xmax == integ::MOCK_INF) {
                // Then these are correct, and use +- inf for y range too.
                _ymin = -integ::MOCK_INF;
                _ymax = integ::MOCK_INF;
            } else {
                double R = _xmax;
                // The transformation takes each point on the circle to the following new coordinates:
                // (x,y) -> (A*x + B*y + x0 , C*x + D*y + y0)
                // Using x = R cos(t) and y = R sin(t), we can find the minimum wrt t as:
                // xmax = R sqrt(A^2 + B^2) + x0
                // xmin = -R sqrt(A^2 + B^2) + x0
                // ymax = R sqrt(C^2 + D^2) + y0
                // ymin = -R sqrt(C^2 + D^2) + y0
                double AApBB = _mA*_mA + _mB*_mB;
                double sqrtAApBB = sqrt(AApBB);
                double temp = sqrtAApBB * R;
                _xmin = -temp + _cen.x;
                _xmax = temp + _cen.x;
                double CCpDD = _mC*_mC + _mD*_mD;
                double sqrtCCpDD = sqrt(CCpDD);
                temp = sqrt(CCpDD) * R;
                _ymin = -temp + _cen.y;
                _ymax = temp + _cen.y;
                _ysplits.resize(_xsplits.size());
                for (size_t k=0;k<_xsplits.size();++k) {
                    // The split points work the same way.  Scale them by the same factor we
                    // scaled the R value above, then add _cen.x or _cen.y.
                    double split = _xsplits[k];
                    xxdbg<<"Adaptee split at "<<split<<std::endl;
                    _xsplits[k] = sqrtAApBB * split + _cen.x;
                    _ysplits[k] = sqrtCCpDD * split + _cen.y;
                    xxdbg<<"-> x,y splits at "<<_xsplits[k]<<"  "<<_ysplits[k]<<std::endl;
                }
                // Now a couple of calculations that get reused in getYRangeX(x,yminymax):
                _coeff_b = (_mA*_mC + _mB*_mD) / AApBB;
                _coeff_c = CCpDD / AApBB;
                _coeff_c2 = _absdet*_absdet / AApBB;
                xxdbg<<"adaptee is axisymmetric.\n";
                xxdbg<<"adaptees maxR = "<<R<<std::endl;
                xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
                xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            }
        } else {
            // Apply the transformation to each of the four corners of the original
            // and find the minimum and maximum.
            double xmin_1, xmax_1;
            std::vector<double> xsplits0;
            _adaptee.getXRange(xmin_1,xmax_1,xsplits0);
            double ymin_1, ymax_1;
            std::vector<double> ysplits0;
            _adaptee.getYRange(ymin_1,ymax_1,ysplits0);
            // Note: This doesn't explicitly check for MOCK_INF values.
            // It shouldn't be a problem, since the integrator will still treat
            // large values near MOCK_INF as infinity, but it just means that
            // the following calculations might be wasted flops.
            Position<double> bl = fwd(Position<double>(xmin_1,ymin_1));
            Position<double> br = fwd(Position<double>(xmax_1,ymin_1));
            Position<double> tl = fwd(Position<double>(xmin_1,ymax_1));
            Position<double> tr = fwd(Position<double>(xmax_1,ymax_1));
            _xmin = std::min(std::min(std::min(bl.x,br.x),tl.x),tr.x) + _cen.x;
            _xmax = std::max(std::max(std::max(bl.x,br.x),tl.x),tr.x) + _cen.x;
            _ymin = std::min(std::min(std::min(bl.y,br.y),tl.y),tr.y) + _cen.y;
            _ymax = std::max(std::max(std::max(bl.y,br.y),tl.y),tr.y) + _cen.y;
            xxdbg<<"adaptee is not axisymmetric.\n";
            xxdbg<<"adaptees x range = "<<xmin_1<<" ... "<<xmax_1<<std::endl;
            xxdbg<<"adaptees y range = "<<ymin_1<<" ... "<<ymax_1<<std::endl;
            xxdbg<<"Corners are: bl = "<<bl<<std::endl;
            xxdbg<<"             br = "<<br<<std::endl;
            xxdbg<<"             tl = "<<tl<<std::endl;
            xxdbg<<"             tr = "<<tr<<std::endl;
            xxdbg<<"xmin..xmax = "<<_xmin<<" ... "<<_xmax<<std::endl;
            xxdbg<<"ymin..ymax = "<<_ymin<<" ... "<<_ymax<<std::endl;
            if (bl.x + _cen.x > _xmin && bl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from bl.x = "<<bl.x+_cen.x<<std::endl;
                _xsplits.push_back(bl.x+_cen.x);
            }
            if (br.x + _cen.x > _xmin && br.x + _cen.x < _xmax) {
                xxdbg<<"X Split from br.x = "<<br.x+_cen.x<<std::endl;
                _xsplits.push_back(br.x+_cen.x);
            }
            if (tl.x + _cen.x > _xmin && tl.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<std::endl;
                _xsplits.push_back(tl.x+_cen.x);
            }
            if (tr.x + _cen.x > _xmin && tr.x + _cen.x < _xmax) {
                xxdbg<<"X Split from tr.x = "<<tr.x+_cen.x<<std::endl;
                _xsplits.push_back(tr.x+_cen.x);
            }
            if (bl.y + _cen.y > _ymin && bl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from bl.y = "<<bl.y+_cen.y<<std::endl;
                _ysplits.push_back(bl.y+_cen.y);
            }
            if (br.y + _cen.y > _ymin && br.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from br.y = "<<br.y+_cen.y<<std::endl;
                _ysplits.push_back(br.y+_cen.y);
            }
            if (tl.y + _cen.y > _ymin && tl.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tl.y = "<<tl.y+_cen.y<<std::endl;
                _ysplits.push_back(tl.y+_cen.y);
            }
            if (tr.y + _cen.y > _ymin && tr.y + _cen.y < _ymax) {
                xxdbg<<"Y Split from tr.y = "<<tr.y+_cen.y<<std::endl;
                _ysplits.push_back(tr.y+_cen.y);
            }
            // If the adaptee has any splits, try to propagate those up
            for(size_t k=0;k<xsplits0.size();++k) {
                xxdbg<<"Adaptee xsplit at "<<xsplits0[k]<<std::endl;
                Position<double> bx = fwd(Position<double>(xsplits0[k],ymin_1));
                Position<double> tx = fwd(Position<double>(xsplits0[k],ymax_1));
                if (bx.x + _cen.x > _xmin && bx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from bx.x = "<<bx.x+_cen.x<<std::endl;
                    _xsplits.push_back(bx.x+_cen.x);
                }
                if (tx.x + _cen.x > _xmin && tx.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tx.x = "<<tx.x+_cen.x<<std::endl;
                    _xsplits.push_back(tx.x+_cen.x);
                }
                if (bx.y + _cen.y > _ymin && bx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from bx.y = "<<bx.y+_cen.y<<std::endl;
                    _ysplits.push_back(bx.y+_cen.y);
                }
                if (tx.y + _cen.y > _ymin && tx.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from tx.y = "<<tx.y+_cen.y<<std::endl;
                    _ysplits.push_back(tx.y+_cen.y);
                }
            }
            for(size_t k=0;k<ysplits0.size();++k) {
                xxdbg<<"Adaptee ysplit at "<<ysplits0[k]<<std::endl;
                Position<double> yl = fwd(Position<double>(xmin_1,ysplits0[k]));
                Position<double> yr = fwd(Position<double>(xmax_1,ysplits0[k]));
                if (yl.x + _cen.x > _xmin && yl.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from tl.x = "<<tl.x+_cen.x<<std::endl;
                    _xsplits.push_back(yl.x+_cen.x);
                }
                if (yr.x + _cen.x > _xmin && yr.x + _cen.x < _xmax) {
                    xxdbg<<"X Split from yr.x = "<<yr.x+_cen.x<<std::endl;
                    _xsplits.push_back(yr.x+_cen.x);
                }
                if (yl.y + _cen.y > _ymin && yl.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yl.y = "<<yl.y+_cen.y<<std::endl;
                    _ysplits.push_back(yl.y+_cen.y);
                }
                if (yr.y + _cen.y > _ymin && yr.y + _cen.y < _ymax) {
                    xxdbg<<"Y Split from yr.y = "<<yr.y+_cen.y<<std::endl;
                    _ysplits.push_back(yr.y+_cen.y);
                }
            }
        }
    }

    void SBTransform::SBTransformImpl::getXRange(
        double& xmin, double& xmax, std::vector<double>& splits) const
    {
        setupRanges();
        xmin = _xmin; xmax = _xmax;
        splits.insert(splits.end(),_xsplits.begin(),_xsplits.end());
    }

    void SBTransform::SBTransformImpl::getYRange(
        double& ymin, double& ymax, std::vector<double>& splits) const
    {
        setupRanges();
        ymin = _ymin; ymax = _ymax;
        splits.insert(splits.end(),_ysplits.begin(),_ysplits.end());
    }

    void SBTransform::SBTransformImpl::getYRangeX(
        double x, double& ymin, double& ymax, std::vector<double>& splits) const
    {
        setupRanges();
        xxdbg<<"Transformation getYRangeX for x = "<<x<<std::endl;
        if (_adaptee.isAxisymmetric()) {
            std::vector<double> splits0;
            _adaptee.getYRange(ymin,ymax,splits0);
            if (ymax == integ::MOCK_INF) return;
            double R = ymax;
            // The circlue with radius R is mapped onto an ellipse with (x,y) given by:
            // x = A R cos(t) + B R sin(t) + x0
            // y = C R cos(t) + D R sin(t) + y0
            //
            // Or equivalently:
            // (A^2+B^2) (y-y0)^2 - 2(AC+BD) (x-x0)(y-y0) + (C^2+D^2) (x-x0)^2 = R^2 (AD-BC)^2
            //
            // Given a particular value for x, we solve the latter equation for the
            // corresponding range for y.
            // y^2 - 2 b y = c
            // -> y^2 - 2b y = c
            //    (y - b)^2 = c + b^2
            //    y = b +- sqrt(c + b^2)
            double b = _coeff_b * (x-_cen.x);
            double c = _coeff_c2 * R*R - _coeff_c * (x-_cen.x) * (x-_cen.x);
            double d = sqrt(c + b*b);
            ymax = b + d + _cen.y;
            ymin = b - d + _cen.y;
            for (size_t k=0;k<splits0.size();++k) if (splits0[k] >= 0.) {
                double r = splits0[k];
                double c = _coeff_c2 * r*r - _coeff_c * (x-_cen.x) * (x-_cen.x);
                double d = sqrt(c+b*b);
                splits.push_back(b + d + _cen.y);
                splits.push_back(b - d + _cen.y);
            }
            xxdbg<<"Axisymmetric adaptee with R = "<<R<<std::endl;
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
        } else {
            // There are 4 lines to check for where they intersect the given x.
            // Start with the adaptee's given ymin.
            // This line is transformed onto the line:
            // (x',ymin) -> ( A x' + B ymin + x0 , C x' + D ymin + y0 )
            // x' = (x - x0 - B ymin) / A
            // y = C x' + D ymin + y0
            //   = C (x - x0 - B ymin) / A + D ymin + y0
            // The top line is analagous for ymax instead of ymin.
            //
            // The left line is transformed as:
            // (xmin,y) -> ( A xmin + B y' + x0 , C xmin + D y' + y0 )
            // y' = (x - x0 - A xmin) / B
            // y = C xmin + D (x - x0 - A xmin) / B + y0
            // And again, the right line is analgous.
            //
            // We also need to check for A or B = 0, since then only one pair of lines is
            // relevant.
            xxdbg<<"Non-axisymmetric adaptee\n";
            if (_mA == 0.) {
                xxdbg<<"_mA == 0:\n";
                double xmin_1, xmax_1;
                std::vector<double> xsplits0;
                _adaptee.getXRange(xmin_1,xmax_1,xsplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin = _mC * xmin_1 + _mD * (x - _cen.x - _mA*xmin_1) / _mB + _cen.y;
                ymax = _mC * xmax_1 + _mD * (x - _cen.x - _mA*xmax_1) / _mB + _cen.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(_mC * xx + _mD * (x - _cen.x - _mA*xx) / _mB + _cen.y);
                }
            } else if (_mB == 0.) {
                xxdbg<<"_mB == 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> ysplits0;
                _adaptee.getYRange(ymin_1,ymax_1,ysplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = _mC * (x - _cen.x - _mB*ymin_1) / _mA + _mD*ymin_1 + _cen.y;
                ymax = _mC * (x - _cen.x - _mB*ymax_1) / _mA + _mD*ymax_1 + _cen.y;
                if (ymax < ymin) std::swap(ymin,ymax);
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(_mC * (x - _cen.x - _mB*yy) / _mA + _mD*yy + _cen.y);
                }
            } else {
                xxdbg<<"_mA,B != 0:\n";
                double ymin_1, ymax_1;
                std::vector<double> xsplits0;
                _adaptee.getYRange(ymin_1,ymax_1,xsplits0);
                xxdbg<<"ymin_1, ymax_1 = "<<ymin_1<<','<<ymax_1<<std::endl;
                ymin = _mC * (x - _cen.x - _mB*ymin_1) / _mA + _mD*ymin_1 + _cen.y;
                ymax = _mC * (x - _cen.x - _mB*ymax_1) / _mA + _mD*ymax_1 + _cen.y;
                xxdbg<<"From top and bottom: ymin,ymax = "<<ymin<<','<<ymax<<std::endl;
                if (ymax < ymin) std::swap(ymin,ymax);
                double xmin_1, xmax_1;
                std::vector<double> ysplits0;
                _adaptee.getXRange(xmin_1,xmax_1,ysplits0);
                xxdbg<<"xmin_1, xmax_1 = "<<xmin_1<<','<<xmax_1<<std::endl;
                ymin_1 = _mC * xmin_1 + _mD * (x - _cen.x - _mA*xmin_1) / _mB + _cen.y;
                ymax_1 = _mC * xmax_1 + _mD * (x - _cen.x - _mA*xmax_1) / _mB + _cen.y;
                xxdbg<<"From left and right: ymin,ymax = "<<ymin_1<<','<<ymax_1<<std::endl;
                if (ymax_1 < ymin_1) std::swap(ymin_1,ymax_1);
                if (ymin_1 > ymin) ymin = ymin_1;
                if (ymax_1 < ymax) ymax = ymax_1;
                for(size_t k=0;k<ysplits0.size();++k) {
                    double yy = ysplits0[k];
                    splits.push_back(_mC * (x - _cen.x - _mB*yy) / _mA + _mD*yy + _cen.y);
                }
                for(size_t k=0;k<xsplits0.size();++k) {
                    double xx = xsplits0[k];
                    splits.push_back(_mC * xx + _mD * (x - _cen.x - _mA*xx) / _mB + _cen.y);
                }
            }
            xxdbg<<"ymin .. ymax = "<<ymin<<" ... "<<ymax<<std::endl;
        }
    }

    double SBTransform::SBTransformImpl::xValue(const Position<double>& p) const
    { return _adaptee.xValue(inv(p-_cen)) * _ampScaling; }

    std::complex<double> SBTransform::SBTransformImpl::kValue(const Position<double>& k) const
    {
        if (!_kValue) {
            // Figure out which function we need for kValue and kValueNoPhase
            if (std::abs(_fluxScaling-1.) < this->gsparams.kvalue_accuracy) {
                xdbg<<"fluxScaling = "<<_fluxScaling<<" = 1, so use NoDet version.\n";
                _kValueNoPhase = &SBTransform::SBTransformImpl::_kValueNoPhaseNoDet;
            } else {
                xdbg<<"fluxScaling = "<<_fluxScaling<<" != 1, so use WithDet version.\n";
                _kValueNoPhase = &SBTransform::SBTransformImpl::_kValueNoPhaseWithDet;
            }
            if (_zeroCen) {
                _kValue = _kValueNoPhase;
            } else {
                _kValue = &SBTransform::SBTransformImpl::_kValueWithPhase;
            }
        }
        return _kValue(_adaptee,fwdT(k),_fluxScaling,k,_cen);
    }

    std::complex<double> SBTransform::SBTransformImpl::kValueNoPhase(
        const Position<double>& k) const
    { return _kValueNoPhase(_adaptee,fwdT(k),_fluxScaling,k,_cen); }

    std::complex<double> SBTransform::SBTransformImpl::_kValueNoPhaseNoDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
        const Position<double>& , const Position<double>& )
    { return adaptee.kValue(fwdTk); }

    std::complex<double> SBTransform::SBTransformImpl::_kValueNoPhaseWithDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
        const Position<double>& , const Position<double>& )
    { return fluxScaling * adaptee.kValue(fwdTk); }

    std::complex<double> SBTransform::SBTransformImpl::_kValueWithPhase(
        const SBProfile& adaptee, const Position<double>& fwdTk, double fluxScaling,
        const Position<double>& k, const Position<double>& cen)
    { return adaptee.kValue(fwdTk) * std::polar(fluxScaling , -k.x*cen.x-k.y*cen.y); }

    // A helper class for doing the inner loops in the below fill*Image functions.
    // This lets us do type-specific optimizations on just this portion.
    // First the normal (legible) version that we use if there is no SSE support.
    template <typename T>
    struct InnerLoopHelper
    {
        static inline void phaseloop_1d(std::complex<T>*& ptr, const std::complex<T>* kxit,
                                        int m, const std::complex<T>& kyflux)
        {
            for (; m; --m)
                *ptr++ *= *kxit++ * kyflux;
        }
    };

#ifdef __SSE__
    template <>
    struct InnerLoopHelper<float>
    {
        static inline void phaseloop_1d(std::complex<float>*& ptr, const std::complex<float>* kxit,
                                        int m, const std::complex<float>& kyflux)
        {
            // First get to an aligned value
            for (; m && !IsAligned(ptr); --m)
                *ptr++ *= *kxit++ * kyflux;

            int m2 = m>>1;
            int ma = m2<<1;
            m -= ma;

            // Do 2 at a time as far as possible
            if (m2) {
                const float kyfr = kyflux.real();
                const float kyfi = kyflux.imag();
                const __m128 mkyfr = _mm_set1_ps(kyfr);
                const __m128 mkyfi = _mm_set_ps(kyfi, -kyfi, kyfi, -kyfi);
                const __m128 mneg = _mm_set_ps(1, -1, 1, -1);
                do {
                    // Separate out calculation into components
                    // z = u * v
                    // zr = ur * vr - ui * vi
                    // zi = ur * vi + ui * vr
                    // Do this twice, since we have two complex products
                    __m128 mkx = _mm_loadu_ps(reinterpret_cast<const float*>(kxit));
                    kxit += 2;
                    __m128 mp = _mm_load_ps(reinterpret_cast<float*>(ptr));
                    // For now, u is kyf, v is kx, z* are temporaries
                    __m128 mvir = _mm_shuffle_ps(mkx, mkx, _MM_SHUFFLE(2,3,0,1));  // (vi, vr)
                    __m128 mz1 = _mm_mul_ps(mkyfr, mkx);  // (ur * vr, ur * vi)
                    __m128 mz2 = _mm_mul_ps(mkyfi, mvir);  // (-ui * vi, ui * vr)
                    __m128 mz = _mm_add_ps(mz1, mz2);    // (ur vr - ui vi, ur vi + ui vr)
                    // Repeat taking z as u and p as v
                    mvir = _mm_shuffle_ps(mp, mp, _MM_SHUFFLE(2,3,0,1));  // (vi, vr)
                    __m128 mur = _mm_shuffle_ps(mz, mz, _MM_SHUFFLE(2,2,0,0));  // (ur, ur)
                    __m128 mui = _mm_shuffle_ps(mz, mz, _MM_SHUFFLE(3,3,1,1));  // (ui, ui)
                    mui = _mm_mul_ps(mneg, mui); // (-ui, ui)
                    mz1 = _mm_mul_ps(mur, mp);  // (ur * vr, ur * vi)
                    mz2 = _mm_mul_ps(mui, mvir);  // (-ui * vi, ui * vr)
                    mz = _mm_add_ps(mz1, mz2);    // (ur vr - ui vi, ur vi + ui vr)
                    _mm_store_ps(reinterpret_cast<float*>(ptr), mz);
                    ptr += 2;
                } while (--m2);
            }

            // Finally finish up the last one, if any
            if (m) {
                *ptr++ *= *kxit++ * kyflux;
            }
        }
    };
#endif
#ifdef __SSE2__
    template <>
    struct InnerLoopHelper<double>
    {
        static inline void phaseloop_1d(std::complex<double>*& ptr,
                                        const std::complex<double>* kxit,
                                        int m, const std::complex<double>& kyflux)
        {
            // If not aligned, do the normal loop.  (Should be rare.)
            if (!IsAligned(ptr)) {
                for (; m; --m)
                    *ptr++ *= *kxit++ * kyflux;
                return;
            }

            const double kyfr = kyflux.real();
            const double kyfi = kyflux.imag();
            const __m128d mkyfr = _mm_set1_pd(kyfr);
            const __m128d mkyfi = _mm_set_pd(kyfi, -kyfi);
            const __m128d mneg = _mm_set_pd(1, -1);
            for (; m; --m) {
                // Separate out calculation into components
                // z = u * v
                // zr = ur * vr - ui * vi
                // zi = ur * vi + ui * vr
                // Do this twice, since we have two complex products
                __m128d mkx = _mm_loadu_pd(reinterpret_cast<const double*>(kxit++));
                __m128d mp = _mm_load_pd(reinterpret_cast<double*>(ptr));
                // For now, u is kyf, v is kx, z* are temporaries
                __m128d mvir = _mm_shuffle_pd(mkx, mkx, _MM_SHUFFLE2(0,1));  // (vi, vr)
                __m128d mz1 = _mm_mul_pd(mkyfr, mkx);  // (ur * vr, ur * vi)
                __m128d mz2 = _mm_mul_pd(mkyfi, mvir);  // (-ui * vi, ui * vr)
                __m128d mz = _mm_add_pd(mz1, mz2);    // (ur vr - ui vi, ur vi + ui vr)
                // Repeat taking z as u and p as v
                mvir = _mm_shuffle_pd(mp, mp, _MM_SHUFFLE2(0,1));  // (vi, vr)
                __m128d mur = _mm_shuffle_pd(mz, mz, _MM_SHUFFLE2(0,0));  // (ur, ur)
                __m128d mui = _mm_shuffle_pd(mz, mz, _MM_SHUFFLE2(1,1));  // (ui, ui)
                mui = _mm_mul_pd(mneg, mui); // (-ui, ui)
                mz1 = _mm_mul_pd(mur, mp);  // (ur * vr, ur * vi)
                mz2 = _mm_mul_pd(mui, mvir);  // (-ui * vi, ui * vr)
                mz = _mm_add_pd(mz1, mz2);    // (ur vr - ui vi, ur vi + ui vr)
                _mm_store_pd(reinterpret_cast<double*>(ptr++), mz);
            }
        }
    };
#endif

    template <typename T>
    void SBTransform::SBTransformImpl::fillXImage(ImageView<T> im,
                                                  double x0, double dx, int izero,
                                                  double y0, double dy, int jzero) const
    {
        dbg<<"SBTransform fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<", izero = "<<izero<<std::endl;
        dbg<<"y = "<<y0<<" + j * "<<dy<<", jzero = "<<jzero<<std::endl;
        dbg<<"A,B,C,D = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", zerocen = "<<_zeroCen<<std::endl;
        dbg<<"fluxScaling = "<<_fluxScaling<<", invdet = "<<_invdet<<std::endl;
        dbg<<"ampScaling = "<<_ampScaling<<std::endl;
        const int m = im.getNCol();
        const int n = im.getNRow();

        // Subtract cen
        if (!_zeroCen) {
            x0 -= _cen.x;
            y0 -= _cen.y;
            // Check if the new center falls on an integer index.
            // 0 = x0 + iz * dx
            // 0 = y0 + jz * dy
            xdbg<<"x0,y0 = "<<x0<<','<<y0<<std::endl;
            int iz = int(-x0/dx+0.5);
            int jz = int(-y0/dy+0.5);
            xdbg<<"iz,jz = "<<iz<<','<<jz<<std::endl;
            xdbg<<"near zero at "<<(x0+iz*dx)<<"  "<<(y0+jz*dy)<<std::endl;

            if (std::abs(x0 + iz*dx) < 1.e-10 && iz > 0 && iz < m) izero = iz;
            else izero = 0;
            if (std::abs(y0 + jz*dy) < 1.e-10 && jz > 0 && jz < n) jzero = jz;
            else jzero = 0;
        }

        // Apply inv to x,y
        if (_mB == 0. && _mC == 0.) {
            double xscal = _invdet * _mD;
            double yscal = _invdet * _mA;
            x0 *= xscal;
            dx *= xscal;
            y0 *= yscal;
            dy *= yscal;

            GetImpl(_adaptee)->fillXImage(im,x0,dx,izero,y0,dy,jzero);
        } else {
            Position<double> inv0 = inv(Position<double>(x0,y0));
            Position<double> inv1 = inv(Position<double>(dx,0.));
            Position<double> inv2 = inv(Position<double>(0.,dy));
            xdbg<<"inv0 = "<<inv0<<std::endl;
            xdbg<<"inv1 = "<<inv1<<std::endl;
            xdbg<<"inv2 = "<<inv2<<std::endl;

            GetImpl(_adaptee)->fillXImage(im,inv0.x,inv1.x,inv2.x,inv0.y,inv2.y,inv1.y);
        }

        // Apply flux scaling
        if (std::abs(_ampScaling - 1.) > this->gsparams.xvalue_accuracy)
            im *= T(_ampScaling);
    }

    template <typename T>
    void SBTransform::SBTransformImpl::fillXImage(ImageView<T> im,
                                                  double x0, double dx, double dxy,
                                                  double y0, double dy, double dyx) const
    {
        dbg<<"SBTransform fillXImage\n";
        dbg<<"x = "<<x0<<" + i * "<<dx<<" + j * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + i * "<<dyx<<" + j * "<<dy<<std::endl;
        dbg<<"A,B,C,D = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", zerocen = "<<_zeroCen<<std::endl;
        dbg<<"fluxScaling = "<<_fluxScaling<<", invdet = "<<_invdet<<std::endl;
        dbg<<"ampScaling = "<<_ampScaling<<std::endl;

        // Subtract cen
        if (!_zeroCen) {
            x0 -= _cen.x;
            y0 -= _cen.y;
        }

        // Apply inv to x,y
        Position<double> inv0 = inv(Position<double>(x0,y0));
        Position<double> inv1 = inv(Position<double>(dx,dyx));
        Position<double> inv2 = inv(Position<double>(dxy,dy));
        xdbg<<"inv0 = "<<inv0<<std::endl;
        xdbg<<"inv1 = "<<inv1<<std::endl;
        xdbg<<"inv2 = "<<inv2<<std::endl;

        GetImpl(_adaptee)->fillXImage(im,inv0.x,inv1.x,inv2.x,inv0.y,inv2.y,inv1.y);

        // Apply flux scaling
        if (std::abs(_ampScaling - 1.) > this->gsparams.xvalue_accuracy)
            im *= T(_ampScaling);
    }

    // A helper function for filKImage below.
    // Probably not worth specializing using SSE, since not much time spent in this.
    template <typename T>
    inline void fillphase_1d(std::complex<T>* kit, int m, T k, T dk)
    {
#if 0
        // Original, more legible code
        for (; m; --m, k+=dk)
            *kit++ = std::polar(T(1), -k);
#else
        // Implement by repeated multiplications by polar(1, -dk), rather than computing
        // the polar form each time. (slow trig!)
        // This is mildly unstable, so guard the magnitude by multiplying by
        // 1/|z|.  Since z ~= 1, 1/|z| is very nearly = |z|^2^-1/2 ~= 1.5 - 0.5|z|^2.
        std::complex<T> kpol = std::polar(T(1), -k);
        std::complex<T> dkpol = std::polar(T(1), -dk);
        *kit++ = kpol;
        for (--m; m; --m) {
            kpol = kpol * dkpol;
            kpol = kpol * T(1.5 - 0.5 * std::norm(kpol));
            *kit++ = kpol;
        }
#endif
    }

    template <typename T>
    void ApplyKImagePhases(ImageView<std::complex<T> > im,
                           double kx0, double dkx, double ky0, double dky,
                           double cenx, double ceny, double fluxScaling)
    {
        // Make phase terms = |det| exp(-i(kx*cenx + ky*ceny))
        // In this case, the terms are separable, so only need to make kx and ky phases
        // separately.
        const int m = im.getNCol();
        int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= cenx;
        dkx *= cenx;
        ky0 *= ceny;
        dky *= ceny;

        // Use the stack rather than the heap for these, since a bit faster and small
        // enough that they should fit without any problem.
        T xphase_kx[2*m];
        T xphase_ky[2*n];
        std::complex<T>* phase_kx = reinterpret_cast<std::complex<T>*>(xphase_kx);
        std::complex<T>* phase_ky = reinterpret_cast<std::complex<T>*>(xphase_ky);

        fillphase_1d<T>(phase_kx, m, kx0, dkx);
        fillphase_1d<T>(phase_ky, n, ky0, dky);

        for (; n; --n, ptr+=skip, ++phase_ky) {
            InnerLoopHelper<T>::phaseloop_1d(ptr, phase_kx, m, T(fluxScaling) * *phase_ky);
        }
    }

    template <typename T>
    void ApplyKImagePhases(ImageView<std::complex<T> > im,
                           double kx0, double dkx, double dkxy,
                           double ky0, double dky, double dkyx,
                           double cenx, double ceny, double fluxScaling)
    {
        const int m = im.getNCol();
        const int n = im.getNRow();
        std::complex<T>* ptr = im.getData();
        int skip = im.getNSkip();
        assert(im.getStep() == 1);

        kx0 *= cenx;
        dkx *= cenx;
        dkxy *= cenx;
        ky0 *= ceny;
        dky *= ceny;
        dkyx *= ceny;

        // Only ever use these as sum of kx + ky, so add them together now.
        T k0 = kx0 + ky0;
        T dk0 = dkxy + dky;
        T dk1 = dkx + dkyx;

        for (int j=n; j; --j, k0+=dk0, ptr+=skip) {
            T k = k0;
#if 0
            // Original, more legible code
            for (int i=m; i; --i, k+=dk1) {
                *ptr++ *= std::polar(T(fluxScaling), -k);
            }
#else
            // See comments above in fillphase_1d for what's going on here.
            // MJ: Could consider putting this in the InnerLoop struct above and write
            // specialized SSE versions, since native complex multiplication is terribly slow.
            // But this use case is very rare, so probably not worth it.
            std::complex<T> kpol = std::polar(T(1), -k);
            std::complex<T> dkpol = std::polar(T(1), -dk1);
            *ptr++ *= fluxScaling * kpol;
            for (int i=m-1; i; --i) {
                kpol = kpol * dkpol;
                kpol = kpol * T(1.5 - 0.5 * std::norm(kpol));
                *ptr++ *= fluxScaling * kpol;
            }
#endif
        }
    }

    // This one is exposed to Python
    template <typename T>
    void ApplyKImagePhases(ImageView<std::complex<T> > image, double imscale, const double* jac,
                           double cenx, double ceny, double fluxScaling)
    {
        dbg<<"Start ApplyKImagePhases: \n";
        dbg<<"bounds = "<<image.getBounds()<<std::endl;
        dbg<<"imscale = "<<imscale<<std::endl;
        assert(image.getStep() == 1);

        int xmin = image.getXMin();
        int ymin = image.getYMin();
        double x0 = xmin*imscale;
        double y0 = ymin*imscale;

        if (!jac) {
            dbg<<"no jac\n";
            ApplyKImagePhases(image, x0, imscale, y0, imscale, cenx, ceny, fluxScaling);
        } else if (jac[1] == 0. && jac[2] == 0.) {
            double mA = jac[0];
            double mD = jac[3];
            dbg<<"diag jac: "<<mA<<','<<mD<<std::endl;
            double new_x0 = x0 * mA;
            double new_y0 = y0 * mD;
            double dx = imscale * mA;
            double dy = imscale * mD;
            ApplyKImagePhases(image, new_x0, dx, new_y0, dy, cenx, ceny, fluxScaling);
        } else {
            double mA = jac[0];
            double mB = jac[1];
            double mC = jac[2];
            double mD = jac[3];
            dbg<<"jac = "<<mA<<','<<mB<<','<<mC<<','<<mD<<std::endl;
            double new_x0 = mA*x0 + mC*y0;
            double new_y0 = mB*x0 + mD*y0;
            double dx = mA * imscale;
            double dxy = mC * imscale;
            double dy = mD * imscale;
            double dyx = mB * imscale;
            ApplyKImagePhases(image, new_x0, dx, dxy, new_y0, dy, dyx, cenx, ceny, fluxScaling);
        }
    }

    template <typename T>
    void SBTransform::SBTransformImpl::fillKImage(ImageView<std::complex<T> > im,
                                                  double kx0, double dkx, int izero,
                                                  double ky0, double dky, int jzero) const
    {
        dbg<<"SBTransform fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<", izero = "<<izero<<std::endl;
        dbg<<"ky = "<<ky0<<" + j * "<<dky<<", jzero = "<<jzero<<std::endl;
        dbg<<"A,B,C,D = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", zerocen = "<<_zeroCen<<std::endl;
        dbg<<"fluxScaling = "<<_fluxScaling<<", invdet = "<<_invdet<<std::endl;
        dbg<<"ampScaling = "<<_ampScaling<<std::endl;

        // Apply fwdT to kx,ky
        if (_mB == 0. && _mC == 0.) {
            double fwdT_kx0 = _mA * kx0;
            double fwdT_dkx = _mA * dkx;
            double fwdT_ky0 = _mD * ky0;
            double fwdT_dky = _mD * dky;

            GetImpl(_adaptee)->fillKImage(im,fwdT_kx0,fwdT_dkx,izero,fwdT_ky0,fwdT_dky,jzero);
        } else {
            Position<double> fwdT0 = fwdT(Position<double>(kx0,ky0));
            Position<double> fwdT1 = fwdT(Position<double>(dkx,0.));
            Position<double> fwdT2 = fwdT(Position<double>(0.,dky));
            xdbg<<"fwdT0 = "<<fwdT0<<std::endl;
            xdbg<<"fwdT1 = "<<fwdT1<<std::endl;
            xdbg<<"fwdT2 = "<<fwdT2<<std::endl;

            GetImpl(_adaptee)->fillKImage(im,fwdT0.x,fwdT1.x,fwdT2.x,fwdT0.y,fwdT2.y,fwdT1.y);
        }

        // Apply phases
        if (_zeroCen) {
            xdbg<<"zeroCen\n";
            if (std::abs(_fluxScaling - 1.) > this->gsparams.kvalue_accuracy)
                im *= T(_fluxScaling);
        } else {
            xdbg<<"!zeroCen\n";
            ApplyKImagePhases(im, kx0, dkx, ky0, dky, _cen.x, _cen.y, _fluxScaling);
        }
    }

    template <typename T>
    void SBTransform::SBTransformImpl::fillKImage(ImageView<std::complex<T> > im,
                                                  double kx0, double dkx, double dkxy,
                                                  double ky0, double dky, double dkyx) const
    {
        dbg<<"SBTransform fillKImage\n";
        dbg<<"kx = "<<kx0<<" + i * "<<dkx<<" + j * "<<dkxy<<std::endl;
        dbg<<"ky = "<<ky0<<" + i * "<<dkyx<<" + j * "<<dky<<std::endl;
        dbg<<"A,B,C,D = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"cen = "<<_cen<<", zerocen = "<<_zeroCen<<std::endl;
        dbg<<"fluxScaling = "<<_fluxScaling<<", invdet = "<<_invdet<<std::endl;
        dbg<<"ampScaling = "<<_ampScaling<<std::endl;

        // Apply fwdT to kx,ky
        // Original (x,y):
        //     kx = kx0 + i dkx + j dkxy
        //     ky = ky0 + i dkyx + j dky
        // (kx',ky') = fwdT(kx,ky)
        //     kx' = A kx + C ky
        //         = (A kx0 + C ky0) + i (A dkx + C dkyx) + j (A dkxy + C dky)
        //     ky' = B kx + D ky
        //         = (B kx0 + D ky0) + i (B dkx + D dkyx) + j (B dkxy + D dky)
        //
        Position<double> fwdT0 = fwdT(Position<double>(kx0,ky0));
        Position<double> fwdT1 = fwdT(Position<double>(dkx,dkyx));
        Position<double> fwdT2 = fwdT(Position<double>(dkxy,dky));
        xdbg<<"fwdT0 = "<<fwdT0<<std::endl;
        xdbg<<"fwdT1 = "<<fwdT1<<std::endl;
        xdbg<<"fwdT2 = "<<fwdT2<<std::endl;

        GetImpl(_adaptee)->fillKImage(im,fwdT0.x,fwdT1.x,fwdT2.x,fwdT0.y,fwdT2.y,fwdT1.y);

        // Apply phase terms = |det| exp(-i(kx*cenx + ky*ceny))
        if (_zeroCen) {
            xdbg<<"zeroCen\n";
            if (std::abs(_fluxScaling - 1.) > this->gsparams.kvalue_accuracy)
                im *= T(_fluxScaling);
        } else {
            xdbg<<"!zeroCen\n";
            ApplyKImagePhases(im, kx0, dkx, dkxy, ky0, dky, dkyx, _cen.x, _cen.y, _fluxScaling);
        }
    }

    void SBTransform::SBTransformImpl::shoot(PhotonArray& photons, UniformDeviate ud) const
    {
        const int N = photons.size();
        dbg<<"Distort shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Simple job here: just remap coords of each photon, then change flux
        // If there is overall magnification in the transform
        _adaptee.shoot(photons,ud);
        for (int i=0; i<N; i++) {
            Position<double> xy = fwd(Position<double>(photons.getX(i), photons.getY(i)))+_cen;
            photons.setPhoton(i, xy.x, xy.y, photons.getFlux(i)*_fluxScaling);
        }
        dbg<<"Distort Realized flux = "<<photons.getTotalFlux()<<std::endl;
    }

    template void ApplyKImagePhases(ImageView<std::complex<double> > image,
                                    double imscale, const double* jac,
                                    double cenx, double ceny, double fluxScaling);
    template void ApplyKImagePhases(ImageView<std::complex<float> > image,
                                    double imscale, const double* jac,
                                    double cenx, double ceny, double fluxScaling);
}
