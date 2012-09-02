
//#define DEBUGLOGGING

#include "TMV.h"
#include "SBTransform.h"
#include "SBTransformImpl.h"
#include <iomanip>

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {
   
    SBTransform::SBTransform(const SBProfile& sbin,
                double mA, double mB, double mC, double mD,
                const Position<double>& cen, double fluxScaling) :
        SBProfile(new SBTransformImpl(sbin,mA,mB,mC,mD,cen,fluxScaling)) {}

    SBTransform::SBTransform(const SBProfile& sbin,
                const CppEllipse& e, double fluxScaling) :
        SBProfile(new SBTransformImpl(sbin,e,fluxScaling)) {}

    SBTransform::SBTransform(const SBTransform& rhs) : SBProfile(rhs) {}

    SBTransform::~SBTransform() {}

    SBTransform::SBTransformImpl::SBTransformImpl(
        const SBProfile& sbin, double mA, double mB, double mC, double mD,
        const Position<double>& cen, double fluxScaling) :
        _adaptee(sbin), _mA(mA), _mB(mB), _mC(mC), _mD(mD), _cen(cen), _fluxScaling(fluxScaling)
    {
        dbg<<"Start TransformImpl (1)\n";
        dbg<<"matrix = "<<std::setprecision(15)<<_mA<<','<<std::setprecision(15)<<_mB<<','<<std::setprecision(15)<<_mC<<','<<std::setprecision(15)<<_mD<<std::endl;
        dbg<<"cen = "<<std::setprecision(15)<<_cen<<", fluxScaling = "<<std::setprecision(15)<<_fluxScaling<<std::endl;

        // All the actual initialization is in a separate function so we can share code
        // with the other constructor.
        initialize();
    }

    SBTransform::SBTransformImpl::SBTransformImpl(
        const SBProfile& sbin, const CppEllipse& e, double fluxScaling) :
        _adaptee(sbin), _cen(e.getX0()), _fluxScaling(fluxScaling)
    {
        dbg<<"Start TransformImpl (2)\n";
        dbg<<"e = "<<std::setprecision(15)<<e<<", fluxScaling = "<<std::setprecision(15)<<_fluxScaling<<std::endl;
        // First get what we need from the CppEllipse:
        tmv::Matrix<double> m = e.getMatrix();
        _mA = m(0,0);
        _mB = m(0,1);
        _mC = m(1,0);
        _mD = m(1,1);

        // Then move on to the rest of the initialization process.
        initialize();
    }

    void SBTransform::SBTransformImpl::initialize()
    {
        dbg<<"Start SBTransformImpl initialize\n";
        // First check if our adaptee is really another SBTransform:
        assert(SBProfile::GetImpl(_adaptee));
        const SBTransformImpl* sbd = dynamic_cast<const SBTransformImpl*>(
            SBProfile::GetImpl(_adaptee));
        dbg<<"sbd = "<<sbd<<std::endl;
        if (sbd) {
            dbg<<"wrapping another transformation.\n";
            // We are transforming something that's already a transformation.
            dbg<<"this transformation = "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_fluxScaling<<std::endl;
            dbg<<"adaptee transformation = "<<
                sbd->_mA<<','<<sbd->_mB<<','<<sbd->_mC<<','<<sbd->_mD<<','<<
                sbd->_cen<<','<<sbd->_fluxScaling<<std::endl;
            dbg<<"adaptee getFlux = "<<_adaptee.getFlux()<<std::endl;
            // We are transforming something that's already a transformation.
            // So just compound the affine transformaions
            // New matrix is product (M_this) * (M_old)
            double mA = _mA; double mB=_mB; double mC=_mC; double mD=_mD;
            _cen += Position<double>(mA*sbd->_cen.x + mB*sbd->_cen.y,
                                     mC*sbd->_cen.x + mD*sbd->_cen.y);
            _mA = mA*sbd->_mA + mB*sbd->_mC;
            _mB = mA*sbd->_mB + mB*sbd->_mD;
            _mC = mC*sbd->_mA + mD*sbd->_mC;
            _mD = mC*sbd->_mB + mD*sbd->_mD;
            _fluxScaling *= sbd->_fluxScaling;
            dbg<<"this transformation => "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_fluxScaling<<std::endl;
            _adaptee = sbd->_adaptee;
        } else {
            dbg<<"wrapping a non-transformation.\n";
            dbg<<"this transformation = "<<
                _mA<<','<<_mB<<','<<_mC<<','<<_mD<<','<<
                _cen<<','<<_fluxScaling<<std::endl;
        }

        // It will be reasonably common to have an identity matrix (for just
        // a flux scaling and/or shift) for (A,B,C,D).  If so, we can use simpler
        // versions of fwd and inv:
        if (_mA == 1. && _mB == 0. && _mC == 0. && _mD == 1.) {
            dbg<<"Using identity functions for fwd and inv\n";
            _fwd = &SBTransform::_ident;
            _inv = &SBTransform::_ident;
        } else {
            dbg<<"Using normal fwd and inv\n";
            _fwd = &SBTransform::_fwd_normal;
            _inv = &SBTransform::_inv_normal;
        }

        // Calculate some derived quantities:
        double det = _mA*_mD-_mB*_mC;
        if (det==0.) throw SBError("Attempt to SBTransform with degenerate matrix");
        _absdet = std::abs(det);
        _invdet = 1./det;

        double h1 = hypot( _mA+_mD, _mB-_mC);
        double h2 = hypot( _mA-_mD, _mB+_mC);
        _major = 0.5*std::abs(h1+h2);
        _minor = 0.5*std::abs(h1-h2);
        if (_major<_minor) std::swap(_major,_minor);
        _stillIsAxisymmetric = _adaptee.isAxisymmetric() 
            && (_mB==-_mC) 
            && (_mA==_mD)
            && (_cen.x==0.) && (_cen.y==0.); // Need pure rotation

        xdbg<<"Transformation init\n";
        xdbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        xdbg<<"_cen = "<<_cen<<std::endl;
        xdbg<<"_invdet = "<<_invdet<<std::endl;
        xdbg<<"_absdet = "<<_absdet<<std::endl;
        xdbg<<"_fluxScaling = "<<_fluxScaling<<std::endl;
        xdbg<<"_major, _minor = "<<_major<<", "<<_minor<<std::endl;
        xdbg<<"maxK() = "<<_adaptee.maxK() / _minor<<std::endl;
        xdbg<<"stepK() = "<<_adaptee.stepK() / _major<<std::endl;

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
        // At this point we are done with _absdet per se.  Multiply it by _fluxScaling
        // so we can use it as the scale factor for kValue and getFlux.
        _absdet *= _fluxScaling;
        xdbg<<"_absdet -> "<<_absdet<<std::endl;

        // Figure out which function we need for kValue and kValueNoPhase
        if (std::abs(_absdet-1.) < sbp::kvalue_accuracy) {
            xdbg<<"absdet*fluxScaling = "<<_absdet<<" = 1, so use NoDet version.\n";
            _kValueNoPhase = &SBTransform::_kValueNoPhaseNoDet;
        } else {
            xdbg<<"absdet*fluxScaling = "<<_absdet<<" != 1, so use WithDet version.\n";
            _kValueNoPhase = &SBTransform::_kValueNoPhaseWithDet;
        }
        if (_cen.x == 0. && _cen.y == 0.) _kValue = _kValueNoPhase;
        else _kValue = &SBTransform::_kValueWithPhase;
    }

    void SBTransform::SBTransformImpl::getXRange(
        double& xmin, double& xmax, std::vector<double>& splits) const
    {
        xmin = _xmin; xmax = _xmax;
        splits.insert(splits.end(),_xsplits.begin(),_xsplits.end());
    }

    void SBTransform::SBTransformImpl::getYRange(
        double& ymin, double& ymax, std::vector<double>& splits) const
    {
        ymin = _ymin; ymax = _ymax;
        splits.insert(splits.end(),_ysplits.begin(),_ysplits.end());
    }

    void SBTransform::SBTransformImpl::getYRangeX(
        double x, double& ymin, double& ymax, std::vector<double>& splits) const
    {
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

    // Specialization of fillKGrid is desired since the phase terms from shift 
    // are factorizable:
    void SBTransform::SBTransformImpl::fillKGrid(KTable& kt) const
    {
        int N = kt.getN();
        double dk = kt.getDk();

#if 0
        // The simpler version, saved for reference
        if (_cen.x==0. && _cen.y==0.) {
            // Branch to faster calculation if there is no centroid shift:
            for (int iy = -N/2; iy < N/2; iy++) {
                // only need ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValueNoPhase(k));
                }
            }
        } else {
            std::complex<double> dxexp(0,-dk*_cen.x),   dyexp(0,-dk*_cen.y);
            std::complex<double> dxphase(std::exp(dxexp)), dyphase(std::exp(dyexp));
            // xphase, yphase: current phase value
            std::complex<double> yphase(std::exp(-dyexp*N/2.));
            for (int iy = -N/2; iy < N/2; iy++) {
                std::complex<double> phase = yphase; // since kx=0 to start
                // Only ix>=0 since it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    Position<double> k(ix*dk,iy*dk);
                    kt.kSet(ix,iy,kValueNoPhase(k) * phase);
                    phase *= dxphase;
                }
                yphase *= dyphase;
            }
        }
#else
        // A faster version that pulls out all the if statements
        // and keeps track of fwdT(k) as we go

        dbg<<"Start Transformation fillKGrid\n";
        dbg<<"N = "<<N<<", dk = "<<dk<<std::endl;
        dbg<<"matrix = "<<_mA<<','<<_mB<<','<<_mC<<','<<_mD<<std::endl;
        dbg<<"_cen = "<<_cen<<std::endl;
        dbg<<"_invdet = "<<_invdet<<std::endl;
        dbg<<"_absdet = "<<_absdet<<std::endl;
        dbg<<"_fluxScaling = "<<_fluxScaling<<std::endl;

        if (_mA == 1. && _mB == 0. && _mC == 0. && _mD == 1. && 
            _cen.x == 0. && _cen.y == 0.) {
            dbg<<"Simple transformation.  Only flux scaling required.\n";
            dbg<<"Passing onto the adapteed.\n";
            // Then only a fluxScaling.  Call the adaptee's fillKGrid directly and rescale:
            SBProfile::GetImpl(_adaptee)->fillKGrid(kt);
            dbg<<"And now scale flux by "<<_absdet<<std::endl;
            kt *= _absdet;
            return;
        } 

        kt.clearCache();
        double dkA = dk*_mA;
        double dkB = dk*_mB;
        if (_cen.x==0. && _cen.y==0.) {
            dbg<<"No centroid shift, so no need to deal with phases.\n";
            // Branch to faster calculation if there is no centroid shift:
            Position<double> k1(0.,0.);
            Position<double> fwdTk1(0.,0.);
            for (int ix = 0; ix <= N/2; ix++, fwdTk1.x += dkA, fwdTk1.y += dkB) {
                // NB: the last two terms are not used by _kValueNoPhase,
                // so it's ok that k1.x is not kept up to date.
                kt.kSet2(ix,0,_kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
            }
            k1.y = dk; 
            Position<double> k2(0.,-dk);
            Position<double> fwdTk2;
            for (int iy = 1; iy < N/2; ++iy, k1.y += dk, k2.y -= dk) {
                fwdTk1 = fwdT(k1); fwdTk2 = fwdT(k2);
                for (int ix = 0; ix <= N/2; ++ix,
                     fwdTk1.x += dkA, fwdTk1.y += dkB, fwdTk2.x += dkA, fwdTk2.y += dkB) {
                    kt.kSet2(ix,iy, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
                    kt.kSet2(ix,N-iy, _kValueNoPhase(_adaptee,fwdTk2,_absdet,k2,_cen));
                }
            }
            fwdTk1 = fwdT(k1);
            for (int ix = 0; ix <= N/2; ix++, fwdTk1.x += dkA, fwdTk1.y += dkB) {
                kt.kSet2(ix,N/2,_kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen));
            }
        } else {
            dbg<<"Has centroid shift, so use phases.\n";
            std::complex<double> dxphase = std::polar(1.,-dk*_cen.x);
            std::complex<double> dyphase = std::polar(1.,-dk*_cen.y);
            // xphase, yphase: current phase value
            std::complex<double> yphase = 1.;
            Position<double> k1(0.,0.);
            Position<double> fwdTk1(0.,0.);
            std::complex<double> phase = yphase; // since kx=0 to start
            for (int ix = 0; ix <= N/2; ++ix,
                 fwdTk1.x += dkA, fwdTk1.y += dkB, phase *= dxphase) {
                kt.kSet2(ix,0, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
            }
            k1.y = dk; yphase *= dyphase;
            Position<double> k2(0.,-dk);  
            Position<double> fwdTk2;
            std::complex<double> phase2;
            for (int iy = 1; iy < N/2; iy++, k1.y += dk, k2.y -= dk, yphase *= dyphase) {
                fwdTk1 = fwdT(k1); fwdTk2 = fwdT(k2);
                phase = yphase; phase2 = conj(yphase);
                for (int ix = 0; ix <= N/2; ++ix,
                     fwdTk1.x += dkA, fwdTk1.y += dkB, fwdTk2.x += dkA, fwdTk2.y += dkB,
                     phase *= dxphase, phase2 *= dxphase) {
                    kt.kSet2(ix,iy, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
                    kt.kSet2(ix,N-iy, _kValueNoPhase(_adaptee,fwdTk2,_absdet,k1,_cen) * phase2);
                }
            }
            fwdTk1 = fwdT(k1);
            phase = yphase; 
            for (int ix = 0; ix <= N/2; ++ix, fwdTk1.x += dkA, fwdTk1.y += dkB, phase *= dxphase) {
                kt.kSet2(ix,N/2, _kValueNoPhase(_adaptee,fwdTk1,_absdet,k1,_cen) * phase);
            }
        }
#endif
    }

    std::complex<double> SBTransform::SBTransformImpl::kValue(const Position<double>& k) const
    { return _kValue(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBTransform::SBTransformImpl::kValueNoPhase(const Position<double>& k) const
    { return _kValueNoPhase(_adaptee,fwdT(k),_absdet,k,_cen); }

    std::complex<double> SBTransform::_kValueNoPhaseNoDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return adaptee.kValue(fwdTk); }

    std::complex<double> SBTransform::_kValueNoPhaseWithDet(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& , const Position<double>& )
    { return absdet * adaptee.kValue(fwdTk); }

    std::complex<double> SBTransform::_kValueWithPhase(
        const SBProfile& adaptee, const Position<double>& fwdTk, double absdet,
        const Position<double>& k, const Position<double>& cen)
    { return adaptee.kValue(fwdTk) * std::polar(absdet , -k.x*cen.x-k.y*cen.y); }

    boost::shared_ptr<PhotonArray> SBTransform::SBTransformImpl::shoot(
        int N, UniformDeviate u) const 
    {
        dbg<<"Distort shoot: N = "<<N<<std::endl;
        dbg<<"Target flux = "<<getFlux()<<std::endl;
        // Simple job here: just remap coords of each photon, then change flux
        // If there is overall magnification in the transform
        boost::shared_ptr<PhotonArray> result = _adaptee.shoot(N,u);
        for (int i=0; i<result->size(); i++) {
            Position<double> xy = fwd(Position<double>(result->getX(i), result->getY(i))+_cen);
            result->setPhoton(i,xy.x, xy.y, result->getFlux(i)*_absdet);
        }
        dbg<<"Distort Realized flux = "<<result->getTotalFlux()<<std::endl;
        return result;
    }
}
