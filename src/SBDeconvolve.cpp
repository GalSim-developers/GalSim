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

//#define DEBUGLOGGING

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee,
                               boost::shared_ptr<GSParams> gsparams) :
        SBProfile(new SBDeconvolveImpl(adaptee,gsparams)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

    SBDeconvolve::SBDeconvolveImpl::SBDeconvolveImpl(
        const SBProfile& adaptee, boost::shared_ptr<GSParams> gsparams) :
        SBProfileImpl(gsparams.get() ? gsparams :
                      SBProfile::GetImpl(adaptee)->gsparams),
        _adaptee(adaptee)
    {
        double maxk = maxK();
        _maxksq = maxk*maxk;
        dbg<<"SBDeconvolve constructor: _maxksq = "<<_maxksq<<std::endl;
    }

    SBDeconvolve::SBDeconvolveImpl::~SBDeconvolveImpl() {}

    // xValue() not implemented for SBDeconvolve.
    double SBDeconvolve::SBDeconvolveImpl::xValue(const Position<double>& p) const 
    { throw SBError("SBDeconvolve::xValue() not implemented (and not possible)"); }

    std::complex<double> SBDeconvolve::SBDeconvolveImpl::kValue(const Position<double>& k) const 
    {
        return (k.x*k.x + k.y*k.y <= _maxksq) ? 1./_adaptee.kValue(k) : 0.;
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double x0, double dx, int ix_zero,
                                                    double y0, double dy, int iy_zero) const
    {
        dbg<<"SBDeconvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<", ix_zero = "<<ix_zero<<std::endl;
        dbg<<"y = "<<y0<<" + iy * "<<dy<<", iy_zero = "<<iy_zero<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,ix_zero,y0,dy,iy_zero);

        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,y0+=dy) {
            double x = x0;
            double ysq = y0*y0;
            for (int i=0;i<m;++i,x+=dx,++valit) 
                *valit = (x*x+ysq <= _maxksq) ? 1./(*valit) : 0.;
        }
    }

    void SBDeconvolve::SBDeconvolveImpl::fillKValue(tmv::MatrixView<std::complex<double> > val,
                                                    double x0, double dx, double dxy,
                                                    double y0, double dy, double dyx) const
    {
        dbg<<"SBDeconvolve fillKValue\n";
        dbg<<"x = "<<x0<<" + ix * "<<dx<<" + iy * "<<dxy<<std::endl;
        dbg<<"y = "<<y0<<" + ix * "<<dyx<<" + iy * "<<dy<<std::endl;
        GetImpl(_adaptee)->fillKValue(val,x0,dx,dxy,y0,dy,dyx);

        assert(val.stepi() == 1);
        assert(val.canLinearize());
        const int m = val.colsize();
        const int n = val.rowsize();
        typedef tmv::VIt<std::complex<double>,1,tmv::NonConj> It;
        It valit(val.linearView().begin().getP(),1);
        for (int j=0;j<n;++j,x0+=dxy,y0+=dy) {
            double x = x0;
            double y = y0;
            for (int i=0;i<m;++i,x+=dx,y+=dyx,++valit) 
                *valit = (x*x+y*y <= _maxksq) ? 1./(*valit) : 0.;
        }
    }

    Position<double> SBDeconvolve::SBDeconvolveImpl::centroid() const 
    { return -_adaptee.centroid(); }

    double SBDeconvolve::SBDeconvolveImpl::getFlux() const 
    { return 1./_adaptee.getFlux(); }

    boost::shared_ptr<PhotonArray> SBDeconvolve::SBDeconvolveImpl::shoot(
        int N, UniformDeviate u) const 
    {
        throw SBError("SBDeconvolve::shoot() not implemented");
        return boost::shared_ptr<PhotonArray>();
    }
>>>>>>> master

}
