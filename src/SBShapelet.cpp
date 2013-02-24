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

#include "SBShapelet.h"
#include "SBShapeletImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBShapelet::SBShapelet(double sigma, LVector bvec) :
        SBProfile(new SBShapeletImpl(sigma, bvec)) {}

    SBShapelet::SBShapelet(const SBShapelet& rhs) : SBProfile(rhs) {}

    SBShapelet::~SBShapelet() {}

    const LVector& SBShapelet::getBVec() const
    { 
        assert(dynamic_cast<const SBShapeletImpl*>(_pimpl.get()));
        return static_cast<const SBShapeletImpl&>(*_pimpl).getBVec(); 
    }

    double SBShapelet::getSigma() const 
    {
        assert(dynamic_cast<const SBShapeletImpl*>(_pimpl.get()));
        return static_cast<const SBShapeletImpl&>(*_pimpl).getSigma();
    }

    // ??? Have not really investigated these:
    double SBShapelet::SBShapeletImpl::maxK() const 
    {
        // Start with value for plain old Gaussian:
        double maxk = sqrt(-2.*std::log(sbp::maxk_threshold))/_sigma; 
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) maxk *= sqrt(double(_bvec.getOrder()));
        return maxk;
    }

    double SBShapelet::SBShapeletImpl::stepK() const 
    {
        // Start with value for plain old Gaussian:
        double R = std::max(4., sqrt(-2.*std::log(sbp::alias_threshold)));
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) R *= sqrt(double(_bvec.getOrder()));
        return M_PI / (R*_sigma);
    }

    double SBShapelet::SBShapeletImpl::xValue(const Position<double>& p) const 
    {
        LVector psi(_bvec.getOrder());
        psi.fillBasis(p.x/_sigma, p.y/_sigma, _sigma);
        double xval = _bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBShapelet::SBShapeletImpl::kValue(const Position<double>& k) const 
    {
        int N=_bvec.getOrder();
        LVector psi(N);
        psi.fillBasis(k.x*_sigma, k.y*_sigma);  // Fourier[Psi_pq] is unitless
        // rotate kvalues of Psi with i^(p+q)
        // dotting b_pq with psi in k-space:
        double rr=0.;
        double ii=0.;
        {
            for (PQIndex pq(0,0); !pq.pastOrder(N); pq.nextDistinct()) {
                int j = pq.rIndex();
                double x = _bvec[j]*psi[j] + (pq.isReal() ? 0 : _bvec[j+1]*psi[j+1]);
                switch (pq.N() % 4) {
                  case 0: 
                       rr += x;
                       break;
                  case 1: 
                       ii -= x;
                       break;
                  case 2: 
                       rr -= x;
                       break;
                  case 3: 
                       ii += x;
                       break;
                }
            }  
        }
        // difference in Fourier convention with FFTW ???
        return std::complex<double>(2.*M_PI*rr, 2.*M_PI*ii);
    }

    double SBShapelet::SBShapeletImpl::getFlux() const 
    {
        double flux=0.;
        for (PQIndex pp(0,0); !pp.pastOrder(_bvec.getOrder()); pp.incN())
            flux += _bvec[pp].real();  // _bvec[pp] is real, but need type conv.
        return flux;
    }

    double SBShapelet::SBShapeletImpl::getSigma() const { return _sigma; }
    const LVector& SBShapelet::SBShapeletImpl::getBVec() const { return _bvec; }

    template <typename T>
    void ShapeletFitImage(double sigma, LVector& bvec, const BaseImage<T>& image,
                          const Position<double>& center)
    {
        // TODO
    }



    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<double>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<float>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int32_t>& image,
        const Position<double>& center);
    template void ShapeletFitImage(
        double sigma, LVector& bvec, const BaseImage<int16_t>& image,
        const Position<double>& center);
}

