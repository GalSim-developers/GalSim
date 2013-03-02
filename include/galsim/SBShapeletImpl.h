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

#ifndef SBSHAPELET_IMPL_H
#define SBSHAPELET_IMPL_H

#include "SBProfileImpl.h"
#include "SBShapelet.h"

namespace galsim {

    class SBShapelet::SBShapeletImpl : public SBProfile::SBProfileImpl 
    {
    public:
        SBShapeletImpl(double sigma, const LVector& bvec) :
            // Make a fresh copy of bvec, so we don't need to worry about the source changing
            // behind our backs.
            _sigma(sigma), _bvec(bvec.copy()) {}

        ~SBShapeletImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;


        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const;

        double getFlux() const;
        double getSigma() const;
        const LVector& getBVec() const;

        /// @brief Photon-shooting is not implemented for SBShapelet, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const 
        { throw SBError("SBShapelet::shoot() is not implemented"); }

        // Override for better efficiency:
        void xValue(tmv::VectorView<double> x, tmv::VectorView<double> y,
                    tmv::MatrixView<double> val) const;
        void xValue(tmv::MatrixView<double> x, tmv::MatrixView<double> y,
                    tmv::MatrixView<double> val) const;
        void kValue(tmv::VectorView<double> kx, tmv::VectorView<double> ky,
                    tmv::MatrixView<std::complex<double> > kval) const;
        void kValue(tmv::MatrixView<double> kx, tmv::MatrixView<double> ky,
                    tmv::MatrixView<std::complex<double> > kval) const;

    private:
        double _sigma;
        LVector _bvec;  

        // Copy constructor and op= are undefined.
        SBShapeletImpl(const SBShapeletImpl& rhs);
        void operator=(const SBShapeletImpl& rhs);
    };

}

#endif // SBSHAPELET_IMPL_H

