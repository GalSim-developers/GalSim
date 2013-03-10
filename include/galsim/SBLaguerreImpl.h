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

#ifndef SBLAGUERRE_IMPL_H
#define SBLAGUERRE_IMPL_H

#include "SBProfileImpl.h"
#include "SBLaguerre.h"

namespace galsim {

    class SBLaguerre::SBLaguerreImpl : public SBProfile::SBProfileImpl 
    {
    public:
        SBLaguerreImpl(const LVector& bvec, double sigma, boost::shared_ptr<GSParams> gsparams);
        ~SBLaguerreImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const 
        { throw SBError("SBLaguerre::centroid calculations not yet implemented"); }

        double getFlux() const;

        /// @brief Photon-shooting is not implemented for SBLaguerre, will throw an exception.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const 
        { throw SBError("SBLaguerre::shoot() is not implemented"); }

    private:
        /// `bvec[n,n]` contains flux information for the `(n, n)` basis function.
        LVector _bvec;  

        double _sigma;  ///< Scale size of Gauss-Laguerre basis set.

        // Copy constructor and op= are undefined.
        SBLaguerreImpl(const SBLaguerreImpl& rhs);
        void operator=(const SBLaguerreImpl& rhs);
    };

}

#endif // SBLAGUERRE_IMPL_H

