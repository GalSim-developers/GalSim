// -*- c++ -*-
#ifndef SBLAGUERRE_IMPL_H
#define SBLAGUERRE_IMPL_H

#include "SBProfileImpl.h"
#include "SBLaguerre.h"

namespace galsim {

    class SBLaguerre::SBLaguerreImpl : public SBProfile::SBProfileImpl 
    {
    public:
        SBLaguerreImpl(const LVector& bvec, double sigma) : 
            _bvec(bvec.duplicate()), _sigma(sigma) {}

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

