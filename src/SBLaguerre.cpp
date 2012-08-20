
//#define DEBUGLOGGING

#include "SBLaguerre.h"
#include "SBLaguerreImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {


    SBLaguerre::SBLaguerre(LVector bvec, double sigma) : 
        SBProfile(new SBLaguerreImpl(bvec,sigma)) {}

    SBLaguerre::SBLaguerre(const SBLaguerre& rhs) : SBProfile(rhs) {}

    SBLaguerre::~SBLaguerre() {}

    // ??? Have not really investigated these:
    double SBLaguerre::SBLaguerreImpl::maxK() const 
    {
        // Start with value for plain old Gaussian:
        double maxk = sqrt(-2.*std::log(sbp::maxk_threshold))/_sigma; 
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) maxk *= sqrt(double(_bvec.getOrder()));
        return maxk;
    }

    double SBLaguerre::SBLaguerreImpl::stepK() const 
    {
        // Start with value for plain old Gaussian:
        double R = std::max(4., sqrt(-2.*std::log(sbp::alias_threshold)));
        // Grow as sqrt of order
        if (_bvec.getOrder() > 1) R *= sqrt(double(_bvec.getOrder()));
        return M_PI / (R*_sigma);
    }

    double SBLaguerre::SBLaguerreImpl::xValue(const Position<double>& p) const 
    {
        LVector psi(_bvec.getOrder());
        psi.fillBasis(p.x/_sigma, p.y/_sigma, _sigma);
        double xval = _bvec.dot(psi);
        return xval;
    }

    std::complex<double> SBLaguerre::SBLaguerreImpl::kValue(const Position<double>& k) const 
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

    double SBLaguerre::SBLaguerreImpl::getFlux() const 
    {
        double flux=0.;
        for (PQIndex pp(0,0); !pp.pastOrder(_bvec.getOrder()); pp.incN())
            flux += _bvec[pp].real();  // _bvec[pp] is real, but need type conv.
        return flux;
    }


}
