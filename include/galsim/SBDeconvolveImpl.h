// -*- c++ -*-
#ifndef SBDECONVOLVE_IMPL_H
#define SBDECONVOLVE_IMPL_H

#include "SBProfileImpl.h"
#include "SBDeconvolve.h"

namespace galsim {

    class SBDeconvolve::SBDeconvolveImpl : public SBProfile::SBProfileImpl
    {
    public:
        SBDeconvolveImpl(const SBProfile& adaptee) : _adaptee(adaptee)
        { _maxksq = std::pow(maxK(),2.); }

        ~SBDeconvolveImpl() {}

        // xValue() not implemented for SBDeconvolve.
        double xValue(const Position<double>& p) const 
        { throw SBError("SBDeconvolve::xValue() not implemented"); }

        std::complex<double> kValue(const Position<double>& k) const 
        {
            return (k.x*k.x+k.y*k.y) <= _maxksq ?
                1./_adaptee.kValue(k) :
                std::complex<double>(0.,0.); 
        }

        double maxK() const { return _adaptee.maxK(); }
        double stepK() const { return _adaptee.stepK(); }

        bool isAxisymmetric() const { return _adaptee.isAxisymmetric(); }

        // Of course, a deconvolution could have hard edges, but since we can't use this
        // in a real-space convolution anyway, just return false here.
        bool hasHardEdges() const { return false; }

        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const { return -_adaptee.centroid(); }

        double getFlux() const { return 1./_adaptee.getFlux(); }

        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate u) const 
        {
            throw SBError("SBDeconvolve::shoot() not implemented");
            return boost::shared_ptr<PhotonArray>();
        }

    protected:

        // Override for better efficiency if adaptee has it:
        void fillKGrid(KTable& kt) const 
        {
            assert(SBProfile::GetImpl(_adaptee));
            SBProfile::GetImpl(_adaptee)->fillKGrid(kt);
            // Flip or clip:
            int N = kt.getN();
            int maxiksq = int(floor(_maxksq / (kt.getDk()*kt.getDk())));
            // Only need ix>=0 because it's Hermitian, but also
            // don't want to repeat the ix=0, N/2 twice:
            for (int iy = -N/2; iy < N/2; iy++) {
                if (iy>=0) {
                    int ix=0;
                    if (ix*ix+iy*iy <= maxiksq) 
                        kt.kSet(ix,iy,1./kt.kval(ix,iy));
                    else
                        kt.kSet(ix,iy,std::complex<double>(0.,0.));
                    ix=N/2;
                    if (ix*ix+iy*iy <= maxiksq) 
                        kt.kSet(ix,iy,1./kt.kval(ix,iy));
                    else
                        kt.kSet(ix,iy,std::complex<double>(0.,0.));
                }
                for (int ix = 0; ix <= N/2; ix++) {
                    if (ix*ix+iy*iy <= maxiksq) 
                        kt.kSet(ix,iy,1./kt.kval(ix,iy));
                    else
                        kt.kSet(ix,iy,std::complex<double>(0.,0.));
                }
            }
        }

    private:
        SBProfile _adaptee;
        double _maxksq;

        // Copy constructor and op= are undefined.
        SBDeconvolveImpl(const SBDeconvolveImpl& rhs);
        void operator=(const SBDeconvolveImpl& rhs);
    };

}

#endif // SBDECONVOLVE_IMPL_H
