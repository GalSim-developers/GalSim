
// SBProfile adapter which inverts its subject in k space to
// effecta  deconvolution

#ifndef SBDECONVOLVE_H
#define SBDECONVOLVE_H

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"

namespace sbp {

    class SBDeconvolve : public SBProfile 
    {
    public:
        SBDeconvolve(const SBProfile& adaptee_) : adaptee(adaptee_.duplicate()) 
        { maxksq = pow(maxK(),2.); }

        SBDeconvolve(const SBDeconvolve& rhs) : adaptee(rhs.adaptee->duplicate()) 
        { maxksq = pow(maxK(),2.); }

        ~SBDeconvolve() { delete adaptee; }

        SBProfile* duplicate() const { return new SBDeconvolve(*this); }

        // These are all the base class members that must be implemented:
        double xValue(Position<double> p) const 
        { throw SBError("SBDeconvolve::xValue() not implemented"); }

        std::complex<double> kValue(Position<double> p) const 
        {
            return (p.x*p.x+p.y*p.y) <= maxksq ?
                1./adaptee->kValue(p) :
                std::complex<double>(0.,0.); 
        }

        double maxK() const { return adaptee->maxK(); }

        // Require output FTs to be period on scale > original image extent + kernel footprint:
        double stepK() const { return adaptee->stepK(); }

        bool isAxisymmetric() const { return adaptee->isAxisymmetric(); }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        double centroidX() const { return -adaptee->centroidX(); }
        double centroidY() const { return -adaptee->centroidY(); }
        void setCentroid(Position<double> _p) 
        { throw SBError("setCentroid not allowed for SBDeconvolve"); }

        double getFlux() const { return 1./adaptee->getFlux(); }
        void setFlux(double flux=1.) { adaptee->setFlux(1./flux); }

        // Override for better efficiency if adaptee has it:
        virtual void fillKGrid(KTable& kt) const 
        {
            adaptee->fillKGrid(kt);
            // Flip or clip:
            int N = kt.getN();
            int maxiksq = maxksq / kt.getDk();
            for (int iy = -N/2; iy < N/2; iy++) {
                // Only need ix>=0 because it's Hermitian:
                for (int ix = 0; ix <= N/2; ix++) {
                    if (ix*ix+iy*iy <= maxiksq) 
                        kt.kSet(ix,iy,1./kt.kval(ix,iy));
                    else
                        kt.kSet(ix,iy,std::complex<double>(0.,0.));
                }
            }
        }

    private:
        SBProfile* adaptee;
        double maxksq;
    };

}

#endif // SBDECONVOLVE_H
