
// SBProfile adapter which inverts its subject in k space to
// effecta  deconvolution

#ifndef SBDECONVOLVE_H
#define SBDECONVOLVE_H

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"

namespace galsim {

    class SBDeconvolve : public SBProfile 
    {
    public:
        SBDeconvolve(const SBProfile& adaptee_) : adaptee(adaptee_.duplicate()) 
            { maxksq = pow(maxK(),2.); }
        
        SBDeconvolve(const SBDeconvolve& rhs) : adaptee(rhs.adaptee->duplicate()) 
            { maxksq = pow(maxK(),2.); }

        SBDeconvolve& operator=(const SBDeconvolve& rhs)
            {
                if (&rhs == this) return *this;
                if (adaptee) {
                    delete adaptee; 
                    adaptee = 0;
                }
                adaptee = rhs.adaptee->duplicate();
                maxksq = rhs.maxksq;
                return *this;
            }

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
            int maxiksq = maxksq / (kt.getDk()*kt.getDk());
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
        SBProfile* adaptee;
        double maxksq;
    };

}

#endif // SBDECONVOLVE_H
