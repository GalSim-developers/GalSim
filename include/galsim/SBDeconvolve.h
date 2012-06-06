/** 
 * @file SBDeconvolve.h @brief SBProfile adapter which inverts its subject in k space to effect a
 * deconvolution.
 */

#ifndef SBDECONVOLVE_H
#define SBDECONVOLVE_H

#include "TMV.h"

#include "Std.h"
#include "SBProfile.h"
#include "Interpolant.h"

namespace galsim {

    /**
     * @brief SBProfile adapter which inverts its subject in k space to effect a deconvolvution.
     *
     * (TODO: Add more docs here!)
     */
    class SBDeconvolve : public SBProfile 
    {
    public:
        /// @brief Constructor.
        SBDeconvolve(const SBProfile& adaptee) : _adaptee(adaptee.duplicate()) 
        { _maxksq = std::pow(maxK(),2.); }

        /// @brief Copy constructor.
        SBDeconvolve(const SBDeconvolve& rhs) : _adaptee(rhs._adaptee->duplicate()) 
        { _maxksq = std::pow(maxK(),2.); }

        /// @brief Operator (TODO: ask Gary about this bit...)
        SBDeconvolve& operator=(const SBDeconvolve& rhs)
        {
            if (&rhs == this) return *this;
            if (_adaptee) {
                delete _adaptee; 
                _adaptee = 0;
            }
            _adaptee = rhs._adaptee->duplicate();
            _maxksq = rhs._maxksq;
            return *this;
        }

        /// @brief Destructor.
        ~SBDeconvolve() { delete _adaptee; }

        SBProfile* duplicate() const { return new SBDeconvolve(*this); }

        // These are all the base class members that must be implemented:

        /// @brief xValue() not implemented for SBDeconvolve.
        double xValue(const Position<double>& p) const 
        { throw SBError("SBDeconvolve::xValue() not implemented"); }

        std::complex<double> kValue(const Position<double>& k) const 
        {
            return (k.x*k.x+k.y*k.y) <= _maxksq ?
                1./_adaptee->kValue(k) :
                std::complex<double>(0.,0.); 
        }

        double maxK() const { return _adaptee->maxK(); }
        double stepK() const { return _adaptee->stepK(); }

        bool isAxisymmetric() const { return _adaptee->isAxisymmetric(); }

        // Of course, a deconvolution could have hard edges, but since we can't use this
        // in a real-space convolution anyway, just return false here.
        bool hasHardEdges() const { return false; }

        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }

        Position<double> centroid() const { return -_adaptee->centroid(); }

        double getFlux() const { return 1./_adaptee->getFlux(); }
        void setFlux(double flux=1.) { _adaptee->setFlux(1./flux); }

        PhotonArray shoot(int N, UniformDeviate& u) const 
        {
            throw SBError("SBDeconvolve::shoot() not implemented");
            return PhotonArray(N);
        }

    protected:

        // Override for better efficiency if adaptee has it:
        void fillKGrid(KTable& kt) const 
        {
            _adaptee->fillKGrid(kt);
            // Flip or clip:
            int N = kt.getN();
            int maxiksq = _maxksq / (kt.getDk()*kt.getDk());
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
        SBProfile* _adaptee;
        double _maxksq;
    };

}

#endif // SBDECONVOLVE_H
