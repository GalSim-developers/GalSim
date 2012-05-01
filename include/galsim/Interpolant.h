
#ifndef INTERPOLANT_H
#define INTERPOLANT_H

#include <cmath>

#include "Std.h"
#include "Table.h"
#include "Random.h"
#include "PhotonArray.h"

namespace galsim {

    /** 
     * @brief One-dimensional interpolant base function
     *
     * One-dimensional interpolant function.  X units are in pixels
     * and the frequency-domain u values are in cycles per pixel.
     *
     * All Interpolants are assumed symmetric so that frequency-domain
     * values are real.
     */
    class Interpolant 
    {
    public:
        /// @brief Constructor
        Interpolant() {}
        /// @brief Destructor (virtual for base class)
        virtual ~Interpolant() {}

        // Extent of interpolant in real space and in frequency space.
        // Note that x units are pixels and u units are cycles per pixel.
        // Ranges are assumed to be same in x as in y.
        /**
         * @brief Maximum extent of interpolant from origin in x space (pixels)
         * @returns Range of non-zero values of interpolant.
         */
        virtual double xrange() const =0;
        /**
         * @brief Maximum extent of interpolant from origin in u space (cycles per pixel)
         * @returns Range of non-zero values of interpolant in u space.
         */
        virtual double urange() const =0;
        /**
         * @brief Value of interpolant in real space
         * @params[in] x Distance from sample (pixels)
         * @returns Value of interpolant
         */
        virtual double xval(double x) const =0;
        /**
         * @brief Value of interpolant, wrapped at period N.
         *
         * This returns sum_{j=-inf}^{inf} xval(x + jN):
         * @params[in] x Distance from sample (pixels)
         * @params[in] N Wrapping period (pixels)
         * @returns Value of interpolant after wrapping
         */
        virtual double xvalWrapped(double x, int N) const;
        /**
         * @brief Value of interpolant in frequency space
         * @params[in] u Frequency for evaluation (cycles per pixel)
         * @returns Value of interpolant, normalized so uval(0) = 1 for flux-conserving interpolation.
         */
        virtual double uval(double u) const =0;
        /**
         * @brief Report a generic indication of the accuracy to which Interpolant is calculated
         * @returns Targeted accuracy
         */
        virtual double getTolerance() const =0;  // report target accuracy

        /**
         * @brief Report whether interpolation will reproduce values at samples
         *
         * This will return true if the interpolant is exact at nodes, meaning
         * that F(0)=1 and F(n)=0 for non-zero integer n.  Right now this is true for
         * every implementation.
         * @returns True if samples are returned exactly.
         */
        virtual bool isExactAtNodes() const { return true; }

        // Photon-shooting routines:
        /**
         * @brief Return the integral of the positive portions of the kernel
         *
         * Should return the default value 1 unless the kernel has negative portions.
         *
         * @returns Integral of positive portions of kernel
         */
        virtual double getPositiveFlux() const {return 1.;}
        /**
         * @brief Return the (absolute value of) integral of the negative portions of the kernel
         *
         * Should return the default value 0 unless the kernel has negative portions.
         *
         * @returns Integral of abs value of negative portions of kernel
         */
        virtual double getNegativeFlux() const {return 0.;}
        /**
         * @brief Return array of displacements drawn from this kernel.  
         *
         * Since Interpolant is 1d, will use only x direction of PhotonArray.  It will be assumed
         * that photons returned are randomly ordered (no need to shuffle them).  Also assumed
         * that all photons will have nearly equal absolute value of flux.  Apart from possible shot
         * noise in number of positive and negative photons, the total flux returned will be unity.
         *
         * @params[in] N number of photons to shoot
         * @params[in] ud UniformDeviate used to generate random values
         * @returns a PhotonArray containing the vector of displacements for interpolation kernel.
         */
        virtual PhotonArray shoot(int N, UniformDeviate& ud) const {
            throw std::runtime_error("Interpolant::shoot() not implemented for this kernel");
            return PhotonArray(0);
        }
    };

    // Two-dimensional version
    class Interpolant2d 
    {
    public:
        Interpolant2d() {}
        virtual ~Interpolant2d() {}

        // Ranges are assumed to be same in x as in y.
        virtual double xrange() const=0;
        virtual double urange() const=0;
        virtual double xval(double x, double y) const=0;
        virtual double xvalWrapped(double x, double y, int N) const=0;
        virtual double uval(double u, double v) const=0;
        virtual double getTolerance() const=0;  // report target accuracy
        virtual bool isExactAtNodes() const { return true; }

        // Photon-shooting routines:
        // Return the integral of the positive portions of the kernel (default=1.)
        virtual double getPositiveFlux() const {return 1.;}
        // Return the (abs value of) integral of the negative portions of the kernel (default=0.)
        virtual double getNegativeFlux() const {return 0.;}
        // Return array of displacements drawn from this kernel.
        virtual PhotonArray shoot(int N, UniformDeviate& ud) const {
            throw std::runtime_error("Interpolant2d::shoot() not implemented for this kernel");
            return PhotonArray(0);
        }
    };

    // Instance of 2d that is product of 1d in x and y
    // Note that it only refers to the 1d function, does NOT own it
    class InterpolantXY : public Interpolant2d 
    {
    public:
        InterpolantXY(const Interpolant& i1d_) : i1d(i1d_) {}
        ~InterpolantXY() {}
        double xrange() const { return i1d.xrange(); }
        double urange() const { return i1d.urange(); }
        double xval(double x, double y) const { return i1d.xval(x)*i1d.xval(y); }
        double xvalWrapped(double x, double y, int N) const 
        { return i1d.xvalWrapped(x,N)*i1d.xvalWrapped(y,N); }
        double uval(double u, double v) const { return i1d.uval(u)*i1d.uval(v); }
        double getTolerance() const { return i1d.getTolerance(); }
        virtual bool isExactAtNodes() const { return i1d.isExactAtNodes(); }

        // Photon-shooting routines:
        double getPositiveFlux() const;
        double getNegativeFlux() const;
        PhotonArray shoot(int N, UniformDeviate& ud) const;

        // Give access to 1d functions for more efficient 2d interps:
        double xval1d(double x) const { return i1d.xval(x); }
        double xvalWrapped1d(double x, int N) const { return i1d.xvalWrapped(x,N); }
        double uval1d(double u) const { return i1d.uval(u); }
        const Interpolant* get1d() const {return &i1d;}

    private:
        const Interpolant& i1d;
    };

    // Some functions we will want: 
    // Note that sinc is defined here as sin(Pi*x) / (Pi*x).
    inline double sinc(double x) 
    {
        if (std::abs(x)<0.001) return 1.- M_PI*M_PI*x*x/6.;
        else return std::sin(M_PI*x)/(M_PI*x);
    }

    // Clever things from Daniel: integral of sin(t)/t from 0 to x
    // Note the official definition does not have pi multiplying t.
    inline double Si(double x) 
    {
        double x2=x*x;
        if(x2>=3.8) {
            // Use rational approximation from Abramowitz & Stegun
            // cf. Eqns. 5.2.38, 5.2.39, 5.2.8 - where it says it's good to <1e-6.
            // ain't this pretty?
            return M_PI/2.*((x>0)?1.:-1.) 
                -(38.102495+x2*(335.677320+x2*(265.187033+x2*(38.027264+x2))))
                / (x* (157.105423+x2*(570.236280+x2*(322.624911+x2*(40.021433+x2)))) ) * std::cos(x)
                -(21.821899+x2*(352.018498+x2*(302.757865+x2*(42.242855+x2))))
                / (x2*(449.690326+x2*(1114.978885+x2*(482.485984+x2*(48.196927+x2)))))*std::sin(x);

        } else {
            // x2<3.8: the series expansion is the better approximation, A&S 5.2.14
            double n1=1.;
            double n2=1.;
            double tt=x;
            double t=0;
            for(int i=1; i<7; i++) {
                t += tt/(n1*n2);
                tt = -tt*x2;
                n1 = 2.*double(i)+1.;
                n2*= n1*2.*double(i);
            }
            return t;
        }
    }

    /** 
     * @brief Delta-function interpolant in 1d
     *
     * The interpolant for when you do not want to interpolate
     * between samples.  Not really intended to be used for 
     * any analytic drawing because it's infinite in the x domain
     * at location of samples, and it extends to infinity in
     * the u domain.  But it could be useful for photon-shooting,
     * where it is trivially implemented as no displacements.
     * The argument in constructor is used to make a crude box
     * approximation to the x-space delta function and to give a
     * large but finite urange.
     *
     */
    class Delta : public Interpolant 
    {
    public:
        Delta(double width=1e-3) : _width(width) {}
        ~Delta() {}
        double xrange() const { return 0.; }
        double urange() const { return 1./_width; }
        double xval(double x) const 
        {
            if (std::abs(x)>0.5*_width) return 0.;
            else return 1./_width;
        }
        double uval(double u) const { return 1.; }

        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double _width;
    };

    // ****** Nearest neighbor interpolation: boxcar *****
    // Tolerance determines how far onto sinc wiggles the uval will go.
    // Very far, by default!
    class Nearest : public Interpolant 
    {
    public:
        Nearest(double tol=1e-3) : tolerance(tol) {}
        ~Nearest() {}
        double getTolerance() const { return tolerance; }
        double xrange() const { return 0.5; }
        double urange() const { return 1./(M_PI*tolerance); }
        double xval(double x) const 
        {
            if (std::abs(x)>0.5) return 0.;
            else if (std::abs(x)<0.5) return 1.;
            else return 0.5;
        }
        double uval(double u) const { return sinc(u); }

        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double tolerance;
    };

    // ****** Sinc interpolation: inverse of Nearest
    // Tolerance determines how far onto sinc wiggles the xval will go.
    // Very far, by default!
    class SincInterpolant : public Interpolant 
    {
    public:
        SincInterpolant(double tol=1e-3) : tolerance(tol) {}
        ~SincInterpolant() {}
        double getTolerance() const { return tolerance; }
        double xrange() const { return 1./(M_PI*tolerance); }
        double urange() const { return 0.5; }
        double uval(double u) const 
        {
            if (std::abs(u)>0.5) return 0.;
            else if (std::abs(u)<0.5) return 1.;
            else return 0.5;
        }
        double xval(double x) const { return sinc(x); }
        double xvalWrapped(double x, int N) const 
        {
            // Magic formula:
            x *= M_PI;
            if (N%2==0) {
                if (std::abs(x) < 1e-4) return 1. - x*x*(1/6.+1/2.-1./(6.*N*N));
                return std::sin(x) * std::cos(x/N) / (N*std::sin(x/N));
            } else {
                if (std::abs(x) < 1e-4) return 1. - x*x*(1-1./(N*N))/6.;
                return std::sin(x) / (N*std::sin(x/N));
            }
        }
    private:
        double tolerance;
    };

    class Linear : public Interpolant 
    {
    public:
        Linear(double tol=1e-3) : tolerance(tol) {}
        ~Linear() {}
        double getTolerance() const { return tolerance; }
        double xrange() const { return 1.-0.5*tolerance; }  // Snip off endpoints near zero
        double urange() const { return std::sqrt(1./tolerance)/M_PI; }
        double xval(double x) const 
        {
            x=std::abs(x);
            if (x>1.) return 0.;
            else return 1.-x;
        }
        double uval(double u) const { return std::pow(sinc(u),2.); }
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double tolerance;
    };

    // The Lanczos interpolation filter.
    // Need to choose its range n on input, and whether you want to have
    // it conserve flux (so that it's not quite Lanczos anymore).
    class Lanczos : public Interpolant 
    {
    public:
        Lanczos(int n_, bool fluxConserve_=false, double tol=1e-3) :  
            n(n_), fluxConserve(fluxConserve_), tolerance(tol), tab(Table<double,double>::spline) 
        { setup(); }

        ~Lanczos() {}

        // tol is error level desired for the Fourier transform
        double getTolerance() const { return tolerance; }
        double xrange() const { return range; }
        double urange() const { return uMax; }
        double xval(double x) const 
        { 
            x = std::abs(x);
            if (x>=n) return 0.;
            double retval = sinc(x)*sinc(x/n);
            if (fluxConserve) retval *= 1 + 2.*u1*(1-std::cos(2*M_PI*x));
            return retval;
        }
        double uval(double u) const 
        {
            u = std::abs(u);
            double retval = u>uMax ? 0. : tab(u);
            if (!fluxConserve) return retval;
            retval *= 1+2*u1;
            if (u+1 < uMax) retval -= u1*tab(u+1);
            if (std::abs(u-1) < uMax) retval -= u1*tab(std::abs(u-1));
            return retval;
        }
        double uCalc(double u) const;
    private:
        double n; // Note saving as double since it's used mostly this way.
        double range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        bool fluxConserve;   
        double tolerance;    
        double uMax;
        double u1; // coefficient for flux correction
        Table<double,double> tab;
        void setup();
    };

    // Cubic interpolator exact to 3rd order Taylor expansion
    // From R. G. Keys , IEEE Trans. Acoustics, Speech, & Signal Proc 29, p 1153, 1981

    class Cubic : public Interpolant 
    {
    public:
        Cubic(double tol=1e-4) : tolerance(tol), tab(Table<double,double>::spline) { setup(); }
        ~Cubic() {}
        // tol is error level desired for the Fourier transform
        double getTolerance() const { return tolerance; }
        double xrange() const { return range; }
        double urange() const { return uMax; }
        double xval(double x) const 
        { 
            x = std::abs(x);
            if (x>=2.) return 0.;
            if (x<1.) return 1 + x*x*(1.5*x-2.5);
            return 2 + x*(-4. + x*(2.5 - 0.5*x));
        }
        double uval(double u) const 
        {
            u = std::abs(u);
            return u>uMax ? 0. : tab(u);
        }
        double uCalc(double u) const;

	double getPositiveFlux() const {return 13./12.;}
	double getNegativeFlux() const {return 1./12.;}
	// ???? PhotonArray shoot(int N, UniformDeviate& ud) const;

    private:
        double range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        double tolerance;    
        double uMax;
        Table<double,double> tab;
        void setup();
    };

    // Cubic interpolator exact to 3rd order Taylor expansion
    // From R. G. Keys , IEEE Trans. Acoustics, Speech, & Signal Proc 29, p 1153, 1981

    class Quintic : public Interpolant 
    {
    public:
        Quintic(double tol=1e-4) : tolerance(tol), tab(Table<double,double>::spline) { setup(); }
        ~Quintic() {}
        // tol is error level desired for the Fourier transform
        double getTolerance() const { return tolerance; }
        double xrange() const { return range; }
        double urange() const { return uMax; }
        double xval(double x) const 
        { 
            x = std::abs(x);
            if (x>=3.) return 0.;
            if (x>=2.) return (x-2)*(x-3)*(x-3)*(-54+x*(50-11*x))/24.;
            if (x>=1.) return (x-1)*(x-2)*(-138.+x*(348+x*(-249.+55*x)))/24.;
            return 1 + x*x*x*(-95+x*(138-55*x))/12.;
        }
        double uval(double u) const 
        {
            u = std::abs(u);
            return u>uMax ? 0. : tab(u);
        }
        double uCalc(double u) const;
    private:
        double range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        double tolerance;    
        double uMax;
        Table<double,double> tab;
        void setup();
    };

}

#endif //INTERPOLANT_H
