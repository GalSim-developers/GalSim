
#ifndef INTERPOLANT_H
#define INTERPOLANT_H

#include <cmath>

#include "Std.h"
#include "Table.h"
#include "Random.h"
#include "PhotonArray.h"
#include "OneDimensionalDeviate.h"

namespace galsim {

    class Interpolant;

    /**
     * @brief Class to interface an interpolant to the `OneDimensionalDeviate` class for photon-shooting
     */
    class InterpolantFunction: public FluxDensity {
    public:
        /**
         * @brief Constructor
         * @param[in] interp Interpolant (one-d) that we'll want to sample
         */
        InterpolantFunction(const Interpolant& interp): _interp(interp) {}
        /// @brief operator() will return the xval() of the `Interpolant`
        double operator()(double x) const;
        ~InterpolantFunction() {}
    private:
        const Interpolant& _interp;  ///< Interpolant being wrapped
    };

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
        Interpolant(): _interp(*this), _sampler(0) {}

        /// @brief Copy constructor: does not copy photon sampler, will need to rebuild.
        Interpolant(const Interpolant& rhs): _interp(*this), _sampler(0) {}

        /// @brief Destructor (virtual for base class).  Deletes photon sampler if it has been built.
        virtual ~Interpolant() {if (_sampler) delete _sampler;}

        /**
         * @brief Maximum extent of interpolant from origin in x space (pixels)
         * @returns Range of non-zero values of interpolant.
         */
        virtual double xrange() const =0;
        /**
         * @brief Maximum extent of interpolant from origin in u space (cycles per pixel)
         * @returns Range of non-zero values of interpolant in u space
         */
        virtual double urange() const =0;
        /**
         * @brief Value of interpolant in real space
         * @param[in] x Distance from sample (pixels)
         * @returns Value of interpolant
         */
        virtual double xval(double x) const =0;
        /**
         * @brief Value of interpolant, wrapped at period N.
         *
         * This returns sum_{j=-inf}^{inf} xval(x + jN):
         * @param[in] x Distance from sample (pixels)
         * @param[in] N Wrapping period (pixels)
         * @returns Value of interpolant after wrapping
         */
        virtual double xvalWrapped(double x, int N) const;
        /**
         * @brief Value of interpolant in frequency space
         * @param[in] u Frequency for evaluation (cycles per pixel)
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

        ////////// Photon-shooting routines:
        /**
         * @brief Return the integral of the positive portions of the kernel
         *
         * Should return 1 unless the kernel has negative portions.  Default is to ask
         * the numerical sampler for its stored value.
         *
         * @returns Integral of positive portions of kernel
         */
        virtual double getPositiveFlux() const {
            checkSampler();
            return _sampler->getPositiveFlux();
        }
        /**
         * @brief Return the (absolute value of) integral of the negative portions of the kernel
         *
         * Should return 0 unless the kernel has negative portions.   Default is to ask
         * the numerical sampler for its stored value.
         *
         * @returns Integral of abs value of negative portions of kernel
         */
        virtual double getNegativeFlux() const {
            checkSampler();
            return _sampler->getNegativeFlux();
        }
        /**
         * @brief Return array of displacements drawn from this kernel.  
         *
         * Since Interpolant is 1d, will use only x array of PhotonArray.  It will be assumed
         * that photons returned are randomly ordered (no need to shuffle them).  Also assumed
         * that all photons will have nearly equal absolute value of flux.  Total flux returned
         * may not equal 1 due to shot noise in negative/positive photons, and small fluctuations
         * in photon weights.
         *
         * @param[in] N number of photons to shoot
         * @param[in] ud UniformDeviate used to generate random values
         * @returns a PhotonArray containing the vector of displacements for interpolation kernel.
         */
        virtual PhotonArray shoot(int N, UniformDeviate& ud) const {
            checkSampler();
            return _sampler->shoot(N, ud);
        }
    protected:
        InterpolantFunction _interp; ///< The function to interface the Interpolant to sampler
        mutable OneDimensionalDeviate* _sampler;  ///< Class that draws photons from this Interpolant
        /// @brief Allocate photon sampler and do all of its pre-calculations
        virtual void checkSampler() const {
            if (_sampler) return;
            // Will assume by default that the Interpolant kernel changes sign at non-zero
            // integers, with one extremum in each integer range.
            int nKnots = static_cast<int> (ceil(xrange()));
            std::vector<double> ranges(2*nKnots);
            for (int i=1; i<=nKnots; i++) {
                double knot = std::min(1.*i, xrange());
                ranges[nKnots-i] = -knot;
                ranges[nKnots+i-1] = knot;
            }
            _sampler = new OneDimensionalDeviate(_interp, ranges);
        }
    };

    ///< @brief Two-dimensional version of the `Interpolant` interface.  Methods have same meaning as in 1d
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
        /// @brief Return the integral of the positive portions of the kernel (default=1.)
        virtual double getPositiveFlux() const {return 1.;}
        /// @brief Return the (abs value of) integral of the negative portions of the kernel (default=0.)
        virtual double getNegativeFlux() const {return 0.;}
        /// @brief Return array of displacements drawn from this kernel.  Default is to throw an runtime_error
        virtual PhotonArray shoot(int N, UniformDeviate& ud) const {
            throw std::runtime_error("Interpolant2d::shoot() not implemented for this kernel");
            return PhotonArray(0);
        }
    };

    /**
     * @brief An interpolant that is product of same 1d `Interpolant` in x and y
     *
     * Note that it only refers to the 1d function, does *not* own it, so the 1d must
     * be kept in existence.  Typically will create a given `Interpolant` once and use
     * for the whole program.
     */
    class InterpolantXY : public Interpolant2d 
    {
    public:
        /**
         * @brief Constructor
         *
         * Note that the referenced `Interpolant` is not copied, it must stay in existence.
         * @param[in] i1d_ One-dimensional `Interpolant` to be applied to x and y coordinates.
         */
        InterpolantXY(const Interpolant& i1d_) : i1d(i1d_) {}
        /// @brief Destructor
        ~InterpolantXY() {}
        // All of the calls below implement base class methods.
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

        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] x 1d argument
         * @returns 1d result
         */
        double xval1d(double x) const { return i1d.xval(x); }
        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] x 1d argument
         * @param[in] N wrapping period
         * @returns 1d result, wrapped at period N
         */
        double xvalWrapped1d(double x, int N) const { return i1d.xvalWrapped(x,N); }
        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] u 1d argument
         * @returns 1d result
         */
        double uval1d(double u) const { return i1d.uval(u); }
        /**
         * @brief Access the 1d interpolant 
         * @returns Pointer to the 1d `Interpolant` that this class uses.
         */
        const Interpolant* get1d() const {return &i1d;}

    private:
        const Interpolant& i1d;  ///< The 1d function used in both axes here.
    };

    // Some functions we will want: 
    /**
     * @brief sinc function, defined here as sin(Pi*x) / (Pi*x).
     * @param[in] x sinc argument
     * @returns sinc function
     */
    inline double sinc(double x) 
    {
        if (std::abs(x)<0.001) return 1.- M_PI*M_PI*x*x/6.;
        else return std::sin(M_PI*x)/(M_PI*x);
    }

    /**
     * @brief Function returning integral of sinc function.
     *
     * Clever things from Daniel: integral of sin(t)/t from 0 to x.
     * Note the official definition does not have pi multiplying t.
     * @param[in] x Upper limit of integral
     * @returns Integral of sin(t)/t from 0 to x (no pi factors)
     */
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
        /**
         * @brief Constructor
         * @param[in] width Width of tiny boxcar used to approximate delta function in real space.
         */
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

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const {return 1.;}
        double getNegativeFlux() const {return 0.;}
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double _width;
    };

    /**
     * @brief Nearest-neighbor interpolation: boxcar 
     *
     * Tolerance determines how far onto sinc wiggles the uval will go.
     * Very far, by default!
     */
    class Nearest : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc wiggles the uval will go. Very far, by default!
         */
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

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const {return 1.;}
        double getNegativeFlux() const {return 0.;}
        /// @brief Nearest-neighbor interpolant photon shooting is a simple UniformDeviate call.
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double tolerance;
    };

    /** 
     *@brief Sinc interpolation: inverse of Nearest-neighbor
     */
    class SincInterpolant : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc wiggles the xval will go. Very far, by default!
         */
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
        /// @brief Photon-shooting will be disabled for sinc function since wiggles will make it crazy
        PhotonArray shoot(int N, UniformDeviate& ud) const {
            throw std::runtime_error("Photon shooting is not practical with sinc Interpolant");
            return PhotonArray(N);
        }
    private:
        double tolerance;
    };

    /**
     * @brief Linear interpolant
     *
     */
    class Linear : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc^2 wiggles the kval will go. Very far, by default!
         */
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
        // Override the default numerical photon-shooting method
        double getPositiveFlux() const {return 1.;}
        double getNegativeFlux() const {return 0.;}
        /// @brief Linear interpolant has fast photon-shooting by adding two uniform deviates per axis.
        PhotonArray shoot(int N, UniformDeviate& ud) const;
    private:
        double tolerance;
    };

    /**
     * @brief The Lanczos interpolation filter, nominally sinc(x)*sinc(x/n), truncated at +-n.
     */
    class Lanczos : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         *
         * Note that pure Lanczos, when interpolating a set of constant-valued samples, does
         * not return this constant.  Setting fluxConserve tweaks the function so that it
         * conserves value of constant (DC) input data.
         * @param[in] n_ Filter order; must be given on input and cannot be changed.  
         * @param[in] fluxConserve_ Set true to adjust filter to be exact for constant inputs.
         * @param[in] tol Sets accuracy and extent of Fourier transform.
         */
        Lanczos(int n_, bool fluxConserve_=false, double tol=1e-3) :  
            n(n_), fluxConserve(fluxConserve_), tolerance(tol), tab(Table<double,double>::spline) 
        { setup(); }

        ~Lanczos() {}

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
        double n; ///< Actually storing 2n, since it's used mostly this way.
        double range; ///< Reduce range slightly from n so we're not using zero-valued endpoints.
        bool fluxConserve; ///< Set to insure conservation of constant (sky) flux
        double tolerance;  ///< k-space accuracy parameter
        double uMax;  ///< truncation point for Fourier transform
        double u1; ///< coefficient for flux correction
        Table<double,double> tab; ///< Table for Fourier transform
        void setup();
    };

    /**
     * @brief  Cubic interpolator exact to 3rd order Taylor expansion
     *
     * From R. G. Keys , IEEE Trans. Acoustics, Speech, & Signal Proc 29, p 1153, 1981
     */
    class Cubic : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] tol Sets accuracy and extent of Fourier transform.
         */
        Cubic(double tol=1e-4) : tolerance(tol), tab(Table<double,double>::spline) { setup(); }
        ~Cubic() {}

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

        /// @brief Override numerical calculation with known analytic integral
        double getPositiveFlux() const {return 13./12.;}
        /// @brief Override numerical calculation with known analytic integral
        double getNegativeFlux() const {return 1./12.;}

    private:
        double range; ///< x range, reduced slightly from n=2 so we're not using zero-valued endpoints.
        double tolerance;    
        double uMax;  ///< Truncation point for Fourier transform
        Table<double,double> tab; ///< Tabulated Fourier transform
        void setup();
    };

    /**
     * @brief Piecewise-quintic polynomial interpolant, ideal for k-space interpolation
     *
     * See Bernstein & Gruen (2012) for details
     */

    class Quintic : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Sets accuracy and extent of Fourier transform.
         */
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
    protected:
        /// @brief Override default sampler configuration because Quintic filter has sign change in outer interval
        virtual void checkSampler() const {
            if (_sampler) return;
            std::vector<double> ranges(8);
            ranges[0] = -3.;
            ranges[1] = -(25.+sqrt(31.))/11.;  // This is the extra zero-crossing
            ranges[2] = -2.;
            ranges[3] = -1.;
            for (int i=0; i<4; i++)
                ranges[7-i] = -ranges[i];
            _sampler = new OneDimensionalDeviate(_interp, ranges);
        }

    private:
        double range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        double tolerance;    
        double uMax;
        Table<double,double> tab;
        void setup();
    };

}

#endif //INTERPOLANT_H
