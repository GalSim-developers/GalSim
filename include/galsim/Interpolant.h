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

#ifndef INTERPOLANT_H
#define INTERPOLANT_H

#include <cmath>
#include <boost/shared_ptr.hpp>
#include <map>

#include "Std.h"
#include "Table.h"
#include "Random.h"
#include "PhotonArray.h"
#include "OneDimensionalDeviate.h"

namespace galsim {

    class Interpolant;

    /**
     * @brief Class to interface an interpolant to the `OneDimensionalDeviate` class for 
     * photon-shooting
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
     * @brief Base class representing one-dimensional interpolant functions
     *
     * One-dimensional interpolant function.  X units are in pixels and the frequency-domain u
     * values are in cycles per pixel.
     *
     * All Interpolants are assumed symmetric so that frequency-domain values are real.
     */
    class Interpolant 
    {
    public:
        /// @brief Constructor
        Interpolant(): _interp(*this) {}

        /// @brief Copy constructor: does not copy photon sampler, will need to rebuild.
        Interpolant(const Interpolant& rhs): _interp(*this) {}

        /// @brief Destructor 
        virtual ~Interpolant() {}

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
         * @returns Value of interpolant, normalized so uval(0) = 1 for flux-conserving
         * interpolation.
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
         * This will return true if the interpolant is exact at nodes, meaning that F(0)=1 and
         * F(n)=0 for non-zero integer n.  Right now this is true for every implementation.
         *
         * @returns True if samples are returned exactly.
         */
        virtual bool isExactAtNodes() const { return true; }

        ////////// Photon-shooting routines:
        /**
         * @brief Return the integral of the positive portions of the kernel
         *
         * Should return 1 unless the kernel has negative portions.  Default is to ask the numerical
         * sampler for its stored value.
         *
         * @returns Integral of positive portions of kernel
         */
        virtual double getPositiveFlux() const 
        { checkSampler(); return _sampler->getPositiveFlux(); }

        /**
         * @brief Return the (absolute value of) integral of the negative portions of the kernel
         *
         * Should return 0 unless the kernel has negative portions.  Default is to ask the numerical
         * sampler for its stored value.
         *
         * @returns Integral of abs value of negative portions of kernel
         */
        virtual double getNegativeFlux() const 
        { checkSampler(); return _sampler->getNegativeFlux(); }

        /**
         * @brief Return array of displacements drawn from this kernel.  
         *
         * Since Interpolant is 1d, will use only x array of PhotonArray.  It will be assumed that
         * photons returned are randomly ordered (no need to shuffle them).  Also assumed that all
         * photons will have nearly equal absolute value of flux.  Total flux returned may not equal
         * 1 due to shot noise in negative/positive photons, and small fluctuations in photon
         * weights.
         *
         * @param[in] N number of photons to shoot
         * @param[in] ud UniformDeviate used to generate random values
         * @returns a PhotonArray containing the vector of displacements for interpolation kernel.
         */
        virtual boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const 
        { checkSampler(); return _sampler->shoot(N, ud); }

    protected:
        InterpolantFunction _interp; ///< The function to interface the Interpolant to sampler

        /// Class that draws photons from this Interpolant
        mutable boost::shared_ptr<OneDimensionalDeviate> _sampler;  

        /// @brief Allocate photon sampler and do all of its pre-calculations
        virtual void checkSampler() const 
        {
            if (_sampler.get()) return;
            // Will assume by default that the Interpolant kernel changes sign at non-zero
            // integers, with one extremum in each integer range.
            int nKnots = int(ceil(xrange()));
            std::vector<double> ranges(2*nKnots);
            for (int i=1; i<=nKnots; i++) {
                double knot = std::min(double(i), xrange());
                ranges[nKnots-i] = -knot;
                ranges[nKnots+i-1] = knot;
            }
            _sampler.reset(new OneDimensionalDeviate(_interp, ranges));
        }
    };

    /**
     * @brief Two-dimensional version of the `Interpolant` interface.
     *
     * Methods have same meaning as in 1d.
     */
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
        virtual double getPositiveFlux() const { return 1.; }

        /**
         * @brief Return the (abs value of) integral of the negative portions of the kernel 
         * 
         * Default=0.
         */
        virtual double getNegativeFlux() const { return 0.; }

        /**
         * @brief Return array of displacements drawn from this kernel.
         *
         * Default is to throw an runtime_error
         */
        virtual boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const 
        {
            throw std::runtime_error("Interpolant2d::shoot() not implemented for this kernel");
            return boost::shared_ptr<PhotonArray>();
        }
    };

    /**
     * @brief An interpolant that is product of same 1d `Interpolant` in x and y
     *
     * The 1d interpolant gets passed in by shared_ptr, so there is no need to worry about keeping
     * the 1d interpolant in existence elsewhere.
     */
    class InterpolantXY : public Interpolant2d 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] i1d  One-dimensional `Interpolant` to be applied to x and y coordinates.
         */
        InterpolantXY(boost::shared_ptr<Interpolant> i1d) : _i1d(i1d) {}
        /// @brief Destructor
        ~InterpolantXY() {}
        // All of the calls below implement base class methods.
        double xrange() const { return _i1d->xrange(); }
        double urange() const { return _i1d->urange(); }
        double xval(double x, double y) const { return _i1d->xval(x)*_i1d->xval(y); }
        double xvalWrapped(double x, double y, int N) const 
        { return _i1d->xvalWrapped(x,N)*_i1d->xvalWrapped(y,N); }
        double uval(double u, double v) const { return _i1d->uval(u)*_i1d->uval(v); }
        double getTolerance() const { return _i1d->getTolerance(); }
        virtual bool isExactAtNodes() const { return _i1d->isExactAtNodes(); }

        // Photon-shooting routines:
        double getPositiveFlux() const;
        double getNegativeFlux() const;
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] x 1d argument
         * @returns 1d result
         */
        double xval1d(double x) const { return _i1d->xval(x); }

        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] x 1d argument
         * @param[in] N wrapping period
         * @returns 1d result, wrapped at period N
         */
        double xvalWrapped1d(double x, int N) const { return _i1d->xvalWrapped(x,N); }

        /**
         * @brief Access the 1d interpolant functions for more efficient 2d interps:
         * @param[in] u 1d argument
         * @returns 1d result
         */
        double uval1d(double u) const { return _i1d->uval(u); }

        /**
         * @brief Access the 1d interpolant 
         * @returns Pointer to the 1d `Interpolant` that this class uses.
         */
        const Interpolant* get1d() const { return _i1d.get(); }

    private:
        boost::shared_ptr<Interpolant> _i1d;  ///< The 1d function used in both axes here.
    };

    // Some functions we will want: 
    /**
     * @brief sinc function, defined here as sin(Pi*x) / (Pi*x).
     * @param[in] x sinc argument
     * @returns sinc function
     */
    inline double sinc(double x) 
    {
        if (std::abs(x)<0.001) return 1.- (M_PI*M_PI/6.)*x*x;
        else return std::sin(M_PI*x)/(M_PI*x);
    }

    /**
     * @brief Function returning integral of sinc function.
     *
     * Utility for calculating the integral of sin(t)/t from 0 to x.  Note the official definition
     * does not have pi multiplying t.
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
            return (M_PI/2.)*((x>0.)?1.:-1.) 
                - (38.102495+x2*(335.677320+x2*(265.187033+x2*(38.027264+x2))))
                / (x* (157.105423+x2*(570.236280+x2*(322.624911+x2*(40.021433+x2)))) )*std::cos(x)
                - (21.821899+x2*(352.018498+x2*(302.757865+x2*(42.242855+x2))))
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
     * The interpolant for when you do not want to interpolate between samples.  It is not really
     * intended to be used for any analytic drawing because it is infinite in the x domain at the
     * location of samples, and it extends to infinity in the u domain.  But it could be useful for
     * photon-shooting, where it is trivially implemented as no displacements.  The argument in the
     * constructor is used to make a crude box approximation to the x-space delta function and to
     * give a large but finite urange.
     *
     */
    class Delta : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] width Width of tiny boxcar used to approximate delta function in real space.
         */
        Delta(double width=1.e-3) : _width(width) {}
        ~Delta() {}
        double xrange() const { return 0.; }
        double urange() const { return 1./_width; }
        double xval(double x) const 
        {
            if (std::abs(x)>0.5*_width) return 0.;
            else return 1./_width;
        }
        double uval(double u) const { return 1.; }
        double getTolerance() const { return _width; }

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;
    private:
        double _width;
    };

    /**
     * @brief Nearest-neighbor interpolation: boxcar 
     *
     * The nearest-neighbor interpolant performs poorly as a k-space or x-space interpolant for
     * SBInterpolatedImage.  (See document by Bernstein & Gruen, devel/modules/finterp.pdf in the
     * GalSim repository.)  The objection to its use in Fourier space does not apply when shooting
     * photons to generate an image; in that case, the nearest-neighbor interpolant is quite
     * efficient (but not necessarily the best choice in terms of accuracy).
     *
     * Tolerance determines how far onto sinc wiggles the uval will go.  Very far, by default!
     */
    class Nearest : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc wiggles the uval will go.
         * Very far, by default!
         */
        Nearest(double tol=1.e-3) : _tolerance(tol) {}
        ~Nearest() {}
        double getTolerance() const { return _tolerance; }
        double xrange() const { return 0.5; }
        double urange() const { return 1./(M_PI*_tolerance); }
        double xval(double x) const 
        {
            if (std::abs(x)>0.5) return 0.;
            else if (std::abs(x)<0.5) return 1.;
            else return 0.5;
        }
        double uval(double u) const { return sinc(u); }

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        /// @brief Nearest-neighbor interpolant photon shooting is a simple UniformDeviate call.
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;
    private:
        double _tolerance;
    };

    /** 
     *@brief Sinc interpolation: inverse of Nearest-neighbor
     *
     * The Sinc interpolant (K(x) = sin(pi x)/(pi x)) is mathematically perfect for band-limited
     * data, introducing no spurious frequency content beyond kmax = pi/dx for input data with pixel
     * scale dx.  However, it is formally infinite in extent and, even with reasonable trunction, is
     * still quite large.  It will give exact results in SBInterpolatedImage::kValue() when it is
     * used as a k-space interpolant, but is extremely slow.  The usual compromise between sinc
     * accuracy vs. speed is the Lanczos interpolant (see its documentation for details).
     */
    class SincInterpolant : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc wiggles the xval will go. 
         * Very far, by default!
         */
        SincInterpolant(double tol=1.e-3) : _tolerance(tol) {}
        ~SincInterpolant() {}
        double getTolerance() const { return _tolerance; }
        double xrange() const { return 1./(M_PI*_tolerance); }
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
                if (std::abs(x) < 1.e-4) return 1. - x*x*(1/6.+1/2.-1./(6.*N*N));
                return std::sin(x) * std::cos(x/N) / (N*std::sin(x/N));
            } else {
                if (std::abs(x) < 1.e-4) return 1. - (1./6.)*x*x*(1-1./(N*N));
                return std::sin(x) / (N*std::sin(x/N));
            }
        }
        /**
         * @brief Photon-shooting will be disabled for sinc function since wiggles will make it
         * crazy
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const 
        {
            throw std::runtime_error("Photon shooting is not practical with sinc Interpolant");
            return boost::shared_ptr<PhotonArray>();
        }
    private:
        double _tolerance;
    };

    /**
     * @brief Linear interpolant
     *
     * The linear interpolant is a poor choice for FFT-based operations on SBInterpolatedImage, as
     * it rings to high frequencies.  (See Bernstein & Gruen, devel/modules/finterp.pdf in the
     * GalSim repository.)  This objection does not apply when shooting photons, in which case the
     * linear interpolant is quite efficient (but not necessarily the best choice in terms of
     * accuracy).
     */
    class Linear : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Tolerance determines how far onto sinc^2 wiggles the kval will go.
         * Very far, by default!
         */
        Linear(double tol=1.e-3) : _tolerance(tol) {}
        ~Linear() {}
        double getTolerance() const { return _tolerance; }
        double xrange() const { return 1.-0.5*_tolerance; }  // Snip off endpoints near zero
        double urange() const { return std::sqrt(1./_tolerance)/M_PI; }
        double xval(double x) const 
        {
            x=std::abs(x);
            if (x>1.) return 0.;
            else return 1.-x;
        }
        double uval(double u) const { return std::pow(sinc(u),2.); }
        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        /**
         * @brief Linear interpolant has fast photon-shooting by adding two uniform deviates per
         * axis.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;
    private:
        double _tolerance;
    };

    /**
     * @brief The Lanczos interpolation filter, nominally sinc(x)*sinc(x/n), truncated at +/-n.
     *
     * The Lanczos filter is an approximation to the band-limiting sinc filter with a smooth cutoff
     * at high x.  Order n Lanczos has a range of +/- n pixels.  It typically is a good compromise
     * between kernel size and accuracy.
     *
     * The filter has accuracy parameters `xvalue_accuracy` and `kvalue_accuracy` that relate to the
     * accuracy of building the initial lookup table.  For now, these are fixed in
     * src/Interpolant.cpp to be 0.1 times the input `tol` value, where `tol` is typically very
     * small already (default 1e-4).
     *
     * Note that pure Lanczos, when interpolating a set of constant-valued samples, does not return
     * this constant.  Setting fluxConserve in the constructor tweaks the function so that it 
     * approximately conserves the value of constant (DC) input data.
     * Only the first order correction is applied, which should be accurate to about 1.e-5.
     */
    class Lanczos : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] n  Filter order; must be given on input and cannot be changed.  
         * @param[in] fluxConserve  Set true to adjust filter to be more nearly correct for 
         *                          constant inputs.
         * @param[in] tol  Sets accuracy and extent of Fourier transform.
         */
        Lanczos(int n, bool fluxConserve=true, double tol=1.e-4);
        ~Lanczos() {}

        double getTolerance() const { return _tolerance; }
        double xrange() const { return _range; }
        double urange() const { return _uMax; }
        double xval(double x) const;
        double uval(double u) const;

    private:
        double _n; ///< Actually storing 2n, since it's used mostly this way.
        double _range; ///< Reduce range slightly from n so we're not using zero-valued endpoints.
        bool _fluxConserve; ///< Set to insure conservation of constant (sky) flux
        double _tolerance;  ///< k-space accuracy parameter
        double _uMax;  ///< truncation point for Fourier transform
        double _u1; ///< coefficient for flux correction
        boost::shared_ptr<Table<double,double> > _xtab; ///< Table for x values
        boost::shared_ptr<Table<double,double> > _utab; ///< Table for Fourier transform
        double xCalc(double x) const;
        double uCalc(double u) const;

        // Store the tables in a map, so repeat constructions are quick.
        typedef std::pair<int,std::pair<bool,double> > KeyType;
        static std::map<KeyType,boost::shared_ptr<Table<double,double> > > _cache_xtab; 
        static std::map<KeyType,boost::shared_ptr<Table<double,double> > > _cache_utab; 
        static std::map<KeyType,double> _cache_umax; 
    };

    /**
     * @brief Cubic interpolator exact to 3rd order Taylor expansion
     *
     * From R. G. Keys, IEEE Trans. Acoustics, Speech, & Signal Proc 29, p 1153, 1981
     *
     * The cubic interpolant is a reasonable choice for a four-point interpolant for
     * SBInterpolatedImage.   (See Bernstein & Gruen, devel/modules/finterp.pdf in the
     * GalSim repository.)
     */
    class Cubic : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] tol Sets accuracy and extent of Fourier transform.
         */
        Cubic(double tol=1.e-4);
        ~Cubic() {}

        double getTolerance() const { return _tolerance; }
        double xrange() const { return _range; }
        double urange() const { return _uMax; }
        double xval(double x) const 
        { 
            x = std::abs(x);
            if (x>=2.) return 0.;
            if (x<1.) return 1. + x*x*(1.5*x-2.5);
            return 2. + x*(-4. + x*(2.5 - 0.5*x));
        }
        double uval(double u) const 
        {
            u = std::abs(u);
            return u>_uMax ? 0. : (*_tab)(u);
        }
        double uCalc(double u) const;

        /// @brief Override numerical calculation with known analytic integral
        double getPositiveFlux() const { return 13./12.; }
        /// @brief Override numerical calculation with known analytic integral
        double getNegativeFlux() const { return 1./12.; }

    private:
        /// x range, reduced slightly from n=2 so we're not using zero-valued endpoints.
        double _range; 

        double _tolerance;    
        boost::shared_ptr<Table<double,double> > _tab; ///< Tabulated Fourier transform
        double _uMax;  ///< Truncation point for Fourier transform

        // Store the tables in a map, so repeat constructions are quick.
        static std::map<double,boost::shared_ptr<Table<double,double> > > _cache_tab; 
        static std::map<double,double> _cache_umax; 
    };

    /**
     * @brief Piecewise-quintic polynomial interpolant, ideal for k-space interpolation
     *
     * See Bernstein & Gruen, devel/modules/finterp.pdf in the GalSim repository.
     */

    class Quintic : public Interpolant 
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol Sets accuracy and extent of Fourier transform.
         */
        Quintic(double tol=1.e-4);
        ~Quintic() {}

        double getTolerance() const { return _tolerance; }
        double xrange() const { return _range; }
        double urange() const { return _uMax; }
        double xval(double x) const 
        { 
            x = std::abs(x);
            if (x <= 1.)
                return 1. + (1./12.)*x*x*x*(-95.+x*(138.-55.*x));
            else if (x <= 2.)
                return (1./24.)*(x-1.)*(x-2.)*(-138.+x*(348.+x*(-249.+55.*x)));
            else if (x <= 3.)
                return (1./24.)*(x-2.)*(x-3.)*(x-3.)*(-54.+x*(50.-11.*x));
            else 
                return 0.;
        }
        double uval(double u) const 
        {
            u = std::abs(u);
            return u>_uMax ? 0. : (*_tab)(u);
        }
        double uCalc(double u) const;

    protected:
        /**
         * @brief Override default sampler configuration because Quintic filter has sign change in
         * outer interval
         */
        virtual void checkSampler() const 
        {
            if (_sampler.get()) return;
            std::vector<double> ranges(8);
            ranges[0] = -3.;
            ranges[1] = -(1./11.)*(25.+sqrt(31.));  // This is the extra zero-crossing
            ranges[2] = -2.;
            ranges[3] = -1.;
            for (int i=0; i<4; i++)
                ranges[7-i] = -ranges[i];
            _sampler.reset(new OneDimensionalDeviate(_interp, ranges));
        }

    private:
        double _range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        double _tolerance;    
        boost::shared_ptr<Table<double,double> > _tab; ///< Tabulated Fourier transform
        double _uMax;  ///< Truncation point for Fourier transform

        // Store the tables in a map, so repeat constructions are quick.
        static std::map<double,boost::shared_ptr<Table<double,double> > > _cache_tab; 
        static std::map<double,double> _cache_umax; 
    };

}

#endif //INTERPOLANT_H
