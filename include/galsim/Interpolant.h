/* -*- c++ -*-
 * Copyright (c) 2012-2018 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef GalSim_Interpolant_H
#define GalSim_Interpolant_H

#include <cmath>
#include <map>

#include "Std.h"
#include "Table.h"
#include "Random.h"
#include "PhotonArray.h"
#include "OneDimensionalDeviate.h"
#include "SBProfile.h"

namespace galsim {

    class Interpolant;

    /**
     * @brief Class to interface an interpolant to the `OneDimensionalDeviate` class for
     * photon-shooting
     */
    class InterpolantFunction: public FluxDensity {
    public:
        InterpolantFunction(const Interpolant& interp): _interp(interp) {}
        double operator()(double x) const; // returns the xval() of the `Interpolant`
        ~InterpolantFunction() {}
    private:
        const Interpolant& _interp;  // Interpolant being wrapped
    };

    /**
     * @brief Base class representing one-dimensional interpolant functions
     *
     * One-dimensional interpolant function.  X units are in pixels and the frequency-domain u
     * values are in cycles per pixel.  This differs from the usual definition of k in radians
     * per arcsec, hence the different letter variable.
     *
     * All Interpolants are assumed symmetric so that frequency-domain values are real.
     */
    class Interpolant
    {
    public:
        /**
         * @brief Constructor
         * @param[in] gsparams  GSParams object storing constants that control the accuracy of
         *                      operations, if different from the default.
         */
        Interpolant(const GSParams& gsparams) : _gsparams(gsparams), _interp(*this) {}

        /// @brief Copy constructor: does not copy photon sampler, will need to rebuild.
        Interpolant(const Interpolant& rhs): _gsparams(rhs._gsparams), _interp(rhs._interp) {}

        /// @brief Destructor
        virtual ~Interpolant() {}

        /**
         * @brief Maximum extent of interpolant from origin in x space (pixels)
         * @returns Range of non-zero values of interpolant.
         */
        virtual double xrange() const =0;

        /**
         * @brief The total range as an integer.  Typically xrange() == 0.5 * ixrange().
         */
        virtual int ixrange() const =0;

        /**
         * @brief Maximum extent of interpolant from origin in u space (cycles per pixel)
         * @returns Range of non-zero values of interpolant in u space
         */
        virtual double urange() const =0;

        /**
         * @brief Report a generic indication of the accuracy to which Interpolant is calculated
         * @returns Targeted accuracy
         */
        virtual double getTolerance() const =0;  // report target accuracy

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
        virtual void shoot(PhotonArray& photons, UniformDeviate ud) const
        { checkSampler(); _sampler->shoot(photons, ud, true); }

        virtual std::string makeStr() const =0;

    protected:

        GSParams _gsparams;
        InterpolantFunction _interp;

        // Class that draws photons from this Interpolant
        mutable shared_ptr<OneDimensionalDeviate> _sampler;

        // Allocate photon sampler and do all of its pre-calculations
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
            _sampler.reset(new OneDimensionalDeviate(_interp, ranges, false, _gsparams));
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
        virtual int ixrange() const=0;
        virtual double urange() const=0;
        virtual double getTolerance() const=0;

        virtual double xval(double x, double y) const=0;
        virtual double xvalWrapped(double x, double y, int N) const=0;
        virtual double uval(double u, double v) const=0;
        virtual bool isExactAtNodes() const=0;

        virtual double getPositiveFlux() const=0;
        virtual double getNegativeFlux() const=0;
        virtual void shoot(PhotonArray& photons, UniformDeviate ud) const=0;
    };

    /**
     * @brief An interpolant that is product of same 1d `Interpolant` in x and y
     *
     * The 1d interpolant gets passed in by reference, so it needs to exist elsewhere.
     * (Typically in Python layer.)
     */
    class InterpolantXY : public Interpolant2d
    {
    public:
        InterpolantXY(const Interpolant& i1d) : _i1d(i1d) {}
        ~InterpolantXY() {}

        // All of the calls below implement base class methods.
        double xrange() const { return _i1d.xrange(); }
        int ixrange() const { return _i1d.ixrange(); }
        double urange() const { return _i1d.urange(); }
        double getTolerance() const { return _i1d.getTolerance(); }

        double xval(double x, double y) const { return _i1d.xval(x)*_i1d.xval(y); }
        double xvalWrapped(double x, double y, int N) const
        { return _i1d.xvalWrapped(x,N)*_i1d.xvalWrapped(y,N); }
        double uval(double u, double v) const { return _i1d.uval(u)*_i1d.uval(v); }
        bool isExactAtNodes() const { return _i1d.isExactAtNodes(); }

        // Photon-shooting routines:
        double getPositiveFlux() const;
        double getNegativeFlux() const;
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        // Access the 1d interpolant functions for more efficient 2d interps:
        double xval1d(double x) const { return _i1d.xval(x); }
        double xvalWrapped1d(double x, int N) const { return _i1d.xvalWrapped(x,N); }
        double uval1d(double u) const { return _i1d.uval(u); }
        const Interpolant& get1d() const { return _i1d; }

    private:
        const Interpolant& _i1d;  // The 1d function used in both axes here.
    };

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
         * @param[in] width    Width of tiny boxcar used to approximate delta function in real
         *                     space (default=1.e-3).
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        Delta(double width, const GSParams& gsparams) :
            Interpolant(gsparams), _width(width) {}
        ~Delta() {}

        double xrange() const { return 0.; }
        int ixrange() const { return 0; }
        double urange() const { return 1./_width; }
        double getTolerance() const { return _width; }

        double xval(double x) const
        {
            if (std::abs(x)>0.5*_width) return 0.;
            else return 1./_width;
        }
        double uval(double u) const { return 1.; }

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        std::string makeStr() const;

    private:
        double _width;
    };

    /**
     * @brief Nearest-neighbor interpolation: boxcar
     *
     * The nearest-neighbor interpolant performs poorly as a k-space or x-space interpolant for
     * SBInterpolatedImage.  (See paper by Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
     * The objection to its use in Fourier space does not apply when shooting photons to generate an
     * image; in that case, the nearest-neighbor interpolant is quite efficient (but not necessarily
     * the best choice in terms of accuracy).
     *
     * Tolerance determines how far onto sinc wiggles the uval will go.  Very far, by default!
     */
    class Nearest : public Interpolant
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol      Tolerance determines how far onto sinc wiggles the uval will go.
         *                     Very far, by default!
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        Nearest(double tol, const GSParams& gsparams) :
            Interpolant(gsparams), _tolerance(tol) {}
        ~Nearest() {}

        double xrange() const { return 0.5; }
        int ixrange() const { return 1; }
        double urange() const { return 1./(M_PI*_tolerance); }
        double getTolerance() const { return _tolerance; }

        double xval(double x) const;
        double uval(double u) const;

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        std::string makeStr() const;

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
         * @param[in] tol      Tolerance determines how far onto sinc wiggles the xval will go.
         *                     Very far, by default!
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        SincInterpolant(double tol, const GSParams& gsparams) :
            Interpolant(gsparams), _tolerance(tol) {}
        ~SincInterpolant() {}

        double xrange() const { return 1./(M_PI*_tolerance); }
        int ixrange() const { return 0; }
        double urange() const { return 0.5; }
        double getTolerance() const { return _tolerance; }

        double xval(double x) const;
        double xvalWrapped(double x, int N) const;
        double uval(double u) const;

        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        std::string makeStr() const;

    private:
        double _tolerance;
    };

    /**
     * @brief Linear interpolant
     *
     * The linear interpolant is a poor choice for FFT-based operations on SBInterpolatedImage, as
     * it rings to high frequencies.  (See paper by Bernstein & Gruen,
     * http://arxiv.org/abs/1401.2636.)  This objection does not apply when shooting photons, in
     * which case the linear interpolant is quite efficient (but not necessarily the best choice in
     * terms of accuracy).
     */
    class Linear : public Interpolant
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol      Tolerance determines how far onto sinc^2 wiggles the uval will go.
         *                     Very far, by default!
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        Linear(double tol, const GSParams& gsparams) :
            Interpolant(gsparams), _tolerance(tol) {}
        ~Linear() {}

        double xrange() const { return 1.-0.5*_tolerance; }  // Snip off endpoints near zero
        int ixrange() const { return 2; }
        double urange() const { return std::sqrt(1./_tolerance)/M_PI; }
        double getTolerance() const { return _tolerance; }

        double xval(double x) const;
        double uval(double u) const;

        // Override the default numerical photon-shooting method
        double getPositiveFlux() const { return 1.; }
        double getNegativeFlux() const { return 0.; }
        // Linear interpolant has fast photon-shooting by adding two uniform deviates per
        void shoot(PhotonArray& photons, UniformDeviate ud) const;

        std::string makeStr() const;

    private:
        double _tolerance;
    };

    /**
     * @brief Cubic interpolator exact to 3rd order Taylor expansion
     *
     * From R. G. Keys, IEEE Trans. Acoustics, Speech, & Signal Proc 29, p 1153, 1981
     *
     * The cubic interpolant is a reasonable choice for a four-point interpolant for
     * SBInterpolatedImage.   (See paper by Bernstein & Gruen, http://arxiv.org/abs/1401.2636.)
     */
    class Cubic : public Interpolant
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] tol      Sets accuracy and extent of Fourier transform.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        Cubic(double tol, const GSParams& gsparams);
        ~Cubic() {}

        double xrange() const { return _range; }
        int ixrange() const { return 4; }
        double urange() const { return _uMax; }
        double getTolerance() const { return _tolerance; }

        double xval(double x) const;
        double uval(double u) const;

        // Override numerical calculation with known analytic integral
        double getPositiveFlux() const { return 13./12.; }
        double getNegativeFlux() const { return 1./12.; }

        std::string makeStr() const;

    private:
        // x range, reduced slightly from n=2 so we're not using zero-valued endpoints.
        double _range;

        double _tolerance;
        shared_ptr<TableBuilder> _tab; // Tabulated Fourier transform
        double _uMax;  // Truncation point for Fourier transform

        // Calculate the FT from a direct integration.
        double uCalc(double u) const;

        // Store the tables in a map, so repeat constructions are quick.
        static std::map<double,shared_ptr<TableBuilder> > _cache_tab;
        static std::map<double,double> _cache_umax;
    };

    /**
     * @brief Piecewise-quintic polynomial interpolant, ideal for Fourier-space interpolation
     *
     * See paper by Bernstein & Gruen, http://arxiv.org/abs/1401.2636.
     */

    class Quintic : public Interpolant
    {
    public:
        /**
         * @brief Constructor
         * @param[in] tol      Sets accuracy and extent of Fourier transform.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of
         *                     operations, if different from the default.
         */
        Quintic(double tol, const GSParams& gsparams);
        ~Quintic() {}

        double xrange() const { return _range; }
        int ixrange() const { return 6; }
        double urange() const { return _uMax; }
        double getTolerance() const { return _tolerance; }

        double xval(double x) const;
        double uval(double u) const;

        // Override numerical calculation with known analytic integral
        // Not as simple as the Cubic one, but still a straightforward integral for Maple.
        // For the curious, the + flux is (13018561 / 11595672) + (17267 / 14494590) * sqrt(31).
        double getPositiveFlux() const { return 1.1293413499280066555; }
        double getNegativeFlux() const { return 0.1293413499280066555; }

        std::string makeStr() const;

    protected:
        // Override default sampler configuration because Quintic filter has sign change in
        // outer interval
        void checkSampler() const;

    private:
        double _range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        double _tolerance;
        shared_ptr<TableBuilder> _tab; // Tabulated Fourier transform
        double _uMax;  // Truncation point for Fourier transform

        // Calculate the FT from a direct integration.
        double uCalc(double u) const;

        // Store the tables in a map, so repeat constructions are quick.
        static std::map<double,shared_ptr<TableBuilder> > _cache_tab;
        static std::map<double,double> _cache_umax;
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
     * this constant.  Setting conserve_dc in the constructor tweaks the function so that it
     * approximately conserves the value of constant (DC) input data (accurate to better than
     * 1.e-5 when used in two dimensions).
     */
    class Lanczos : public Interpolant
    {
    public:
        /**
         * @brief Constructor
         *
         * @param[in] n              Filter order; must be given on input and cannot be changed.
         * @param[in] conserve_dc    Set true to adjust filter to be more nearly correct for
         *                           constant inputs.
         * @param[in] tol            Sets accuracy and extent of Fourier transform.
         * @param[in] gsparams       GSParams object storing constants that control the accuracy of
         *                           operations, if different from the default.
         */
        Lanczos(int n, bool conserve_dc, double tol, const GSParams& gsparams);
        ~Lanczos() {}

        double xrange() const { return _range; }
        int ixrange() const { return 2*_n; }
        double urange() const { return _uMax; }
        double getTolerance() const { return _tolerance; }
        int getN() const { return _n; }
        bool conservesDC() const { return _conserve_dc; }

        double xval(double x) const;
        double uval(double u) const;

        std::string makeStr() const;

    private:
        int _n; // Store the filter order, n
        double _nd; // Store n as a double, since that's often how it is used.
        double _range; // Reduce range slightly from n so we're not using zero-valued endpoints.
        bool _conserve_dc; // Set to insure conservation of constant (sky) flux
        double _tolerance;  // u-space accuracy parameter
        double _uMax;  // truncation point for Fourier transform
        std::vector<double> _K; // coefficients for flux correction in xval
        std::vector<double> _C; // coefficients for flux correction in uval
        shared_ptr<TableBuilder> _xtab; // Table for x values
        shared_ptr<TableBuilder> _utab; // Table for Fourier transform

        double xCalc(double x) const;
        double uCalc(double u) const;
        double uCalcRaw(double u) const; // uCalc without any flux conservation.

        // Store the tables in a map, so repeat constructions are quick.
        typedef std::pair<int,std::pair<bool,double> > KeyType;
        static std::map<KeyType,shared_ptr<TableBuilder> > _cache_xtab;
        static std::map<KeyType,shared_ptr<TableBuilder> > _cache_utab;
        static std::map<KeyType,double> _cache_umax;
    };

}

#endif
