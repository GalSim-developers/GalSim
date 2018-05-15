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
#ifndef GalSim_FFT_H
#define GalSim_FFT_H

/**
 * @file FFT.h
 *
 * @brief Objects that make use of 2d FFT's in VERSION 3 of the FFTW package.
 *
 * Notes:
 *
 * * Requires that the FFTW is set up to do double-precision.
 *
 * * All tables have even dimensions (enforced on construction).
 *
 * * The complex arrays (KTables) must be Hermitian, so transforms are real.
 *
 * * All arrays are 0-indexed.
 *
 * * FFTW arrays are stored in ROW-MAJOR order, meaning that in matrix notation, the most
 * rapidly varying index is the last one. However in an "image" view of the array, we would
 * label this as the x value that is increasing along rows.  When doing real-to-complex
 * transforms, the complex output contains (N/2+1) elements along this latter index, 0<=j<=N/2,
 * and the j<0 elements must be obtained via conjugation. Imaginary parts of 0 and N/2 are zero.
 *
 * * This interface will assume an "image" based convention in which all access to real
 * or complex elements is in (ix, iy) format, and ix will be the rapidly varying index, with
 * the k-space arrays being half-sized in the x direction.
 *
 * * **SO**: when filling arrays, make ix your inner loop.  And provide only kx>=0 to fill KTable.
 *
 * * xTable arrays have indices -N/2 <= ix,iy < N/2.  To store in FFTW arrays, which assume
 * 0<=j < N, we add N/2 to indices before accessing FFTW arrays. This means that k
 * values need to be multiplied by -1^(ix+iy) before/after transforms.
 *
 * * kTable arrays can be accessed by -N/2 <= jx, jy <= N/2.  FFTW puts DC at [0,0] element,
 * so the code in this class changes negative input indices to wrap them properly, also
 * considering that jx<0 must be conjugated.
 *
 * * "forward" transform, x->k, has -1 in exponent.
 */

#include <stdexcept>
#include <deque>
#include <complex>

#include <fftw3.h>

#include "Std.h"
#include "Interpolant.h"

// Define this to get extra debugging checks in the FFT routines.
// Since these routines are not available to the end user, once code is working
// it should be ok to turn these off for some modest speed up.
//#define FFT_DEBUG

namespace galsim {

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /// @brief Basic exception class thrown by XTable and KTable
    class FFTError : public std::runtime_error
    {
    public:
        FFTError(const std::string& m) : std::runtime_error("FFT error: " + m) {}
    };

    /// @brief Exception class for XTable and KTable access ouside the allowed range
    class FFTOutofRange : public FFTError
    {
    public:
        FFTOutofRange(const std::string& m="value out of range") : FFTError(m) {}
    };

    /// @brief Exception class thrown when fftw3 returns an invalid plan.
    class FFTInvalid : public FFTError
    {
    public:
        FFTInvalid(const std::string& m="invalid plan or data") : FFTError(m) {}
    };

    // Quick helper struct to tell if T is real or complex
    template <typename T>
    struct FFTW_Traits
    {
        enum { isreal = true };
        typedef T fftw_type;
        size_t size_2d(size_t n) { return n*n; }
    };
    template <typename T>
    struct FFTW_Traits<std::complex<T> >
    {
        enum { isreal = false };
        typedef fftw_complex fftw_type;
        size_t size_2d(size_t n) { return n*(n/2+1); }
    };

    // Handle the FFTW3 memory allocation, which assured 16-bit alignment for SSE usage.
    // FFTW3 now states that C++ complex<double> will be bit-compatible with
    // the fftw_complex type.  So all interfaces will be through std::complex<double>
    // And the fftw real type is now just double.
    template <typename T>
    class FFTW_Array
    {
        typedef typename FFTW_Traits<T>::fftw_type fftw_type;
    public:
        FFTW_Array() : _n(0), _p(0) {}

        FFTW_Array(size_t n) : _n(0), _p(0)
        { resize(n); }

        FFTW_Array(size_t n, T val) : _n(0), _p(0)
        {
            resize(n);
            fill(val);
        }

        FFTW_Array(const FFTW_Array<T>& rhs) : _n(0), _p(0)
        {
            resize(rhs._n);
            for (size_t i=0; i<_n; ++i) _p[i] = rhs._p[i];
        }

        FFTW_Array<T>& operator=(const FFTW_Array<T>& rhs)
        {
            if (this != &rhs) {
                resize(rhs._n);
                for (size_t i=0; i<_n; ++i) _p[i] = rhs._p[i];
            }
            return *this;
        }

        ~FFTW_Array();

        void resize(size_t n);

        void fill(T val)
        {
            for (size_t i=0; i<_n; ++i) _p[i] = val;
        }

        size_t size() { return _n; }

        T* get() { return _p; }
        const T* get() const { return _p; }

        fftw_type* get_fftw()
        { return reinterpret_cast<fftw_type*>(_p); }
        const fftw_type* get_fftw() const
        { return reinterpret_cast<const fftw_type*>(_p); }

        T& operator[](size_t i) { return _p[i]; }
        const T& operator[](size_t i) const { return _p[i]; }

    private:
        size_t _n;
        T* _p;
    };

    //! @endcond

    class XTable;

    /**
     * @brief KTable is a class holding the k-space representation of a real function.
     *
     * It will be based on an assumed Hermitian 2d square array.
     * The table will be forced to be of even size.
     */
    class KTable
    {
        typedef std::complex<double> function1(double kx, double ky);
        typedef std::complex<double> function2(double kx, double ky, std::complex<double> val);
    public:
        /// dummy constructor
        KTable() : _N(0), _No2(0), _Nd(0.), _halfNd(0.), _invNd(0.), _dk(0.), _invdk(0.) {}

        /// Construct with size and spacing.  Default is to zero out the table.
        KTable(int N, double dk, std::complex<double> value=0.);

        KTable(const KTable& rhs) :
            _array(rhs._array),
            _N(rhs._N), _No2(rhs._No2), _Nd(rhs._Nd), _halfNd(rhs._halfNd), _invNd(rhs._invNd),
            _dk(rhs._dk), _invdk(rhs._invdk) {}

        KTable& operator=(const KTable& rhs)
        {
            if (this != &rhs) {
                clearCache();
                _array = rhs._array;
                _N=rhs._N;
                _No2=rhs._No2;
                _Nd=rhs._Nd;
                _halfNd=rhs._halfNd;
                _invNd=rhs._invNd;
                _dk=rhs._dk;
                _invdk=rhs._invdk;
            }
            return *this;
        }

        ~KTable() {}

        /**
         * @brief Fourier transform from (complex) k to x.
         *
         * This version returns a pointer to the result in real space.
         */
        shared_ptr<XTable> transform() const;

        /**
         * @brief Fourier transform from (complex) k to x.
         *
         * This version writes the result to the provided XTable argument.
         */
        void transform(XTable& xt) const;

        /// Have FFTW develop "wisdom" on doing this kind of transform
        void fftwMeasure() const;

        /// This one does a "dumb" Fourier transform for a single (x,y) point:
        double xval(double x, double y) const;

        /// Return value at grid point ix,iy (k = (ix*dk, iy*dk))
        std::complex<double> kval(int ix, int iy) const;

        /// Same as kval, but assumes ix,iy are already known to be valid arguments of index2.
        std::complex<double> kval2(int ix, int iy) const
        { return _array[index2(ix,iy)]; }

        /// interpolate to k=(kx, ky) - WILL wrap k values to fill interpolant kernel
        std::complex<double> interpolate(double kx, double ky, const Interpolant2d& interp) const;

        /// Set the value of a grid point ix,iy (k = (ix*dk, iy*dk)) to a given value.
        void kSet(int ix, int iy, std::complex<double> value);

        /// Same as kSet, but assumes ix,iy are already known to be valid arguments of index2.
        void kSet2(int ix, int iy, std::complex<double> value)
        { _array[index2(ix,iy)]=value; }

        /// Set all values to zero
        void clear();

        /// Clear any cached values that had been set from previous passes.
        void clearCache() const
        {
            _cache.clear();
            _xwt.clear();
        }

        /// this += scalar*rhs
        void accumulate(const KTable& rhs, double scalar=1.);

        /// Multiply each element by the corresponding element in rhs.
        void operator*=(const KTable& rhs);

        /// Multiply each element by a scalar.
        void operator*=(double scalar);

        /// Produce a new KTable which wraps this one onto range +-Nout/2.  Nout will
        /// be raised to even value.  In other words, aliases the data.
        shared_ptr<KTable> wrap(int Nout) const;

        /// Get the size of the table.
        int getN() const { return _N; }
        /// Get the pixel spacing of the table
        double getDk() const { return _dk; }

        /// Translate to move origin at (x0,y0)
        void translate(double x0, double y0);

        /// Fill table from a function or function object:
        void fill(function1 func);
        template <class T> void fill( const T& f) ;

        /// New table is function of this one:
        shared_ptr<KTable> function(function2 func) const;

        /// Integrate a function over d^2k:
        std::complex<double> integrate(function2 func) const;

        /// Integrate KTable over d^2k (sum of all pixels * dk * dk)
        std::complex<double> integratePixels() const;

        /// Allow the ability to directly write to the array.
        std::complex<double>* getArray() { return _array.get(); }
        const std::complex<double>* getArray() const { return _array.get(); }

    private:

        FFTW_Array<std::complex<double> > _array;
        int _N; // Size in each dimension.
        int _No2; // N/2
        double _Nd; // double version of N to avoid repeated int->double conversions
        double _halfNd; // 0.5 * N
        double _invNd; // 1/N
        double _dk; // k-space increment
        double _invdk; // 1/dk

        size_t index(int ix, int iy) const  //Return index into data array.
        {
            // this is also responsible for bounds checking when FFT_DEBUG is turned on.
#ifdef FFT_DEBUG
            if (ix<-_No2 || ix>_No2 || iy<-_No2 || iy>_No2)
                FormatAndThrow<FFTOutofRange>() << "KTable index (" << ix << "," << iy
                    << ") out of range for N=" << _N;
#endif
            if (ix<0) {
                ix=-ix; iy=-iy; //need the conjugate in this case
            }
            if (iy<0) iy+=_N;
            return iy*(_No2+1)+ix;
        }

        // This skips all the adjustments to ix,iy, so both should be positive and
        // folded appropriately.
        size_t index2(int ix, int iy) const  //Return index into data array.
        { return iy*(_No2+1)+ix; }

#ifdef FFT_DEBUG
        void check_array() const
        { if (!_array.get()) throw FFTError("KTable operation on null array"); }
#else
        void check_array() const {}
#endif

        int wrapKValue(double k) const;  // wrap floor(k) to be within [-N/2,N/2-1]

        // Objects used to accelerate interpolation with separable interpolants:
        mutable std::deque<std::complex<double> > _cache;
        mutable std::vector<double> _xwt;
        mutable int _cacheStartY;
        mutable double _cacheX;
        mutable const InterpolantXY* _cacheInterp;

        friend class XTable;
    };

    /**
     * @brief XTable is a class holding a lookup table of a real 2-d function.
     *
     * N is forced to be even, and the origin is taken to be (N/2, N/2).
     */
    class XTable
    {
        typedef double function1(double x, double y);
        typedef double function2(double x, double y, double val);
    public:
        /// Construct with size and spacing.  Default is to zero out the table.
        XTable(int N, double dx, double value=0.);

        XTable(const XTable& rhs) :
            _array(rhs._array),
            _N(rhs._N), _No2(rhs._No2), _Nd(rhs._Nd), _halfNd(rhs._halfNd), _invNd(rhs._invNd),
            _dx(rhs._dx), _invdx(rhs._invdx) {}

        XTable& operator=(const XTable& rhs)
        {
            if (this != &rhs) {
                clearCache();
                _array = rhs._array;
                _N=rhs._N;
                _No2=rhs._No2;
                _Nd=rhs._Nd;
                _halfNd=rhs._halfNd;
                _invNd=rhs._invNd;
                _dx=rhs._dx;
                _invdx=rhs._invdx;
            }
            return *this;
        }

        ~XTable() {}

        /**
         * @brief Fourier transform from x to (complex) k.
         *
         * This version returns a pointer to the result in Fourier space.
         */
        shared_ptr<KTable> transform() const;

        /**
         * @brief Fourier transform from x to (complex) k.
         *
         * This version writes the result to the provided KTable argument.
         */
        void transform(KTable& kt) const;

        /// Have FFTW develop "wisdom" on doing this kind of transform
        void fftwMeasure() const;

        /// Do a "dumb" FT at a single frequency:
        std::complex<double> kval(double kx, double ky) const;

        /// Get value at grid point (x,y) = (ix*dx, iy*dx)
        double xval(int ix, int iy) const;

        /// interpolate to (x,y) - will NOT wrap the x data around +-N/2
        double interpolate(double x, double y, const Interpolant2d& interp) const;

        /// Set the value of a grid point ix,iy ((x,y) = (ix*dk, iy*dk)) to a given value.
        void xSet(int ix, int iy, double value);

        /// Set all values to zero
        void clear();

        /// Clear any cached values that had been set from previous passes.
        void clearCache() const
        {
            _cache.clear();
            _xwt.clear();
        }

        /// this += scalar*rhs
        void accumulate(const XTable& rhs, double scalar=1.);

        /// Multiply each element by a scalar.
        void operator*=(double scalar);

        /// Produce a new XTable which wraps this one onto range +-Nout/2.  Nout will
        /// be raised to even value.
        shared_ptr<XTable> wrap(int Nout) const;

        /// Get the size of the table.
        int getN() const { return _N; }
        /// Get the pixel spacing of the table
        double getDx() const { return _dx; }

        /// Fill table from a function:
        void fill(function1 func);

        /// Integrate a (real) function over d^2x; set flag for sum:
        double integrate(function2 func, bool sumonly=false) const;

        /// Integrate XTable over d^2x (sum of all pixels * dx * dx)
        double integratePixels() const;

        /// Allow the ability to directly write to the array.
        double* getArray() { return _array.get(); }
        const double* getArray() const { return _array.get(); }

    private:
        FFTW_Array<double> _array; //hold the values.
        int _N; // Size in each dimension.
        int _No2; // N/2
        double _Nd; // double version of N to avoid repeated int->double conversions
        double _halfNd; // 0.5 * N
        double _invNd; // 1/N
        double _dx; // k-space increment
        double _invdx; // 1/dx

        size_t index(int ix, int iy) const //Return index into data array.
        {
            // this is also responsible for bounds checking.
            // origin will be in center.
            ix += _No2;
            iy += _No2;
#ifdef FFT_DEBUG
            if (ix<0 || ix>=_N || iy<0 || iy>=_N)
                FormatAndThrow<FFTOutofRange>() << "XTable index (" << ix << "," << iy
                    << ") out of range for N=" << _N;
#endif
            return iy*_N+ix;
        }

#ifdef FFT_DEBUG
        void check_array() const
        { if (!_array.get()) throw FFTError("KTable operation on null array"); }
#else
        void check_array() const {}
#endif

        // Objects used to accelerate interpolation with separable interpolants:
        mutable std::deque<double> _cache;
        mutable std::vector<double> _xwt;
        mutable double _cacheX;
        mutable int _cacheStartY;
        mutable const InterpolantXY* _cacheInterp;

        friend class KTable;
    };

    /// Fill table from a function class:
    template <class T>
    void KTable::fill(const T& f)
    {
        clearCache(); // invalidate any stored interpolations
        std::complex<double>* zptr=_array.get();
        double kx, ky;
        for (int iy=0; iy< _No2; iy++) {
            ky = iy*_dk;
            for (int ix=0; ix< _No2+1 ; ix++) {
                kx = ix*_dk;
                *(zptr++) = f(kx,ky);
            }
        }
        // wrap to the negative ky's
        for (int iy=-_No2; iy< 0; iy++) {
            ky = iy*_dk;
            for (int ix=0; ix< _No2+1 ; ix++) {
                kx = ix*_dk;
                *(zptr++) = f(kx,ky);
            }
        }
        return;
    }

}

#endif
