/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

// Manipulation of Laguerr-decomposition-vector representation of images.

#ifndef GalSim_Laguerre_H
#define GalSim_Laguerre_H

#include <string>
#include <iostream>
#include <sstream>
#ifdef USE_TMV
#include "TMV.h"
typedef tmv::Vector<double> VectorXd;
typedef tmv::Matrix<double> MatrixXd;
typedef tmv::Vector<std::complex<double> > VectorXcd;
typedef tmv::Matrix<std::complex<double> > MatrixXcd;
#else
#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
#include "Eigen/Dense"
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;
#endif

#include "Std.h"

namespace galsim {

    // LVector will store a coefficient array as a real vector of the real degrees of freedom
    // for a vector that is Hermitian. Indexing by integer will retrieve these values.
    //  Indexing by a PQIndex or by an integer pair (p,q) will return the complex-valued
    // vector member, with proper conjugations applied both on input and output of elements.

    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    // First define an index taking p & q of the eigenfunction, and
    // derived vector and matrix classes which can be indexed this way.

    class PUBLIC_API PQIndex {
        // Index vectors/matrices of Laguerre coefficients using the
        // p & q quantum numbers (or others).
        // When we want to store the coefficients as a complex array,
        //  the storage order will be increasing in N=p+q, with decreasing
        //  m=p-q within a given N:
        //  pq:      00;  10, 01; 20, 11, 02; 30, 21, 12, 03; ...
        //  cIndex:   0    1   2   3   4  5   6   7    8   9  ...
        //
        // Also will need to get the index into a Vector that stores
        // the real degrees of freedom of the (Hermitian) Laguerre vector
        // for a real-valued image.  The storage order here is:
        //  pq:        Re(00); Re(10), Im(10); Re(20), Im(20), Re(11); ...
        //  rIndex:     0        1       2       3       4       5     ...
        //
        // Methods include increments for p, q, N, and m.  This object can
        // be used as an index into any of our Laguerre arrays.
    public:
        PQIndex() : p(0), q(0) {}
        PQIndex(const int p_, const int q_) { setPQ(p_,q_); }

        bool pqValid() const { return p>=0 && q>=0; }

        int getP() const { return p; }
        int getQ() const { return q; }
        PQIndex& setPQ(const int p_, const int q_)
        {
            p=p_;
            q=q_;
            return *this;
        }

        int N() const { return p+q; }
        int m() const { return p-q; }
        PQIndex& setNm(const int N, const int m)
        {
            xassert(std::abs(m)<=N && (N-m)%2==0);
            p=(N+m)/2;
            q=(N-m)/2;
            return *this;
        }

        int cIndex() const { return makeCIndex(p,q); }
        int rIndex() const { return makeRIndex(p,q); }

        bool needsConjugation() const { return p<q; }

        bool isReal() const { return p==q; }

        int iSign() const { return p < q ? -1 : p == q ? 0 : 1; }

        // Operations that update the indices:
        void incP() { p++; }
        void incQ() { q++; }
        void incN() { p++; q++; }  //raise N by 2, same m
        void decN() { p--; q--; }  //lower N by 2, same m (could be invalid)
        void incm() { p++; q--; } // raise m by 2, same N (could be invalid)
        void decm() { p--; q++; } // lower m by 2, same N (could be invalid)

        // get next one in complex sequence
        PQIndex& operator++()
        {
            if (p==0) { p=q+1; q=0; }
            else { --p; ++q; }
            return *this;
        }

        // get next pq index that has m>=0
        PQIndex& nextDistinct()
        {
            if (p-q<2) { p=p+q+1; q=0; }
            else { --p; ++q; }
            return *this;
        }


        // Functions to report incremented/decremented indices without
        // updating this index:
        PQIndex swapPQ() const { return PQIndex(q,p); }
        PQIndex pp1() const { return PQIndex(p+1,q); }
        PQIndex qp1() const { return PQIndex(p,q+1); }
        PQIndex pm1() const { return PQIndex(p-1,q); }
        PQIndex qm1() const { return PQIndex(p,q-1); }

        bool operator==(const PQIndex rhs) const
        { return p==rhs.p && q==rhs.q; }

        // Other useful things:
        static int size(int order)
        {
            // Size of a CVector to this order N, same as number of real DOF:
            assert(order>=0);
            return (order+1)*(order+2)/2;
        }

        // write and ??? read
        void write(std::ostream& os) const;

        // Returns true if index has advanced past order:
        bool pastOrder(const int order) const { return p+q>order; }

    private:
        int p;
        int q;
        // Index of this element into a complex-valued array
        static int makeCIndex(const int p_, const int q_) {
            return (p_+q_)*(p_+q_+1)/2+q_;
        }
        // Index of real part of this element in real-valued storage order.
        // Gauranteed that imaginary part, if it exists, has 1-higher index.
        static int makeRIndex(const int p_, const int q_) {
            return (p_+q_)*(p_+q_+1)/2 + 2*std::min(p_,q_);
        }

    };

    inline std::ostream& operator<<(std::ostream& os, const PQIndex& pq)
    { pq.write(os); return os; }

    //--------------------------------------------------------------
    // Next object is a vector of Laguerre coefficients.  Note this is
    // a HANDLE to the coefficient vector, so it can be passed into
    // subroutines without referencing.  Copy/assignment create a new link;
    // for fresh copy, use copy() method.
    //
    // LVectors are assumed to be Hermitian complex (b_qp=b_pq*), and the
    // internal storage currently enforces this and actually stores the
    // data as a Vector of the real degrees of freedom.
    // So when you change b_qp, you are also changing b_pq.

    // Reference to a pq-indexed complex element of an LVector:
    class PUBLIC_API LVectorReference
    {
    public:
        operator std::complex<double>() const
        {
            if (_isign==0) return std::complex<double>(*_re,0.);
            else return std::complex<double>(*_re, *(_re+1)*_isign);
        }
        LVectorReference& operator=(std::complex<double> z)
        {
            *_re = z.real();
            if (_isign!=0) *(_re+1)=z.imag()*_isign;
            // Choosing *not* to check for zero imaginary part here
            return *this;
        }
        LVectorReference& operator=(double d)
        {
            *_re = d;
            if (_isign!=0) *(_re+1)=0.;
            return *this;
        }
        // ??? +=, -=, etc.

    private:
        LVectorReference(VectorXd& v, PQIndex pq) :
            _re(&v[pq.rIndex()]), _isign(pq.iSign()) {}
        double *_re;
        int _isign; // 0 if this is a real element, -1 if needs conjugation, else +1

        friend class LVector;
    };

    class PUBLIC_API LVector
    {
    public:
        // Construct/destruct:
        LVector(int order) : _order(order)
        {
            allocateMem();
            _v->setZero();
        }

        LVector(int order, const VectorXd& v) :
            _order(order)
        {
            allocateMem();
            *_v = v;
            xassert(v.size() == PQIndex::size(order));
        }

        LVector(const LVector& rhs) : _order(rhs._order), _v(rhs._v) {}

        LVector& operator=(const LVector& rhs)
        {
            if (_v.get()==rhs._v.get()) return *this;
            _order=rhs._order;
            _v = rhs._v;
            return *this;
        }

        ~LVector() {}

        LVector copy() const
        { return LVector(_order,*_v); }

        void resize(int order)
        {
            if (_order != order) {
                _order = order;
                allocateMem();
                _v->setZero();
            } else {
                // The caller may be relying on resize to get a unique vector, so if
                // we don't make a new one, at least take ownership of the current one.
                take_ownership();
            }
        }

        // We keep the Vector in a shared_ptr to make it cheap to return an LVector by value.
        // However, this is confusing if you edit one LVector that had been originally made
        // from another LVector, since then both objects' internal Vectors would be changed.
        // So for all non-const methods, we first take ownership of the internal vector
        // by making a new copy of the vector first.  If it is already the sole owner,
        // then nothing is done.  (FYI: The term for this is "Copy on Write" semantics.)
        void take_ownership()
        { if (!_v.unique()) { _v.reset(new VectorXd(*_v)); } }

        void clear() { take_ownership(); _v->setZero(); }

        int getOrder() const { return _order; }

        std::string repr() const;

        // Returns number of real DOF = number of complex coeffs
        int size() const { return _v->size(); }

        // Access the real-representation vector directly.
        const VectorXd& rVector() const { return *_v; }
        VectorXd& rVector() { take_ownership(); return *_v; }

        // op[] with int returns real
        double operator[](int i) const { return (*_v)[i]; }
        double& operator[](int i) { take_ownership(); return (*_v)[i]; }

        // Access as complex elements
        // op[] with PQIndex returns complex
        std::complex<double> operator[](PQIndex pq) const
        {
            int isign=pq.iSign();
            if (isign==0) return std::complex<double>( (*_v)[pq.rIndex()], 0.);
            else return std::complex<double>( (*_v)[pq.rIndex()], isign*(*_v)[pq.rIndex()+1]);
        }
        LVectorReference operator[](PQIndex pq)
        { take_ownership(); return LVectorReference(*_v, pq); }

        // op() with p,q values returns complex
        std::complex<double> operator()(int p, int q) const
        { return (*this)[PQIndex(p,q)]; }
        LVectorReference operator()(int p, int q)
        { take_ownership(); return (*this)[PQIndex(p,q)]; }

        // scalar arithmetic:
        LVector& operator*=(double s)
        { take_ownership(); *_v *= s; return *this; }
        LVector& operator/=(double s)
        { take_ownership(); *_v /= s; return *this; }

        LVector operator*(double s) const
        {
            LVector fresh = *this;
            fresh *= s;
            return fresh;
        }

        LVector operator/(double s) const
        {
            LVector fresh = *this;
            fresh /= s;
            return fresh;
        }

        LVector& operator+=(const LVector& rhs)
        {
            take_ownership();
            assert(_order==rhs._order);
            *_v += *(rhs._v);
            return *this;
        }

        LVector& operator-=(const LVector& rhs)
        {
            take_ownership();
            assert(_order==rhs._order);
            *_v -= *(rhs._v);
            return *this;
        }

        LVector operator+(const LVector& rhs) const
        {
            LVector fresh = *this;
            fresh += rhs;
            return fresh;
        }

        LVector operator-(const LVector& rhs) const
        {
            LVector fresh = *this;
            fresh -= rhs;
            return fresh;
        }

        // Inner product of the real values.
        double dot(const LVector& rhs) const {
#ifdef USE_TMV
            return (*_v)*(*rhs._v);
#else
            return _v->dot(*rhs._v);
#endif
        }

        // write to an ostream
        void write(std::ostream& os, int maxorder=-1) const;
        friend std::ostream& operator<<(std::ostream& os, const LVector& lv);

        // read from an istream
        void read(std::istream& is);
        friend std::istream& operator>>(std::istream& is, LVector& lv);

        // Now Gauss-Laguerre functions:

        // Note that what will be produced here are the basis functions that
        // are associated with each *real* degree of freedom, such that
        //  I(x,y) = b.dot(basis(x,y))
        //   So the p!=q values of the basis LVector created are actually
        // 2*conjugate of psi(x,y).

        // Inputs assume that the x and y values have already been transformed
        // to a unit-circle basis, but can optionally rescale output
        // by 1/sigma^2 to obtain proper normalization:
        void fillBasis(double x, double y, double sigma=1.);

        // Create a matrix containing basis values at vector of input points.
        // Output matrix has m(i,j) = jth basis function at ith point
        static shared_ptr<MatrixXd> basis(
            const VectorXd& x, const VectorXd& y,
            int order, double sigma=1.);

        // Create design matrix, including factors of 1/sigma stored in invsig
        static shared_ptr<MatrixXd> design(
            const VectorXd& x, const VectorXd& y,
            const VectorXd& invsig, int order, double sigma=1.);

        // ...or provide your own matrix
        static void design(
            const VectorXd& x, const VectorXd& y,
            const VectorXd& invsig,
            MatrixXd& psi, int order, double sigma=1.);

        static void basis(
            const VectorXd& x, const VectorXd& y,
            MatrixXd& psi, int order, double sigma=1.);

        static shared_ptr<MatrixXcd> kBasis(
            const VectorXd& kx, const VectorXd& ky,
            int order, double sigma);
        static void kBasis(
            const VectorXd& kx, const VectorXd& ky,
            MatrixXcd& psi_k, int order, double sigma);

        // ?? Add routine to decompose a data vector into b's
        // ?? Add routines to evaluate summed basis at a set of x/k points
        // These can be written to use less memory than building the full basis matrix
        //  so that they will run largely in cache.

        // Transformations of coefficient LVectors representing real objects:
        // Rotate represented object by theta:
        void rotate(double theta);

        // Get the total flux or flux within an aperture of size R*sigma
        // Use all monopole terms unless maximum is specified by maxP.
        double flux(int maxP=-1) const;
        double apertureFlux(double R, int maxP=-1) const;

    private:

        void allocateMem()
        {
            int s = PQIndex::size(_order);
            _v.reset(new VectorXd(s));
        }

        int _order;
        shared_ptr<VectorXd> _v;
    };

    PUBLIC_API std::ostream& operator<<(std::ostream& os, const LVector& lv);
    PUBLIC_API std::istream& operator>>(std::istream& is, LVector& lv);

    // This function finds the innermost radius at which the integrated flux
    // of the LVector's shape crosses the specified threshold, using the first
    // maxP monopole terms (or all, if maxP omitted)
    PUBLIC_API double fluxRadius(const LVector& lv, double threshold, int maxP=-1);

}

#endif
