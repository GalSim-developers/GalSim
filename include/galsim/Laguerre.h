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

// Manipulation of Laguerr-decomposition-vector representation of images.

#ifndef LAGUERRE_H
#define LAGUERRE_H

#include <string>
#include <iostream>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include "TMV.h"

#include "Std.h"
#include "CppShear.h"
#include "Angle.h"

namespace galsim {

    // LVector will store a coefficient array as a real vector of the real degrees of freedom
    // for a vector that is Hermitian. Indexing by integer will retrieve these values.
    //  Indexing by a PQIndex or by an integer pair (p,q) will return the complex-valued
    // vector member, with proper conjugations applied both on input and output of elements.

    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    // First define an index taking p & q of the eigenfunction, and
    // derived vector and matrix classes which can be indexed this way.

    class PQIndex {
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
        PQIndex& setPQ(const int p_=0, const int q_=0) 
        { 
            p=p_;
            q=q_;
            return *this; 
        }

        int N() const { return p+q; }
        int m() const { return p-q; }
        PQIndex& setNm(const int N, const int m) 
        {
            assert(std::abs(m)<=N && (N-m)%2==0);
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
    class LVectorReference 
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
        LVectorReference(tmv::Vector<double>& v, PQIndex pq) :
            _re(&v[pq.rIndex()]), _isign(pq.iSign()) {}
        double *_re;
        int _isign; // 0 if this is a real element, -1 if needs conjugation, else +1

        friend class LVector;
    };

    // A custom deleter to allow us to return views to the LVector as numpy arrays
    // which will keep track of the Vector allocation.  When the last LVector _or_
    // external view of _owner goes out of scope, then the tmv::Vector is destroyed.
    class LVectorDeleter
    {
    public:
        LVectorDeleter(boost::shared_ptr<tmv::Vector<double> > v) : _v(v) {}

        void operator()(double * p) const {} // the _v shared_ptr will delete for us!

        boost::shared_ptr<tmv::Vector<double> > _v;
    };

    class LVector 
    {
    public:
        // Construct/destruct:
        LVector(int order=0) : _order(order) 
        { 
            allocateMem(); 
            _v->setZero();
        }

        LVector(int order, const tmv::GenVector<double>& v) :
            _order(order)
        {
            allocateMem();
            *_v = v;
            assert(v.size() == PQIndex::size(order)); 
        }

        LVector(int order, boost::shared_ptr<tmv::Vector<double> > v) :
            _order(order), _v(v),
            _owner(_v->ptr(), LVectorDeleter(_v))
        { assert(v->size() == PQIndex::size(order)); }

        LVector(const LVector& rhs) : _order(rhs._order), _v(rhs._v), _owner(rhs._owner) {}

        LVector& operator=(const LVector& rhs) 
        {
            if (_v.get()==rhs._v.get()) return *this;
            _order=rhs._order;
            _v = rhs._v;
            _owner = rhs._owner;
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
        { if (!_v.unique()) { _v.reset(new tmv::Vector<double>(*_v)); } }

        void clear() { take_ownership(); _v->setZero(); }

        int getOrder() const { return _order; }

        // Returns number of real DOF = number of complex coeffs
        int size() const { return _v->size(); }

        // Access the real-representation vector directly.
        const tmv::Vector<double>& rVector() const { return *_v; }
        tmv::Vector<double>& rVector() { take_ownership(); return *_v; }

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
        double dot(const LVector& rhs) const { return (*_v)*(*rhs._v); }

        // write and ??? read
        void write(std::ostream& os, int maxorder=-1) const;
        friend std::ostream& operator<<(std::ostream& os, const LVector& lv);

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
        static boost::shared_ptr<tmv::Matrix<double> > basis(
            const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
            int order, double sigma=1.);

        // Create design matrix, including factors of 1/sigma stored in invsig
        static boost::shared_ptr<tmv::Matrix<double> > design(
            const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
            const tmv::ConstVectorView<double>& invsig, int order, double sigma=1.);

        // ...or provide your own matrix
        static void design(
            const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
            const tmv::ConstVectorView<double>& invsig,
            tmv::MatrixView<double> psi, int order, double sigma=1.);

        static void basis(
            const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
            tmv::MatrixView<double> psi, int order, double sigma=1.);

        static boost::shared_ptr<tmv::Matrix<std::complex<double> > > kBasis(
            const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
            int order, double sigma);
        static void kBasis(
            const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
            tmv::MatrixView<std::complex<double> > psi_k, int order, double sigma);

        // ?? Add routine to decompose a data vector into b's
        // ?? Add routines to evaluate summed basis at a set of x/k points
        // These can be written to use less memory than building the full basis matrix
        //  so that they will run largely in cache.

        // Transformations of coefficient LVectors representing real objects:
        // Rotate represented object by theta:
        // (note this is faster than using RotationLTransform)
        void rotate(const Angle& theta);

        // Get the total flux or flux within an aperture of size R*sigma
        // Use all monopole terms unless maximum is specified by maxP.
        double flux(int maxP=-1) const;
        double apertureFlux(double R, int maxP=-1) const;

        // Return reference to a matrix that generates ???realPsi transformations
        // under infinitesimal point transforms (translate, dilate, shear).
        // Returned matrix is at least as large as needed to go order x (order+2)
        // The choices for generators:
        enum GType { iX = 0, iY, iMu, iE1, iE2, iRot, nGen };
        static const tmv::ConstMatrixView<double> Generator(
            GType iparam, int orderOut, int orderIn);

        boost::shared_ptr<double> getOwner() const { return _owner; }

    private:
        // real vs fourier is set by the type of psi.
        // For real, T = double
        // For fourier, T = std::complex<double>
        template <typename T>
        static void mBasis(
            const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
            const tmv::ConstVectorView<double>* invsig,
            tmv::MatrixView<T> psi, int order, double sigma=1.);

        void allocateMem()
        {
            int s = PQIndex::size(_order);
            _v.reset(new tmv::Vector<double>(s));
            _owner.reset(_v->ptr(), LVectorDeleter(_v));
        }

        int _order;
        boost::shared_ptr<tmv::Vector<double> > _v;
        boost::shared_ptr<double> _owner;
    };

    std::ostream& operator<<(std::ostream& os, const LVector& lv);
    std::istream& operator>>(std::istream& is, LVector& lv);

    // To allow iteration over all the generators:
    inline LVector::GType& operator++(LVector::GType& g) { return g=LVector::GType(g+1); }

    // This function finds the innermost radius at which the integrated flux
    // of the LVector's shape crosses the specified threshold, using the first
    // maxP monopole terms (or all, if maxP omitted)
    extern double fluxRadius(const LVector& lv, double threshold, int maxP=-1);


    //--------------------------------------------------------------
    //
    // Next class is a transformation matrix for Laguerre vector.  Internal 
    // storage is as a matrix over the real degrees of freedom.
    // Interface gives you the (complex) matrix elements of  pqIndex pairs.

    // Again this is a HANDLE, so it can be passed into
    // subroutines without referencing.  Copy/assignment create a new link; 
    // for fresh copy, use copy() method.
    class LTransform 
    {
    public:
        LTransform(int orderOut=0, int orderIn=0) : 
            _orderIn(orderIn), _orderOut(orderOut), 
            _m(new tmv::Matrix<double>(PQIndex::size(orderOut),PQIndex::size(orderIn),0.))
        {}

        // Build an LTransform from a tmv::Matrix<double> for the real degrees of freedom.
        // Matrix must have correct dimensions.
        LTransform(int orderOut, int orderIn, const tmv::GenMatrix<double>& m) :
            _orderIn(orderIn), _orderOut(orderOut),
            _m(new tmv::Matrix<double>(m))
        {
            assert(m.ncols() == PQIndex::size(orderIn));
            assert(m.nrows() == PQIndex::size(orderOut));
        }

        LTransform(int orderOut, int orderIn, boost::shared_ptr<tmv::Matrix<double> > m) :
            _orderIn(orderIn), _orderOut(orderOut), _m(m)
        {
            assert(m->ncols() == PQIndex::size(orderIn));
            assert(m->nrows() == PQIndex::size(orderOut));
        }

        LTransform(const LTransform& rhs) : 
            _orderIn(rhs._orderIn), _orderOut(rhs._orderOut), _m(rhs._m) {}

        LTransform& operator=(const LTransform& rhs) 
        {
            if (_m.get()==rhs._m.get()) return *this;
            _orderIn=rhs._orderIn; _orderOut=rhs._orderOut; _m = rhs._m; 
            return *this;
        }

        ~LTransform() {}

        LTransform copy() const 
        {
            LTransform fresh(_orderOut, _orderIn); 
            *(fresh._m) = *_m;
            return fresh;
        }

        int getOrderIn() const { return _orderIn; }
        int getOrderOut() const { return _orderOut; }
        int sizeIn() const { return _m->ncols(); }
        int sizeOut() const { return _m->nrows(); }

        void resize(int orderOut, int orderIn) 
        {
            if (_orderIn != orderIn || _orderOut != orderOut) {
                _orderIn = orderIn;
                _orderOut = orderOut;
                _m.reset(new tmv::Matrix<double>(
                        PQIndex::size(orderOut), PQIndex::size(orderIn), 0.));
            } else {
                take_ownership();
            }
        }

        // As above, we use take_ownership() to implement Copy on Write semantics.
        void take_ownership() 
        { if (!_m.unique()) { _m.reset(new tmv::Matrix<double>(*_m)); } }

        void clear() { take_ownership(); _m->setZero(); }
        void identity() { take_ownership(); _m->setToIdentity(); }

        // Access the real-representation vector directly.
        tmv::Matrix<double>& rMatrix() { take_ownership(); return *_m; }
        const tmv::Matrix<double>& rMatrix() const { return *_m; }

        // Element read
        std::complex<double> operator()(PQIndex pq1, PQIndex pq2) const;
        std::complex<double> operator()(int p1, int q1, int p2, int q2) const 
        { return operator()(PQIndex(p1,q1),PQIndex(p2,q2)); }

        // Element write.  Note that it is necessary to give two complex
        // simultaneously to allow writing the real version of the matrix:
        void set(
            PQIndex pq1, PQIndex pq2,
            std::complex<double> Cpq1pq2, std::complex<double> Cqp1pq2);

        // Operate on other Laguerre vectors/matrices
        LVector operator*(const LVector rhs) const;
        LTransform operator*(const LTransform rhs) const;
        LTransform& operator*=(const LTransform rhs);

    private:
        int _orderIn;
        int _orderOut;
        boost::shared_ptr<tmv::Matrix<double> > _m;
    };

    // Here are the primary types of transformations:
    // For the point transforms, set coordShift=false if we want
    // to transform the FLUX on a fixed coordinate grid.  Set true
    // if want to describe the same flux on a transformed COORD system.

    // Shear:
    LTransform MakeLTransform(
        CppShear eta, int orderOut, int orderIn, bool coordShift=false);

    // Dilation:
    LTransform MakeLTransform(
        double mu, int orderOut, int orderIn, bool coordShift=false);

    // Translation:
    LTransform MakeLTransform(
        Position<double> x0, int orderOut, int orderIn, bool coordShift=false);

    // Rotation:
    LTransform RotationLTransform(
        double theta, int orderOut, int orderIn, bool coordShift=false);

    // Convolution with PSF:
    LTransform MakeLTransform(
        const LVector psf, const double D,
        const int orderOut, const int orderIn, const int orderStar);

}

#endif
