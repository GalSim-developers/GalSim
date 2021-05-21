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

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#include "BinomFact.h"
#include "Laguerre.h"
#include "Solve.h"
#include "math/Angle.h"

#ifdef USE_TMV
#define MatrixXT tmv::Matrix<T>
#else
using Eigen::Dynamic;
#define MatrixXT Eigen::Matrix<T,Dynamic,Dynamic>
#endif

namespace galsim {

    std::string LVector::repr() const
    {
        std::ostringstream oss(" ");
        oss << "galsim._galsim.LVector("<<getOrder()<<", array([";

        // This is copied from the write() function, but then modified.
        // Should probably make a version of write() that could be called directly here
        // and also work for the os << lv usage.  OTOH, I'm planning to revamp all the
        // Shapelet code pretty significantly for #502 (and do more than that issue mentions),
        // so it's probably not worth worrying about at the moment.
        oss.precision(15);
        oss.setf(std::ios::scientific,std::ios::floatfield);
        oss << (*_v)[0];
        for (int n=1; n<=_order; n++) {
            for(PQIndex pq(n,0); !pq.needsConjugation(); pq.decm()) {
                if (pq.isReal()) {
                    oss << ", " << (*this)[pq].real() << std::endl;
                } else {
                    oss << ", " << (*this)[pq].real()
                        << ", " << (*this)[pq].imag() << std::endl;
                }
            }
        }
        oss <<"]))";
        return oss.str();
    }

    void LVector::rotate(double theta)
    {
        take_ownership();
        double s, c;
        math::sincos(theta, s, c);
        std::complex<double> z(c, -s);
        std::complex<double> imz(1., 0.);
        for (int m=1; m<=_order; m++) {
            imz *= z;
            for (PQIndex pq(m,0); !pq.pastOrder(_order); pq.incN()) {
                int r = pq.rIndex();
                std::complex<double> newb = std::complex<double>((*_v)[r], (*_v)[r+1]) * imz;
                (*_v)[r] = newb.real();
                (*_v)[r+1] = newb.imag();
            }
        }
    }

    //----------------------------------------------------------------
    //----------------------------------------------------------------
    // Calculate Laguerre polynomials and wavefunctions:

    // Fill LVector with the basis functions corresponding to each real DOF
    void LVector::fillBasis(double x, double y, double sigma)
    {
        take_ownership();
        // fill with psi_pq(z), where psi now defined to have 1/sigma^2 in
        // front.
        std::complex<double> z(x,-y);
        double rsq = norm(z);

        double tq = std::exp(-0.5*rsq) / (2*M_PI*sigma*sigma);
        double tqm1=tq;
        double tqm2;

        // Ascend m=0 first

        (*_v)[PQIndex(0,0).rIndex()]=tq;

        if (_order>=2) {
            tq = (rsq-1.)*tqm1;
            (*_v)[PQIndex(1,1).rIndex()] = tq;
        }

        PQIndex pq(2,2);
        for (int p=2; 2*p<=_order; ++p, pq.incN()) {
            tqm2 = tqm1;
            tqm1 = tq;
            tq = ((rsq-2.*p+1.)*tqm1 - (p-1.)*tqm2)/p;
            (*_v)[pq.rIndex()] = tq;
        }

        // Ascend all positive m's
        std::complex<double> zm = 2* (*_v)[PQIndex(0,0).rIndex()] * z;

        for (int m=1; m<=_order; m++) {
            pq.setPQ(m,0);
            double *r = &(*_v)[pq.rIndex()];
            *r = zm.real();
            *(r+1) = zm.imag();
            tq = 1.;
            tqm1 = 0.;

            for (pq.incN(); !pq.pastOrder(_order); pq.incN()) {
                tqm2 = tqm1;
                tqm1 = tq;
                int p=pq.getP(); int q=pq.getQ();
                tq = ( (rsq-(p+q-1.))*tqm1 - sqrtn(p-1)*sqrtn(q-1)*tqm2) / (sqrtn(p)*sqrtn(q));
                double *r = &(*_v)[pq.rIndex()];
                *r = tq*zm.real();
                *(r+1) = tq*zm.imag();
            }

            zm *= z/sqrtn(m+1);
        }
    }

    shared_ptr<MatrixXd > LVector::basis(
        const VectorXd& x, const VectorXd& y,
        int order, double sigma)
    {
        assert(x.size()==y.size());
        shared_ptr<MatrixXd > psi(new MatrixXd(x.size(), PQIndex::size(order)));
        basis(x, y, *psi, order, sigma);
        return psi;
    }

    // Forward declaration.  Implemented below.
    template <typename T>
    void CalculateBasis(
        const VectorXd& x, const VectorXd& y, const VectorXd* invsig,
        MatrixXT& psi,
        int order, double sigma);

    void LVector::basis(
        const VectorXd& x, const VectorXd& y,
        MatrixXd& psi, int order, double sigma)
    {
#ifdef USE_TMV
        xassert(y.size() == x.size() && psi.nrows() == x.size());
        xassert(psi.ncols()==PQIndex::size(order));
#else
        xassert(y.size() == x.size() && psi.rows() == x.size());
        xassert(psi.cols()==PQIndex::size(order));
#endif
        CalculateBasis(x, y, 0, psi, order, sigma);
    }

    shared_ptr<MatrixXd > LVector::design(
        const VectorXd& x, const VectorXd& y,
        const VectorXd& invsig, int order, double sigma)
    {
        shared_ptr<MatrixXd > psi(new MatrixXd(x.size(), PQIndex::size(order)));
        design(x, y, invsig, *psi, order, sigma);
        return psi;
    }

    void LVector::design(
        const VectorXd& x, const VectorXd& y,
        const VectorXd& invsig,
        MatrixXd& psi, int order, double sigma)
    {
#ifdef USE_TMV
        xassert(y.size() == x.size() && psi.nrows() == x.size() && invsig.size() == x.size());
        xassert(psi.ncols()==PQIndex::size(order));
#else
        xassert(y.size() == x.size() && psi.rows() == x.size() && invsig.size() == x.size());
        xassert(psi.cols()==PQIndex::size(order));
#endif
        CalculateBasis(x, y, &invsig, psi, order, sigma);
    }

    shared_ptr<MatrixXcd > LVector::kBasis(
        const VectorXd& kx, const VectorXd& ky,
        int order, double sigma)
    {
        assert (ky.size() == kx.size());
        shared_ptr<MatrixXcd > psi_k(new MatrixXcd(kx.size(), PQIndex::size(order)));
        kBasis(kx,ky,*psi_k,order,sigma);
        return psi_k;
    }

    void LVector::kBasis(
        const VectorXd& kx, const VectorXd& ky,
        MatrixXcd& psi_k, int order, double sigma)
    {
#ifdef USE_TMV
        xassert(ky.size() == kx.size() && psi_k.nrows() == kx.size());
        xassert(psi_k.ncols()==PQIndex::size(order));
#else
        xassert(ky.size() == kx.size() && psi_k.rows() == kx.size());
        xassert(psi_k.cols()==PQIndex::size(order));
#endif
        CalculateBasis(kx, ky, 0, psi_k, order, sigma);
    }

    // This helper class deals with the differences between the real and fourier calculations
    // in CalculateBasis.  First the real-space values:
    template <typename T>
    struct BasisHelper
    {
        static double Asign(int ) { return 1.; }

        static double Lsign(double x) { return x; }

        template <class V>
        static void applyPrefactor(V v, double sigma) { v *= 1./(2.*M_PI*sigma*sigma); }
    };

    // Now the fourier space version, marked by T being complex.
    template <typename T>
    struct BasisHelper<std::complex<T> >
    {
        // The "sign" of the eigenvectors are 1, -I, -1, I, and then repeat.
        // The input m4 should be m%4.
        static std::complex<double> Asign(int m4)
        {
            static std::complex<double> vals[4] = {
                std::complex<double>(1.,0.),
                std::complex<double>(0.,-1.),
                std::complex<double>(-1.,0.),
                std::complex<double>(0.,1.)
            };
            return vals[m4];
        }

        static double Lsign(double x) { return -x; }

        template <class V>
        static void applyPrefactor(V , double ) {}
    };

    template <typename T>
    void CalculateBasis(
        const VectorXd& x, const VectorXd& y, const VectorXd* invsig,
        MatrixXT& psi, int order, double sigma)
    {
        assert (y.size()==x.size());
#ifdef USE_TMV
        xassert (psi.nrows()==x.size() && psi.ncols()==PQIndex::size(order));
#else
        xassert (psi.rows()==x.size() && psi.cols()==PQIndex::size(order));
#endif

        const int N=order;
        const int npts_full = x.size();

        // It's faster to build the psi matrix in blocks so that more of the matrix stays in
        // L1 cache.  For a (typical) 256 KB L2 cache size, this corresponds to 8 columns in the
        // cache, which is pretty good, since we are usually working on 4 columns at a time,
        // plus either X and Y or 3 Lq vectors.
        const int BLOCKING_FACTOR=4096;

        const int max_npts = std::max(BLOCKING_FACTOR,npts_full);
        VectorXd Rsq_full(max_npts);
        MatrixXd A_full(max_npts,2);
        MatrixXd tmp_full(max_npts,2);
        VectorXd Lmq_full(max_npts);
        VectorXd Lmqm1_full(max_npts);
        VectorXd Lmqm2_full(max_npts);

        psi.setZero();

        for (int ilo=0; ilo<npts_full; ilo+=BLOCKING_FACTOR) {
            const int ihi = std::min(npts_full, ilo + BLOCKING_FACTOR);
            const int npts = ihi-ilo;

            // Cast arguments as diagonal matrices so we can access
            // vectorized element-by-element multiplication
#ifdef USE_TMV
            tmv::ConstVectorView<double> X = x.subVector(ilo,ihi);
            tmv::ConstVectorView<double> Y = y.subVector(ilo,ihi);
#else
            Eigen::VectorBlock<const VectorXd> X = x.segment(ilo,ihi-ilo);
            Eigen::VectorBlock<const VectorXd> Y = y.segment(ilo,ihi-ilo);
#endif

            // Get the appropriate portion of our temporary matrices.
#ifdef USE_TMV
            tmv::VectorView<double> Rsq = Rsq_full.subVector(0,npts);
            tmv::MatrixView<double> A = A_full.rowRange(0,npts);
            tmv::MatrixView<double> tmp = tmp_full.rowRange(0,npts);
#else
            Eigen::VectorBlock<VectorXd> Rsq = Rsq_full.segment(0,npts);
            Eigen::Block<MatrixXd> A = A_full.topRows(npts);
            Eigen::Block<MatrixXd> tmp = tmp_full.topRows(npts);
#endif

            // We need rsq values twice, so store them here.
#ifdef USE_TMV
            Rsq = ElemProd(X,X);
            Rsq += ElemProd(Y,Y);
#else
            Rsq.array() = X.array() * X.array();
            Rsq.array() += Y.array() * Y.array();
#endif

            // This matrix will keep track of real & imag parts
            // of prefactor * exp(-r^2/2) (x+iy)^m / sqrt(m!)

            // Build the Gaussian factor
#ifdef USE_TMV
            for (int i=0; i<npts; i++) A.ref(i,0) = std::exp(-0.5*Rsq(i));
#else
            for (int i=0; i<npts; i++) A.coeffRef(i,0) = std::exp(-0.5*Rsq(i));
#endif
            BasisHelper<T>::applyPrefactor(A.col(0),sigma);
            A.col(1).setZero();

            // Put 1/sigma factor into every point if doing a design matrix:
#ifdef USE_TMV
            if (invsig) A.col(0) *= tmv::DiagMatrixViewOf(invsig->subVector(ilo,ihi));
#else
            if (invsig) A.col(0).array() *= invsig->segment(ilo,ihi-ilo).array();
#endif

            // Assign the m=0 column first:
#ifdef USE_TMV
            psi.col(PQIndex(0,0).rIndex(), ilo,ihi) = A.col(0);
#else
            psi.col(PQIndex(0,0).rIndex()).segment(ilo,ihi-ilo) = A.col(0).cast<T>();
#endif

            // Then ascend m's at q=0:
            for (int m=1; m<=N; m++) {
                int rIndex = PQIndex(m,0).rIndex();
                // Multiply by (X+iY)/sqrt(m), including a factor 2 first time through
#ifdef USE_TMV
                tmp = DiagMatrixViewOf(Y) * A;
                A = DiagMatrixViewOf(X) * A;
#else
                tmp = Y.asDiagonal() * A;
                A = X.asDiagonal() * A;
#endif
                A.col(0) += tmp.col(1);
                A.col(1) -= tmp.col(0);
                A *= m==1 ? 2. : 1./sqrtn(m);

#ifdef USE_TMV
                psi.subMatrix(ilo,ihi,rIndex,rIndex+2) = BasisHelper<T>::Asign(m%4) * A;
#else
                psi.block(ilo,rIndex,ihi-ilo,2) = BasisHelper<T>::Asign(m%4) * A;
#endif
            }

            // Make three Vectors to hold Lmq's during recurrence calculations
#ifdef USE_TMV
            shared_ptr<tmv::VectorView<double> > Lmq(
                new tmv::VectorView<double>(Lmq_full.subVector(0,npts)));
            shared_ptr<tmv::VectorView<double> > Lmqm1(
                new tmv::VectorView<double>(Lmqm1_full.subVector(0,npts)));
            shared_ptr<tmv::VectorView<double> > Lmqm2(
                new tmv::VectorView<double>(Lmqm2_full.subVector(0,npts)));
#else
            shared_ptr<Eigen::VectorBlock<VectorXd> > Lmq(
                new Eigen::VectorBlock<VectorXd>(Lmq_full.segment(0,npts)));
            shared_ptr<Eigen::VectorBlock<VectorXd> > Lmqm1(
                new Eigen::VectorBlock<VectorXd>(Lmqm1_full.segment(0,npts)));
            shared_ptr<Eigen::VectorBlock<VectorXd> > Lmqm2(
                new Eigen::VectorBlock<VectorXd>(Lmqm2_full.segment(0,npts)));
#endif

            for (int m=0; m<=N; m++) {
                PQIndex pq(m,0);
                int iQ0 = pq.rIndex();
                // Go to q=1:
                pq.incN();
                if (pq.pastOrder(N)) continue;

                { // q == 1
                    const int p = pq.getP();
                    const int q = pq.getQ();
                    const int iQ = pq.rIndex();

#ifdef USE_TMV
                    Lmqm1->setAllTo(1.); // This is Lm0.
                    *Lmq = Rsq;
                    Lmq->addToAll(-(p+q-1.));
#else
                    Lmqm1->setConstant(1.);
                    Lmq->array()  = Rsq.array() - (p+q-1.);
#endif
                    *Lmq *= BasisHelper<T>::Lsign(1.) / (sqrtn(p)*sqrtn(q));

                    if (m==0) {
#ifdef USE_TMV
                        psi.col(iQ,ilo,ihi) = DiagMatrixViewOf(*Lmq) * psi.col(iQ0,ilo,ihi);
#else
                        psi.col(iQ).segment(ilo,ihi-ilo) = Lmq->asDiagonal() *
                            psi.col(iQ0).segment(ilo,ihi-ilo);
#endif
                    } else {
#ifdef USE_TMV
                        psi.subMatrix(ilo,ihi,iQ,iQ+2) = DiagMatrixViewOf(*Lmq) *
                            psi.subMatrix(ilo,ihi,iQ0,iQ0+2);
#else
                        psi.block(ilo,iQ,ihi-ilo,2) = Lmq->asDiagonal() *
                            psi.block(ilo,iQ0,ihi-ilo,2);
#endif
                    }
                }

                // do q=2,...
                for (pq.incN(); !pq.pastOrder(N); pq.incN()) {
                    const int p = pq.getP();
                    const int q = pq.getQ();
                    const int iQ = pq.rIndex();

                    // cycle the Lmq vectors
                    // Lmqm2 <- Lmqm1
                    // Lmqm1 <- Lmq
                    // Lmq <- Lmqm2
                    Lmqm2.swap(Lmqm1);
                    Lmqm1.swap(Lmq);

                    double invsqrtpq = 1./sqrtn(p)/sqrtn(q);
#ifdef USE_TMV
                    *Lmq = Rsq;
                    Lmq->addToAll(-(p+q-1.));
                    *Lmq = BasisHelper<T>::Lsign(invsqrtpq) * ElemProd(*Lmq, *Lmqm1);
#else
                    Lmq->array() = Rsq.array() - (p+q-1.);
                    Lmq->array() *= BasisHelper<T>::Lsign(invsqrtpq) * Lmqm1->array();
#endif
                    *Lmq -= (sqrtn(p-1)*sqrtn(q-1)*invsqrtpq) * (*Lmqm2);

                    if (m==0) {
#ifdef USE_TMV
                        psi.col(iQ,ilo,ihi) = DiagMatrixViewOf(*Lmq) * psi.col(iQ0,ilo,ihi);
#else
                        psi.col(iQ).segment(ilo,ihi-ilo) = Lmq->asDiagonal() *
                            psi.col(iQ0).segment(ilo,ihi-ilo);
#endif
                    } else {
#ifdef USE_TMV
                        psi.subMatrix(ilo,ihi,iQ,iQ+2) = DiagMatrixViewOf(*Lmq) *
                            psi.subMatrix(ilo,ihi,iQ0,iQ0+2);
#else
                        psi.block(ilo,iQ,ihi-ilo,2) = Lmq->asDiagonal() *
                            psi.block(ilo,iQ0,ihi-ilo,2);
#endif
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    // Flux determinations
    double LVector::flux(int maxP) const
    {
        if (maxP<0) maxP = getOrder()/2;
        if (maxP > getOrder()/2) maxP=getOrder()/2;
        double retval=0.;
        for (int p=0; p<=maxP; p++)
            retval += (*_v)[PQIndex(p,p).rIndex()];
        return retval;
    }

    double LVector::apertureFlux(double R_, int maxP) const
    {
        static shared_ptr<VectorXd > fp;
        static double R=-1.;
        static double psize=-1;

        assert(R_>=0.);

        if (maxP<0) maxP= getOrder()/2;
        if (maxP > getOrder()/2) maxP=getOrder()/2;

        if (!fp.get() || R_ != R || maxP>psize) {
            fp.reset(new VectorXd(maxP));
            psize = maxP;
            R = R_;
            VectorXd Lp(maxP+1);
            VectorXd Qp(maxP+1);
            double x = R*R;
            double efact = std::exp(-0.5*x);
            Lp[0] = Qp[0]=1.;
            if (maxP>0) {
                Lp[1] = 1. - x;
                Qp[1] = -1. - x;
            }
            for (int p=1; p<maxP; p++) {
                Lp[p+1] = ((2*p+1-x)*Lp[p]-p*Lp[p-1])/(p+1);
                Qp[p+1] = (-x*Lp[p]-Qp[p]+p*Qp[p-1])/(p+1);
            }
            for (int p=0; p<=maxP; p++)
                (*fp)[p] = 1. - efact*Qp[p]*(p%2==0 ? 1. : -1.);
        }

        double flux = 0.;
        for (int p=0; p<=maxP; p++)
            flux += (*_v)[PQIndex(p,p).rIndex()] * (*fp)[p];
        return flux;
    }

    //---------------------------------------------------------------------------
    //---------------------------------------------------------------------------
    // I/O Routines

    std::ostream& operator<<(std::ostream& os, const LVector& lv)
    { lv.write(os); return os; }

    void LVector::write(std::ostream& os, int maxorder) const
    {
        int oldprec = os.precision(8);
        std::ios::fmtflags oldf = os.setf(std::ios::scientific,std::ios::floatfield);
        if (maxorder < 0 || maxorder > _order)
            maxorder = _order;
        os << _order << std::endl;
        for (int n=0; n<=maxorder; n++) {
            for(PQIndex pq(n,0); !pq.needsConjugation(); pq.decm()) {
                os << " " << std::setw(2) << pq.getP()
                    << " " << std::setw(2) << pq.getQ() ;
                if (pq.isReal()) {
                    os << " " << std::setw(15) << (*this)[pq].real() << std::endl;
                } else {
                    os << " " << std::setw(15) << (*this)[pq].real()
                        << " " << std::setw(15) << (*this)[pq].imag() << std::endl;
                }
            }
        }
        os.precision(oldprec);
        os.flags(oldf);
    }

    std::istream& operator>>(std::istream& is, LVector& lv)
    { lv.read(is); return is; }

    void LVector::read(std::istream& is)
    {
        // get order
        int order;
        is >> order;
        resize(order);
        // discard p,q info, read into rVector
        int p, q;
        double re, im;
        for (int n=0; n<=order; n++) {
            for(PQIndex pq(n,0); !pq.needsConjugation(); pq.decm()) {
                is >> p >> q;
                if (pq.isReal()) {
                    is >> re;  im = 0.;
                    (*this)[pq]=std::complex<double>(re,im);
                } else {
                    is >> re >> im;
                    (*this)[pq]=std::complex<double>(re,im);
                }
            }
        }
    }

    void PQIndex::write(std::ostream& os) const
    {
        os << std::setw(2) << getP()
            << "," << std::setw(2) << getQ() ;
    }

    // Function to solve for radius enclosing a specified flux.
    // Return negative radius if no root is apparent.
    class FRSolve
    {
    public:
        FRSolve(const LVector& lv_, double thresh_, int maxP_):
            lv(lv_), maxP(maxP_), thresh(thresh_)
        { assert(lv.getOrder() >= 2*maxP); }

        double operator()(double u) const { return lv.apertureFlux(u,maxP)-thresh; }

    private:
        const LVector& lv;
        int maxP;
        double thresh;
    };

    double fluxRadius(const LVector& lv, double threshold, int maxP)
    {
        if (maxP<0) maxP= lv.getOrder()/2;
        if (maxP > lv.getOrder()/2) maxP=lv.getOrder()/2;
        FRSolve func(lv, threshold, maxP);

        // First we step through manually at intervals roughly the smallest that
        // a function of this order can oscillate, in order to bracket the root
        // closest to the origin.

        const double TOLERANCE=0.001; //radius accuracy required
        const double maxR = 5.;
        double ustep=0.5/sqrt(double(maxP)+1.);
        double u1 = 0.0001;
        double f1 = func(u1);
        double u2;
        while (u1<maxR) {
            u2 = u1 + ustep;
            double f2 = func(u2);
            if ( f1*f2<=0.) break;
            u1 = u2;
            f1 = f2;
        }
        if (u1>=maxR) {
            u2 = 2*maxR;
            double f2 = func(u2);
            if ( f1*f2>0.) {
                // At this point there appears to be no root.
                return -1.;
            }
        }

        // Now a bisection solution for the root
        Solve<FRSolve> s(func, u1, u2);
        s.setXTolerance(TOLERANCE);
        return s.root();
    }

}
