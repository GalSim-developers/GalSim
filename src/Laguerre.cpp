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

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

#include "BinomFact.h"
#include "Laguerre.h"
#include "Solve.h"

namespace galsim {

    void LVector::rotate(const Angle& theta) 
    {
        take_ownership();
        std::complex<double> z(std::cos(theta.rad()), -std::sin(theta.rad()));
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

    // routines to retrieve and save complex elements of LTransform:
    // ???? Check these ???
    std::complex<double> LTransform::operator()(PQIndex pq1, PQIndex pq2) const 
    {
        assert(pq1.pqValid() && !pq1.pastOrder(_orderOut));
        assert(pq2.pqValid() && !pq2.pastOrder(_orderIn));
        int r1index=pq1.rIndex();
        int r2index=pq2.rIndex();
        int i1index=(pq1.isReal()? r1index: r1index+1);
        int i2index=(pq2.isReal()? r2index: r2index+1);

        double x = (*_m)(r1index,r2index) + pq1.iSign()*pq2.iSign()*(*_m)(i1index,i2index);
        double y = pq1.iSign()*(*_m)(i1index,r2index) - pq2.iSign()*(*_m)(r1index,i2index);

        std::complex<double> z(x,y);
        if (pq2.isReal()) z *= 0.5;

        return z;
    }

    void LTransform::set(
        PQIndex pq1, PQIndex pq2, std::complex<double> Cpq1pq2, std::complex<double> Cqp1pq2) 
    {
        assert(pq1.pqValid() && !pq1.pastOrder(_orderOut));
        assert(pq2.pqValid() && !pq2.pastOrder(_orderIn));

        take_ownership();
        const double RoundoffTolerance=1.e-15;
        std::complex<double> Cpq1qp2;

        if (pq2.needsConjugation()) {
            pq2 = pq2.swapPQ();
            std::complex<double> tmp=conj(Cqp1pq2);
            Cqp1pq2 = conj(Cpq1pq2);
            Cpq1pq2 = tmp;
        }
        if (pq1.needsConjugation()) {
            pq1 = pq1.swapPQ();
            std::complex<double> tmp=Cqp1pq2;
            Cqp1pq2 = Cpq1pq2;
            Cpq1pq2 = tmp;
        }

        int rIndex1 = pq1.rIndex();
        int rIndex2 = pq2.rIndex();
        int iIndex1 = rIndex1+1;
        int iIndex2 = rIndex2+1;

        if (pq1.isReal()) {
            if (Cpq1pq2!=Cqp1pq2) {
                FormatAndThrow<>() 
                    << "Invalid LTransform elements for p1=q1, " << Cpq1pq2
                    << " != " << Cqp1pq2;
            }
            (*_m)(rIndex1,rIndex2) = Cpq1pq2.real() * (pq2.isReal()? 1. : 2.);
            if (pq2.isReal()) {
                if (std::abs(Cpq1pq2.imag()) > RoundoffTolerance) {
                    FormatAndThrow<>() 
                        << "Nonzero imaginary LTransform elements for p1=q1, p2=q2: " 
                        << Cpq1pq2;
                }
            } else {
                (*_m)(rIndex1,iIndex2) = -2.*Cpq1pq2.imag();
            }
            return;
        } else if (pq2.isReal()) {
            // Here we know p1!=q1:
            if (norm(Cpq1pq2-conj(Cqp1pq2))>RoundoffTolerance) {
                FormatAndThrow<>() 
                    << "Inputs to LTransform.set are not conjugate for p2=q2: "
                    << Cpq1pq2 << " vs " << Cqp1pq2 ;
            }
            (*_m)(rIndex1, rIndex2) = Cpq1pq2.real();
            (*_m)(iIndex1, rIndex2) = Cpq1pq2.imag();
        } else {
            // Neither pq is real:
            std::complex<double> z=Cpq1pq2 + Cqp1pq2;
            (*_m)(rIndex1, rIndex2) = z.real();
            (*_m)(rIndex1, iIndex2) = -z.imag();
            z=Cpq1pq2 - Cqp1pq2;
            (*_m)(iIndex1, rIndex2) = z.imag();
            (*_m)(iIndex1, iIndex2) = z.real();
        }
    }

    LVector LTransform::operator*(const LVector rhs) const 
    {
        if (_orderIn != rhs.getOrder()) 
            FormatAndThrow<>() 
                << "Order mismatch between LTransform [" << _orderIn
                << "] and LVector [" << rhs.getOrder()
                << "]";
        boost::shared_ptr<tmv::Vector<double> > out(new tmv::Vector<double>(sizeOut()));
        *out = (*_m) * rhs.rVector();
        return LVector(_orderOut, out);
    }

    LTransform LTransform::operator*(const LTransform rhs) const 
    {
        if (_orderIn != rhs.getOrderOut()) 
            FormatAndThrow<>()  
                << "Order mismatch between LTransform [" << _orderIn
                << "] and LTransform [" << rhs.getOrderOut()
                << "]";
        boost::shared_ptr<tmv::Matrix<double> > out(
            new tmv::Matrix<double>(sizeOut(),rhs.sizeIn()));
        *out = (*_m) * (*rhs._m);
        return LTransform(_orderOut, rhs._orderIn, out);
    }

    LTransform& LTransform::operator*=(const LTransform rhs) 
    {
        take_ownership();
        if (_orderIn != rhs.getOrderOut())
            FormatAndThrow<>()
                << "Order mismatch between LTransform [" << _orderIn
                << "] and LTransform [" << rhs.getOrderOut()
                << "]";
        (*_m) *= (*rhs._m);
        _orderIn = rhs._orderOut;
        return *this;
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

    boost::shared_ptr<tmv::Matrix<double> > LVector::basis(
        const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
        int order, double sigma)
    {
        assert(x.size()==y.size());
        boost::shared_ptr<tmv::Matrix<double> > mr(
            new tmv::Matrix<double>(x.size(), PQIndex::size(order)));
        basis(mr->view(), x, y, order, sigma);
        return mr;
    }

    void LVector::basis(
        tmv::MatrixView<double> psi,
        const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
        int order, double sigma)
    {
        assert(y.size() == x.size() && psi.nrows() == x.size());
        assert(psi.ncols()==PQIndex::size(order));
        mBasis(x, y, 0, &psi, 0, order, sigma);
    }

    boost::shared_ptr<tmv::Matrix<double> > LVector::design(
        const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
        const tmv::ConstVectorView<double>& invsig, int order, double sigma)
    {
        boost::shared_ptr<tmv::Matrix<double> > mr(
            new tmv::Matrix<double>(x.size(), PQIndex::size(order)));
        design(mr->view(), x, y, invsig, order, sigma);
        return mr;
    }

    void LVector::design(
        tmv::MatrixView<double> psi,
        const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
        const tmv::ConstVectorView<double>& invsig, int order, double sigma)
    {
        assert(y.size() == x.size() && psi.nrows() == x.size() && invsig.size() == x.size());
        assert(psi.ncols()==PQIndex::size(order));
        mBasis(x, y, &invsig, &psi, 0, order, sigma);
    }

    void LVector::kBasis(
        boost::shared_ptr<tmv::Matrix<double> >& psi_kReal,
        boost::shared_ptr<tmv::Matrix<double> >& psi_kImag,
        const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
        int order, double sigma)
    {
        assert (ky.size() == kx.size());
        const int npts = kx.size();
        const int ndof=PQIndex::size(order);
        if (!psi_kReal.get() || psi_kReal->nrows()!=npts || psi_kReal->ncols()!=ndof) {
            psi_kReal.reset(new tmv::Matrix<double>(npts, ndof));
        }
        if (!psi_kImag.get() || psi_kImag->nrows()!=npts || psi_kImag->ncols()!=ndof) {
            psi_kImag.reset(new tmv::Matrix<double>(npts, ndof, 0.));
        }
        kBasis(psi_kReal->view(),psi_kImag->view(),kx,ky,order,sigma);
    }

    void LVector::kBasis(
        tmv::MatrixView<double> psi_kReal, tmv::MatrixView<double> psi_kImag,
        const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
        int order, double sigma)
    {
        assert (ky.size() == kx.size());
        assert (psi_kReal.nrows() == kx.size());
        assert (psi_kImag.nrows() == kx.size());
        assert (psi_kReal.ncols() == PQIndex::size(order));
        assert (psi_kImag.ncols() == PQIndex::size(order));
        mBasis(kx, ky, 0, &psi_kReal, &psi_kImag, order, sigma);
    }

    void LVector::kBasis(
        boost::shared_ptr<tmv::Matrix<std::complex<double> > >& psi_k,
        const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
        int order, double sigma)
    {
        assert (ky.size() == kx.size());
        const int ndof=PQIndex::size(order);
        const int npts = kx.size();
        if (!psi_k.get() || psi_k->nrows()!=npts || psi_k->ncols()!=ndof) {
            psi_k.reset(new tmv::Matrix<std::complex<double> >(npts, ndof, 0.));
        }
        kBasis(psi_k->view(),kx,ky,order,sigma);
    }

    void LVector::kBasis(
        tmv::MatrixView<std::complex<double> > psi_k,
        const tmv::ConstVectorView<double>& kx, const tmv::ConstVectorView<double>& ky,
        int order, double sigma)
    { return kBasis(psi_k.realPart(), psi_k.imagPart(), kx, ky, order, sigma); }

    void LVector::mBasis(
        const tmv::ConstVectorView<double>& x, const tmv::ConstVectorView<double>& y,
        const tmv::ConstVectorView<double>* invsig,
        tmv::MatrixView<double>* mr, tmv::MatrixView<double>* mi,
        int order, double sigma)
    {
        assert (y.size()==x.size());
        assert (mr->nrows()==x.size() && mr->ncols()==PQIndex::size(order));
        if (mi) assert (mi->nrows()==x.size() && mi->ncols()==PQIndex::size(order));

        const int N=order;
        const int npts_full = x.size();
        const bool isK = mi;
        if (isK) { mr->setZero(); mi->setZero(); }

        // It's faster to build the psi matrix in blocks so that more of the matrix stays in 
        // L1 cache.  For a (typical) 256 KB L2 cache size, this corresponds to 8 columns in the 
        // cache, which is pretty good, since we are usually working on 4 columns at a time, 
        // plus either X and Y or 3 Lq vectors.
        const int BLOCKING_FACTOR=4096;

        const int max_npts = std::max(BLOCKING_FACTOR,npts_full);
        tmv::DiagMatrix<double> Rsq_full(max_npts);
        tmv::Matrix<double> A_full(max_npts,2);
        tmv::Matrix<double> tmp_full(max_npts,2);
        tmv::DiagMatrix<double> Lmq_full(max_npts);
        tmv::DiagMatrix<double> Lmqm1_full(max_npts);
        tmv::DiagMatrix<double> Lmqm2_full(max_npts);

        for (int ilo=0; ilo<npts_full; ilo+=BLOCKING_FACTOR) {
            const int ihi = std::min(npts_full, ilo + BLOCKING_FACTOR);
            const int npts = ihi-ilo;

            // Cast arguments as diagonal matrices so we can access
            // vectorized element-by-element multiplication
            tmv::ConstDiagMatrixView<double> X = DiagMatrixViewOf(x.subVector(ilo,ihi));
            tmv::ConstDiagMatrixView<double> Y = DiagMatrixViewOf(y.subVector(ilo,ihi));

            // Get the appropriate portion of our temporary matrices.
            tmv::DiagMatrixView<double> Rsq = Rsq_full.subDiagMatrix(0,npts);
            tmv::MatrixView<double> A = A_full.rowRange(0,npts);
            tmv::MatrixView<double> tmp = tmp_full.rowRange(0,npts);

            // We need rsq values twice, so store them here.
            Rsq = X*X;
            Rsq += Y*Y;

            // This matrix will keep track of real & imag parts
            // of prefactor * exp(-r^2/2) (x+iy)^m / sqrt(m!)

            // Build the Gaussian factor
            for (int i=0; i<npts; i++) A.ref(i,0) = std::exp(-0.5*Rsq(i));

            // Apply the appropriate prefactor depending on real or fourier
            A.col(0) *= isK ? 1. : 1./(2.*M_PI*sigma*sigma);
            A.col(1).setZero();

            // Put 1/sigma factor into every point if doing a design matrix:
            if (invsig) A.col(0) *= tmv::DiagMatrixViewOf(invsig->subVector(ilo,ihi));

            // Assign the m=0 column first:
            mr->col( PQIndex(0,0).rIndex(), ilo,ihi ) = A.col(0);

            // Then ascend m's at q=0:
            for (int m=1; m<=N; m++) {
                int rIndex = PQIndex(m,0).rIndex();
                // Multiply by (X+iY)/sqrt(m), including a factor 2 first time through
                tmp = Y * A;
                A = X * A;
                A.col(0) += tmp.col(1);
                A.col(1) -= tmp.col(0);
                A *= (( m==1 ? 2. : 1.) / sqrtn(m)) *
                    ((isK && (m%4 == 1 || m%4 == 2)) ? -1. : 1.);

                if (!isK || m%2 == 0) mr->subMatrix(ilo,ihi,rIndex,rIndex+2) = A;
                else mi->subMatrix(ilo,ihi,rIndex,rIndex+2) = A;
            }

            // Make three DiagMatrix to hold Lmq's during recurrence calculations
            boost::shared_ptr<tmv::DiagMatrixView<double> > Lmq(
                new tmv::DiagMatrixView<double>(Lmq_full.subDiagMatrix(0,npts)));
            boost::shared_ptr<tmv::DiagMatrixView<double> > Lmqm1(
                new tmv::DiagMatrixView<double>(Lmqm1_full.subDiagMatrix(0,npts)));
            boost::shared_ptr<tmv::DiagMatrixView<double> > Lmqm2(
                new tmv::DiagMatrixView<double>(Lmqm2_full.subDiagMatrix(0,npts)));

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

                    Lmqm1->setAllTo(1.); // This is Lm0.
                    *Lmq = Rsq - (p+q-1.);
                    *Lmq *= (isK ? -1. : 1.) / (sqrtn(p)*sqrtn(q));

                    if (m==0) {
                        // even m's have real transforms
                        mr->col(iQ,ilo,ihi) = (*Lmq) * mr->col(iQ0,ilo,ihi);
                    } else if (!isK || m%2==0) {
                        // even m's have real transforms
                        mr->subMatrix(ilo,ihi,iQ,iQ+2) = (*Lmq) * mr->subMatrix(ilo,ihi,iQ0,iQ0+2);
                    } else {
                        // odd m's have imag transforms
                        mi->subMatrix(ilo,ihi,iQ,iQ+2) = (*Lmq) * mi->subMatrix(ilo,ihi,iQ0,iQ0+2);
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
                    *Lmq = Rsq - (p+q-1.);
                    *Lmq *= (isK ? -invsqrtpq : invsqrtpq) * *Lmqm1;
                    *Lmq -= (sqrtn(p-1)*sqrtn(q-1)*invsqrtpq) * (*Lmqm2);

                    if (m==0) {
                        // even m's have real transforms
                        mr->col(iQ,ilo,ihi) = (*Lmq) * mr->col(iQ0,ilo,ihi);
                    } else if (!isK || m%2==0) {
                        // even m's have real transforms
                        mr->subMatrix(ilo,ihi,iQ,iQ+2) = (*Lmq) * mr->subMatrix(ilo,ihi,iQ0,iQ0+2);
                    } else {
                        // odd m's have imag transforms
                        mi->subMatrix(ilo,ihi,iQ,iQ+2) = (*Lmq) * mi->subMatrix(ilo,ihi,iQ0,iQ0+2);
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
        static boost::shared_ptr<tmv::Vector<double> > fp;
        static double R=-1.;
        static double psize=-1;

        assert(R_>=0.);

        if (maxP<0) maxP= getOrder()/2;
        if (maxP > getOrder()/2) maxP=getOrder()/2;

        if (!fp.get() || R_ != R || maxP>psize) {
            fp.reset(new tmv::Vector<double>(maxP));
            psize = maxP;
            R = R_;
            tmv::Vector<double> Lp(maxP+1);
            tmv::Vector<double> Qp(maxP+1);
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

    // Transformation generators - these return a view into static quantities:
    const tmv::ConstMatrixView<double> LVector::Generator(
        GType iparam, int orderOut, int orderIn)
    {
        static boost::shared_ptr<tmv::Matrix<double> > gmu;
        static boost::shared_ptr<tmv::Matrix<double> > gx;
        static boost::shared_ptr<tmv::Matrix<double> > gy;
        static boost::shared_ptr<tmv::Matrix<double> > ge1;
        static boost::shared_ptr<tmv::Matrix<double> > ge2;
        static boost::shared_ptr<tmv::Matrix<double> > grot;

        const int sizeIn = PQIndex::size(orderIn);
        const int sizeOut = PQIndex::size(orderOut);

        const int order = std::max(orderOut, orderIn);
        if (iparam==iMu) {
            if (!gmu.get() || gmu->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);

                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int p=pq.getP();
                    int q=pq.getQ();
                    std::complex<double> zz(-1.,0.);
                    if (pq.isReal()) lt.set(pq,pq,zz, zz);
                    else lt.set(pq,pq,zz, 0.);
                    PQIndex pqprime(p+1, q+1);
                    if (!pqprime.pastOrder(order)) {
                        zz = std::complex<double>(-sqrtn(p+1)*sqrtn(q+1), 0.);
                        if (pq.isReal()) lt.set(pq,pqprime,zz, zz);
                        else lt.set(pq,pqprime,zz, 0.);
                    }
                    if (q>0) {
                        pqprime.setPQ(p-1,q-1);
                        zz = std::complex<double>(sqrtn(p)*sqrtn(q), 0.);
                        if (pq.isReal()) lt.set(pq,pqprime,zz, zz);
                        else lt.set(pq,pqprime,zz, 0.);
                    }
                }
                gmu.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return gmu->subMatrix(0, sizeOut, 0, sizeIn);
        }
        if (iparam==iX) {
            if (!gx.get() || gx->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);

                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int p=pq.getP();
                    int q=pq.getQ();
                    PQIndex pqprime(p+1, q);
                    std::complex<double> zz(-0.5*sqrtn(p+1),0.);
                    if (pq.isReal()) {
                        if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
                        if (p>0) {
                            zz = std::complex<double>(0.5*sqrtn(p), 0.);
                            pqprime.setPQ(p-1,q);
                            lt.set(pq,pqprime,zz, zz);
                        }
                    } else {
                        if (!pqprime.pastOrder(order)) {  
                            lt.set(pq,pqprime,zz, 0.);
                            pqprime.setPQ(p, q+1);
                            zz = std::complex<double>(-0.5*sqrtn(q+1),0.);
                            if (pq.m()==1) {
                                lt.set(pq,pqprime, zz, zz);
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                        pqprime.setPQ(p-1,q);
                        zz = std::complex<double>(0.5*sqrtn(p), 0.);
                        if (pq.m()==1) {
                            lt.set(pq,pqprime, zz, zz);
                        } else {
                            lt.set(pq,pqprime, zz, 0.);
                        }
                        if (q>0) {
                            pqprime.setPQ(p,q-1);
                            zz = std::complex<double>(0.5*sqrtn(q), 0.);
                            lt.set(pq,pqprime, zz, 0.);
                        }
                    }
                }
                gx.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return gx->subMatrix(0, sizeOut, 0, sizeIn);
        }

        if (iparam==iY) {
            if (!gy.get() || gy->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);

                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int p=pq.getP();
                    int q=pq.getQ();
                    PQIndex pqprime(p+1, q);
                    std::complex<double> zz(0.,-0.5*sqrtn(p+1));
                    if (pq.isReal()) {
                        if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
                        if (p>0) {
                            zz = std::complex<double>(0.,0.5*sqrtn(q));
                            pqprime.setPQ(p,q-1);
                            lt.set(pq,pqprime,zz, zz);
                        }
                    } else {
                        if (!pqprime.pastOrder(order)) {
                            lt.set(pq,pqprime,zz, 0.);
                            pqprime.setPQ(p, q+1);
                            zz = std::complex<double>(0.,0.5*sqrtn(q+1));
                            if (pq.m()==1) {
                                lt.set(pq,pqprime, zz, conj(zz));
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                        pqprime.setPQ(p-1,q);
                        zz = std::complex<double>(0.,-0.5*sqrtn(p));
                        if (pq.m()==1) {
                            lt.set(pq,pqprime, zz, conj(zz));
                        } else {
                            lt.set(pq,pqprime, zz, 0.);
                        }
                        if (q>0) {
                            pqprime.setPQ(p,q-1);
                            zz = std::complex<double>(0.,0.5*sqrtn(q));
                            lt.set(pq,pqprime, zz, 0.);
                        }
                    }
                }
                gy.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return gy->subMatrix(0, sizeOut, 0, sizeIn);
        }

        if (iparam==iE1) {
            if (!ge1.get() || ge1->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);

                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int p=pq.getP();
                    int q=pq.getQ();
                    PQIndex pqprime(p+2, q);
                    std::complex<double> zz(-0.25*sqrtn(p+1)*sqrtn(p+2),0.);
                    if (pq.isReal()) {
                        if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
                        if (p>1) {
                            zz = std::complex<double>(0.25*sqrtn(p)*sqrtn(p-1),0.);
                            pqprime.setPQ(p-2,q);
                            lt.set(pq,pqprime,zz, zz);
                        }
                    } else {
                        if (!pqprime.pastOrder(order)) {
                            lt.set(pq,pqprime,zz, 0.);
                            pqprime.setPQ(p, q+2);
                            zz = std::complex<double>(-0.25*sqrtn(q+1)*sqrtn(q+2),0.);
                            if (pq.m()==2) {
                                lt.set(pq,pqprime, zz, zz);
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                        if (p>1) {
                            pqprime.setPQ(p-2,q);
                            zz = std::complex<double>(0.25*sqrtn(p)*sqrtn(p-1),0.);
                            if (pq.m()==2) {
                                lt.set(pq,pqprime, zz, zz);
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                            if (q>1) {
                                pqprime.setPQ(p,q-2);
                                zz = std::complex<double>(0.25*sqrtn(q)*sqrtn(q-1),0.);
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                    }
                }
                ge1.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return ge1->subMatrix(0, sizeOut, 0, sizeIn);
        }

        if (iparam==iE2) {
            if (!ge2.get() || ge2->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);

                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int p=pq.getP();
                    int q=pq.getQ();
                    PQIndex pqprime(p+2, q);
                    std::complex<double> zz(0., -0.25*sqrtn(p+1)*sqrtn(p+2));
                    if (pq.isReal()) {
                        if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
                        if (p>1) {
                            zz = std::complex<double>(0.,-0.25*sqrtn(p)*sqrtn(p-1));
                            pqprime.setPQ(p-2,q);
                            lt.set(pq,pqprime,zz, zz);
                        }
                    } else {
                        if (!pqprime.pastOrder(order)) {
                            lt.set(pq,pqprime,zz, 0.);
                            pqprime.setPQ(p, q+2);
                            zz = std::complex<double>(0.,0.25*sqrtn(q+1)*sqrtn(q+2));
                            if (pq.m()==2) {
                                lt.set(pq,pqprime, zz, conj(zz));
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                        if (p>1) {
                            pqprime.setPQ(p-2,q);
                            zz = std::complex<double>(0.,-0.25*sqrtn(p)*sqrtn(p-1));
                            if (pq.m()==2) {
                                lt.set(pq,pqprime, zz, conj(zz));
                            } else {
                                lt.set(pq,pqprime, zz, 0.);
                            }
                            if (q>1) {
                                pqprime.setPQ(p,q-2);
                                zz = std::complex<double>(0.,0.25*sqrtn(q)*sqrtn(q-1));
                                lt.set(pq,pqprime, zz, 0.);
                            }
                        }
                    }
                }
                ge2.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return ge2->subMatrix(0, sizeOut, 0, sizeIn);
        }

        if (iparam==iRot) {
            // Rotation is diagonal - could use a DiagMatrix perhaps
            if (!grot.get() || grot->nrows()<PQIndex::size(order)) {
                LTransform lt(order, order);
                for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
                    int m = pq.m();
                    if (m>0) lt.set(pq,pq, std::complex<double>(0.,-m), 0.);
                }
                grot.reset(new tmv::Matrix<double>(lt.rMatrix()));
            }
            return grot->subMatrix(0, sizeOut, 0, sizeIn);
        } else {
            throw std::runtime_error("Unknown parameter for LVector::Generator()");
        }
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
