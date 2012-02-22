#include <cmath>
#include <iostream>
#include <string>
#include <ios>

#include "BinomFact.h"
#include "Laguerre.h"
#include "Solve.h"
//**/#include "Stopwatch.h"

// For making the vectorized versions run quickly:
#define PTRLOOPS
const int BLOCKING_FACTOR=1024;

using namespace laguerre;

// Build an LVector from a DVector for the real degrees of freedom.
// DVector must have same dimension as needed for LVector of chosen order
LVector::LVector(const DVector& rhs, int order_): 
  order(new int(order_)), 
  pcount(new int(1)),
  v(new DVector(rhs)) {
  if (v->size()!=PQIndex::size(*order)) {
      delete v;
      delete pcount;
      delete order;
      throw LaguerreError("Input to LVector(DVector) is wrong size for order");
  }
}

void
LVector::rotate(double theta) {
  DComplex z(cos(theta), -sin(theta));
  DComplex imz(1., 0.);
  for (int m=1; m<=*order; m++) {
    imz *= z;
    for (PQIndex pq(m,0); !pq.pastOrder(*order); pq.incN()) {
      double* r= &(*v)[pq.rIndex()];
      DComplex newb = DComplex(*r, *(r+1)) * imz;
      *r = newb.real(); *(r+1)=newb.imag();
    }
  }
}


void
LTransform::identity() {
  m->setZero();
  for (int i=0; i<MIN(m->ncols(), m->nrows()); i++)
    (*m)(i,i)=1.;
}

//
LTransform::LTransform(const DMatrix& rhs, int orderOut_, int orderIn_):
  orderIn(new int(orderIn_)),
  orderOut(new int(orderOut_)),
  pcount(new int(1)),
  m(new DMatrix(rhs))  {
  if (m->ncols()!=PQIndex::size(*orderIn)
      || m->nrows()!=PQIndex::size(*orderOut)) {
      delete m;
      delete pcount;
      delete orderIn;
      delete orderOut;
      throw LaguerreError("Input to LTransform(DMatrix) is wrong size for orders");
  }
}

// routines to retrieve and save complex elements of LTransform:
// ???? Check these ???
DComplex 
LTransform::operator()(PQIndex pq1, PQIndex pq2) const {
  Assert(pq1.pqValid() && !pq1.pastOrder(*orderOut));
  Assert(pq2.pqValid() && !pq2.pastOrder(*orderIn));
  int r1index=pq1.rIndex();
  int r2index=pq2.rIndex();
  int i1index=(pq1.isReal()? r1index: r1index+1);
  int i2index=(pq2.isReal()? r2index: r2index+1);

  DComplex z( (*m)(r1index,r2index) 
	      + pq1.iSign()*pq2.iSign()*(*m)(i1index,i2index),
	      pq1.iSign()*(*m)(i1index,r2index)
	      - pq2.iSign()*(*m)(r1index,i2index) );

  return (pq2.isReal()? z : 0.5*z);
}

void 
LTransform::set(PQIndex pq1, PQIndex pq2,
		DComplex Cpq1pq2,
		DComplex Cqp1pq2) {
  Assert(pq1.pqValid() && !pq1.pastOrder(*orderOut));
  Assert(pq2.pqValid() && !pq2.pastOrder(*orderIn));

  const double RoundoffTolerance=1.e-15;
  DComplex Cpq1qp2;

  if (pq2.needsConjugation()) {
    pq2 = pq2.swapPQ();
    DComplex tmp=conj(Cqp1pq2);
    Cqp1pq2 = conj(Cpq1pq2);
    Cpq1pq2 = tmp;
  }
  if (pq1.needsConjugation()) {
    pq1 = pq1.swapPQ();
    DComplex tmp=Cqp1pq2;
    Cqp1pq2 = Cpq1pq2;
    Cpq1pq2 = tmp;
  }

  int rIndex1 = pq1.rIndex();
  int rIndex2 = pq2.rIndex();
  int iIndex1 = rIndex1+1;
  int iIndex2 = rIndex2+1;

  if (pq1.isReal()) {
    if (Cpq1pq2!=Cqp1pq2) {
      std::ostringstream oss;
      oss << "Invalid LTransform elements for p1=q1, " << Cpq1pq2
	  << " != " << Cqp1pq2;
      throw LaguerreError(oss.str());
    }
    (*m)(rIndex1,rIndex2) = Cpq1pq2.real() * (pq2.isReal()? 1. : 2.);
    if (pq2.isReal()) {
      if (abs(Cpq1pq2.imag()) > RoundoffTolerance) {
	std::ostringstream oss;
	oss << "Nonzero imaginary LTransform elements for p1=q1, p2=q2: " 
	    << Cpq1pq2;
	throw LaguerreError(oss.str());
      }
    } else {
      (*m)(rIndex1,iIndex2) = -2.*Cpq1pq2.imag();
    }
    return;
  } else if (pq2.isReal()) {
    // Here we know p1!=q1:
    if (norm(Cpq1pq2-conj(Cqp1pq2))>RoundoffTolerance) {
      std::ostringstream oss;
      oss << "Inputs to LTransform.set are not conjugate for p2=q2: "
	  << Cpq1pq2 << " vs " << Cqp1pq2 ;
      throw LaguerreError(oss.str());
    }
    (*m)(rIndex1, rIndex2) = Cpq1pq2.real();
    (*m)(iIndex1, rIndex2) = Cpq1pq2.imag();
  } else {
    // Neither pq is real:
    DComplex z=Cpq1pq2 + Cqp1pq2;
    (*m)(rIndex1, rIndex2) = z.real();
    (*m)(rIndex1, iIndex2) = -z.imag();
    z=Cpq1pq2 - Cqp1pq2;
    (*m)(iIndex1, rIndex2) = z.imag();
    (*m)(iIndex1, iIndex2) = z.real();
  }
}

LVector
LTransform::operator*(const LVector rhs) const {
  if (*orderIn != rhs.getOrder()) 
    FormatAndThrow<LaguerreError>() << "Order mismatch between LTransform [" << *orderIn
				    << "] and LVector [" << rhs.getOrder()
				    << "]";
  DVector out = (*m) * rhs.rVector();
  // ??? avoid extra copy by assigning directly to output?
  return LVector(out, *orderOut);
}

LTransform
LTransform::operator*(const LTransform rhs) const {
  if (*orderIn != rhs.getOrderOut()) 
    FormatAndThrow<LaguerreError>()  << "Order mismatch between LTransform [" << *orderIn
				     << "] and LTransform [" << rhs.getOrderOut()
				     << "]";
  DMatrix out = (*m) * (*rhs.m);
  // ??? avoid extra copy by assigning directly to output?
  return LTransform(out, *orderOut, *(rhs.orderIn));
}

LTransform&
LTransform::operator*=(const LTransform rhs) {
  if (*orderIn != rhs.getOrderOut())
    FormatAndThrow<LaguerreError>()  << "Order mismatch between LTransform [" << *orderIn
				     << "] and LTransform [" << rhs.getOrderOut()
				     << "]";
  (*m) = (*m) * (*rhs.m);
  *orderIn = *(rhs.orderOut);
  return *this;
}

//----------------------------------------------------------------
//----------------------------------------------------------------
// Calculate Laguerre polynomials and wavefunctions:

// Fill LVector with the basis functions corresponding to each real DOF
void 
LVector::fillBasis(double xunit, double yunit, 
		   double sigma) {
  // fill with psi_pq(z), where psi now defined to have 1/sigma^2 in
  // front.
  DComplex z(xunit,-yunit);
  double x = norm(z);

  double tq = exp(-0.5*x) / (2*PI*sigma*sigma);
  double tqm1=tq;
  double tqm2;

  // Ascend m=0 first

  (*v)[PQIndex(0,0).rIndex()]=tq;

  if (*order>=2) {
    tq = (x-1)*tqm1;
    (*v)[PQIndex(1,1).rIndex()] = tq;
  }

  PQIndex pq(2,2);
  for (int p=2; 2*p<= *order; ++p, pq.incN()) {
    tqm2 = tqm1;
    tqm1 = tq;
    tq = ((x-2*p+1)*tqm1 - (p-1)*tqm2)/p;
    (*v)[pq.rIndex()] = tq;
  }

  // Ascend all positive m's
  DComplex zm = 2* (*v)[PQIndex(0,0).rIndex()] * z;

  for (int m=1; m<= *order; m++) {
    pq.setPQ(m,0);
    double *r = &(*v)[pq.rIndex()];
    *r = zm.real();
    *(r+1) = zm.imag();
    tq = 1.;
    tqm1 = 0.;

    for (pq.incN(); !pq.pastOrder(*order); pq.incN()) {
      tqm2 = tqm1;
      tqm1 = tq;
      int p=pq.getP(); int q=pq.getQ(); 
      tq = ( (x-(p+q-1))*tqm1 - sqrtn(p-1)*sqrtn(q-1)*tqm2) / (sqrtn(p)*sqrtn(q));
      double *r = &(*v)[pq.rIndex()];
      *r = tq*zm.real();
      *(r+1) = tq*zm.imag();
    }

    zm *= z/sqrtn(m+1);
  }
}

DMatrix* 
LVector::basis(const DVector& xunit, const DVector& yunit,
	       int order_, double sigma) {
  Assert(xunit.size()==yunit.size());
  //**/Stopwatch s; s.start();
  DMatrix* mr = new DMatrix(xunit.size(), PQIndex::size(order_));
  //**/s.stop(); cerr << "setup: " << s << "sec" << endl;
  basis(*mr, xunit, yunit, order_, sigma);
  return mr;
}
void
LVector::basis(DMatrix& out,
	       const DVector& xunit, const DVector& yunit,
	       int order_, double sigma) {
  const int npts=xunit.size();
  Assert(yunit.size() ==npts && out.nrows()==npts);
  Assert(out.ncols()==PQIndex::size(order_));
  for (int ilo=0; ilo<npts; ilo+=BLOCKING_FACTOR) {
    int ihi = MIN(npts, ilo + BLOCKING_FACTOR);
    mBasis(xunit.subVector(ilo,ihi), yunit.subVector(ilo,ihi), 0,
	   &(out.rowRange(ilo,ihi)), 0, 
	   order_, false, sigma);
  }
}

DMatrix* 
LVector::design(const DVector& xunit, const DVector& yunit,
		const DVector& invsig,
	       int order_, double sigma) {
  //**/Stopwatch s; s.start();
  DMatrix* mr = new DMatrix(xunit.size(), PQIndex::size(order_),0.);
  //**/s.stop(); cerr << "setup: " << s << "sec" << endl;
  design(*mr, xunit, yunit, invsig, order_, sigma);
  return mr;
}

void
LVector::design(DMatrix& out,
		const DVector& xunit, const DVector& yunit,
		const DVector& invsig,
		int order_, double sigma) {
  const int npts=xunit.size();
  Assert(yunit.size()==npts && out.nrows()==npts && invsig.size()==npts);
  Assert(out.ncols()==PQIndex::size(order_));
  for (int ilo=0; ilo<npts; ilo+=BLOCKING_FACTOR) {
    int ihi = MIN(npts, ilo + BLOCKING_FACTOR);
    mBasis(xunit.subVector(ilo,ihi), yunit.subVector(ilo,ihi), 
	   &(invsig.subVector(ilo,ihi)),
	   &(out.rowRange(ilo,ihi)), 0, 
	   order_, false, sigma);
  }
}

void 
LVector::kBasis(const DVector& kxunit, const DVector& kyunit,
		DMatrix* &kReal, DMatrix* &kImag,
		int order_) {
  const int ndof=PQIndex::size(order_);
  const int npts = kxunit.size();
  Assert (kyunit.size()==npts);
  if (!kReal || kReal->nrows()!=npts || kReal->ncols()!=ndof) {
    if (kReal) delete kReal;
    kReal = new DMatrix(npts, ndof, 0.);
  } else {
    kReal->setZero();
  }
  if (!kImag || kImag->nrows()!=npts || kImag->ncols()!=ndof) {
    if (kImag) delete kImag;
    kImag = new DMatrix(npts, ndof, 0.);
  } else {
    kImag->setZero();
  }
  for (int ilo=0; ilo<npts; ilo+=BLOCKING_FACTOR) {
    int ihi = MIN(npts, ilo + BLOCKING_FACTOR);
    mBasis(kxunit.subVector(ilo,ihi), kyunit.subVector(ilo,ihi), 0,
	   &(kReal->rowRange(ilo,ihi)), &(kImag->rowRange(ilo,ihi)), 
	   order_, true, 1.);
  }
}

#ifdef PTRLOOPS
void
vvmult(double* p1, const double* p2, const double* p3, const int npts) {
  for (int i=0; i<npts; i++, p1++, p2++, p3++)
    *p1 = *p2 * *p3;
}

void
vsvs(double* p1, const double* p2, double f1, double f2, const int npts) {
  for (int i=0; i<npts; i++, p1++, p2++)
    *p1 = (*p2 + f1)*f2;
}

void
qjump(double* qp, const double* qm1p, const double* qm2p, const double* r2p,
      double f1, double f2, double f3, const int npts) {
  double* ptmp=qp;
  for (int i=0; i<npts; i++, ++ptmp, ++qm1p, ++r2p)
    *ptmp = (*r2p + f1) * *qm1p;
  for (int i=0; i<npts; i++, ++qp, ++qm2p)
    *qp = f2 * *qp  + f3 * *qm2p;
}
#endif

/***
#ifdef PTRLOOPS
void
vvmult(double& v1, double& v2, double& v3, const int npts) {
  double* p1 = &v1; double* p2=&v2; double* p3=&v3;
  for (int i=0; i<npts; i++, p1++, p2++, p3++)
    *p1 = *p2 * *p3;
}

void
vsvs(double& v1, double& v2, double f1, double f2, const int npts) {
  double* p1 = &v1; double* p2=&v2; 
  for (int i=0; i<npts; i++, p1++, p2++)
    *p1 = (*p2 + f1)*f2;
}

void
qjump(double& q, double& qm1, double& qm2, double& rsq,
      double f1, double f2, double f3, const int npts) {
  double* qp = &q;
  double* qm1p = &qm1;
  double* qm2p = &qm2;
  double* r2p = &rsq;
  
  for (int i=0; i<npts; i++, ++qp, ++qm1p, ++r2p)
    *qp = (*r2p + f1) * *qm1p;
  qp = &q;
  for (int i=0; i<npts; i++, ++qp, ++qm2p)
    *qp = f2 * *qp  + f3 * *qm2p;
}
#endif

**/


void 
LVector::mBasis(const tmv::ConstVectorView<double>& x, 
		const tmv::ConstVectorView<double>& y,
		const tmv::ConstVectorView<double>* invsig,
		const tmv::MatrixView<double>* mr,
		const tmv::MatrixView<double>* mi,
		int order_, bool isK, double sigma) {
  const int N=order_;
  const int ndof=PQIndex::size(N);
  const int npts = x.size();
  Assert (y.size()==npts);
  Assert (mr->nrows()==npts && mr->ncols()==ndof);
  if (isK) Assert (mi->nrows()==npts && mi->ncols()==ndof);

  // Cast arguments as diagonal matrices so we can access
  // vectorized element-by-element multiplication
  tmv::DiagMatrix<double> X(x);
  tmv::DiagMatrix<double> Y(y);
#ifdef PTRLOOPS
  DVector Rsq(npts);
#else
  tmv::DiagMatrix<double> Rsq(npts);
  //**/Stopwatch s; s.start(); cerr << "Start..."<<endl;
#endif
  // These vectors will keep track of real & imag parts
  // of prefactor * exp(-r^2/2) (x+iy)^m / sqrt(m!)
  DVector cosm(npts, isK ? 1. : 1./(2*PI*sigma*sigma));
  DVector sinm(npts, 0.);
  for (int i=0; i<Rsq.size(); i++) {
    Rsq(i) = x[i]*x[i]+y[i]*y[i];
    cosm[i] *= exp(-0.5*Rsq(i));
  }
  // Put 1/sigma factor into every point if doing a design matrix:
  if (invsig) 
    cosm *= tmv::DiagMatrixViewOf(*invsig);
  
  // Assign the m=0 column first:
  mr->col( PQIndex(0,0).rIndex() ) = cosm;

  //**/s.stop(); cerr << "For pq=0: " << s << " sec" << endl;   s.reset(); s.start();
  // Then ascend m's at q=0:
  for (int m=1; m<=N; m++) {
    int rIndex = PQIndex(m,0).rIndex();
    // Multiply by (X+iY)/sqrt(m), including a factor 2 first time through
    DVector ctmp = X*cosm - Y*sinm;
    DVector stmp = X*sinm + Y*cosm;
    cosm = ctmp * (( m==1 ? 2. : 1.) /sqrtn(m));
    sinm = stmp * (( m==1 ? 2. : 1.) /sqrtn(m));

    if (isK) {
      switch (m%4) {
      case 0:
	mr->col(rIndex) = cosm; 
	mr->col(rIndex+1) = -sinm;
	break;
      case 1:
	mi->col(rIndex) = -cosm; 
	mi->col(rIndex+1) = sinm;
	break;
      case 2:
	mr->col(rIndex) = -cosm; 
	mr->col(rIndex+1) = sinm;
	break;
      case 3:
	mi->col(rIndex) = cosm; 
	mi->col(rIndex+1) = -sinm; 
	break;
      }
    } else {
      // Real-space
      mr->col(rIndex) = cosm;
      mr->col(rIndex+1) = -sinm;
    }
  }

  //**/s.stop(); cerr << "For q=0: " << s << " sec" << endl;   s.reset(); s.start();

  // Now climb q's at each m
  /**
#ifdef PTRLOOPS
  // Compile these to do arithmetic with pointer loops (TMV DiagMatrix method below)
  DVector* Lmq = new DVector(npts);
  DVector* Lmqm1 = new DVector(npts);
  DVector* Lmqm2 = new DVector(npts);
  for (int m=0; m<=N; m++) {
    PQIndex pq(m,0);
    int iQ0 = pq.rIndex();
    // Go to q=1:
    pq.incN();
    if (pq.pastOrder(N)) continue;
    int iQ = pq.rIndex();
    Lmqm1->SetAllTo(1.);
    const double f1= - (pq.getP()+pq.getQ()-1.);
    const double f2 = (isK ? -1. : 1.) / sqrtn(pq.getP())/sqrtn(pq.getQ());
    vsvs( (*Lmq)[0], Rsq[0], f1, f2, npts);
    if (isK) {
      if (m%2==0) {
	// even m's have real transforms
	//	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	vvmult(mr->col(iQ)[0], (*Lmq)[0], mr->col(iQ0)[0], npts);
	if (m>0) 
	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	  vvmult(mr->col(iQ+1)[0], (*Lmq)[0], mr->col(iQ0+1)[0], npts);
      } else {
	// odd m's have imag transforms
	//	mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	vvmult(mi->col(iQ)[0], (*Lmq)[0], mi->col(iQ0)[0], npts);
	//	mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
	vvmult(mi->col(iQ+1)[0], (*Lmq)[0], mi->col(iQ0+1)[0], npts);
      }
    } else {
      //  mr->col(iQ) = (*Lmq) * mr->col(iQ0);
      vvmult(mr->col(iQ)[0], (*Lmq)[0], mr->col(iQ0)[0], npts);
      if (m>0) 	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	  vvmult(mr->col(iQ+1)[0], (*Lmq)[0], mr->col(iQ0+1)[0], npts);
    }

    // do q=2,...
    for (pq.incN(); !pq.pastOrder(N); pq.incN()) {
      {
	// cycle the Lmq vectors
	DVector* tmp = Lmqm2;
	Lmqm2 = Lmqm1;
	Lmqm1 = Lmq;
	Lmq = tmp;
      }
      {
	double f1 = - (pq.getP()+pq.getQ()-1.);
	double invsqrtpq = 1./sqrtn(pq.getP())/sqrtn(pq.getQ());
	double f2 = isK ? -invsqrtpq : invsqrtpq;
	double f3 = -sqrtn(pq.getP()-1)*sqrtn(pq.getQ()-1) * invsqrtpq;
	qjump((*Lmq)[0], (*Lmqm1)[0],(*Lmqm2)[0], Rsq[0],
	      f1, f2, f3, npts);
      }
      iQ = pq.rIndex();
      if (isK) {
	if (m%2==0) {
	  //	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	  vvmult(mr->col(iQ)[0], (*Lmq)[0], mr->col(iQ0)[0], npts);
	  if (m>0) 
	    //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	    vvmult(mr->col(iQ+1)[0], (*Lmq)[0], mr->col(iQ0+1)[0], npts);
	} else {
	  // odd m's have imag transforms
	  //	mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	  vvmult(mi->col(iQ)[0], (*Lmq)[0], mi->col(iQ0)[0], npts);
	  //	mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
	  vvmult(mi->col(iQ+1)[0], (*Lmq)[0], mi->col(iQ0+1)[0], npts);
	}
      } else {
	//	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	vvmult(mr->col(iQ)[0], (*Lmq)[0], mr->col(iQ0)[0], npts);
	if (m>0) 	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	  vvmult(mr->col(iQ+1)[0], (*Lmq)[0], mr->col(iQ0+1)[0], npts);
      }
    }
  }
  //**s.stop(); cerr << "For q>0: " << s << " sec" << endl;   s.reset(); s.start();
**/
#ifdef PTRLOOPS
  // Compile these to do arithmetic with pointer loops (TMV DiagMatrix method below)
  DVector* Lmq = new DVector(npts);
  DVector* Lmqm1 = new DVector(npts);
  DVector* Lmqm2 = new DVector(npts);
  for (int m=0; m<=N; m++) {
    PQIndex pq(m,0);
    int iQ0 = pq.rIndex();
    // Go to q=1:
    pq.incN();
    if (pq.pastOrder(N)) continue;
    int iQ = pq.rIndex();
    Lmqm1->setAllTo(1.);
    const double f1= - (pq.getP()+pq.getQ()-1.);
    const double f2 = (isK ? -1. : 1.) / sqrtn(pq.getP())/sqrtn(pq.getQ());
    vsvs( Lmq->ptr(), Rsq.cptr(), f1, f2, npts);
    if (isK) {
      if (m%2==0) {
	// even m's have real transforms
	//	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	vvmult(mr->col(iQ).ptr(), Lmq->cptr(), mr->col(iQ0).ptr(), npts);
	if (m>0) 
	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	  vvmult(mr->col(iQ+1).ptr(), Lmq->cptr(), mr->col(iQ0+1).ptr(), npts);
      } else {
	// odd m's have imag transforms
	//	mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	vvmult(mi->col(iQ).ptr(), Lmq->cptr(), mi->col(iQ0).cptr(), npts);
	//	mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
	vvmult(mi->col(iQ+1).ptr(), Lmq->cptr(), mi->col(iQ0+1).cptr(), npts);
      }
    } else {
      //  mr->col(iQ) = (*Lmq) * mr->col(iQ0);
      vvmult(mr->col(iQ).ptr(), Lmq->cptr(), mr->col(iQ0).cptr(), npts);
      if (m>0) 	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	vvmult(mr->col(iQ+1).ptr(), Lmq->cptr(), mr->col(iQ0+1).cptr(), npts);
    }

    // do q=2,...
    for (pq.incN(); !pq.pastOrder(N); pq.incN()) {
      {
	// cycle the Lmq vectors
	DVector* tmp = Lmqm2;
	Lmqm2 = Lmqm1;
	Lmqm1 = Lmq;
	Lmq = tmp;
      }
      {
	double f1 = - (pq.getP()+pq.getQ()-1.);
	double invsqrtpq = 1./sqrtn(pq.getP())/sqrtn(pq.getQ());
	double f2 = isK ? -invsqrtpq : invsqrtpq;
	double f3 = -sqrtn(pq.getP()-1)*sqrtn(pq.getQ()-1) * invsqrtpq;
	qjump(Lmq->ptr(), Lmqm1->cptr(),Lmqm2->cptr(), Rsq.cptr(),
	      f1, f2, f3, npts);
      }
      iQ = pq.rIndex();
      if (isK) {
	if (m%2==0) {
	  //	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	  vvmult(mr->col(iQ).ptr(), Lmq->cptr(), mr->col(iQ0).cptr(), npts);
	  if (m>0) 
	    //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	    vvmult(mr->col(iQ+1).ptr(), Lmq->cptr(), mr->col(iQ0+1).cptr(), npts);
	} else {
	  // odd m's have imag transforms
	  //	mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	  vvmult(mi->col(iQ).ptr(), Lmq->cptr(), mi->col(iQ0).cptr(), npts);
	  //	mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
	  vvmult(mi->col(iQ+1).ptr(), Lmq->cptr(), mi->col(iQ0+1).cptr(), npts);
	}
      } else {
	//	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	vvmult(mr->col(iQ).ptr(), Lmq->cptr(), mr->col(iQ0).cptr(), npts);
	if (m>0) 	  //mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	  vvmult(mr->col(iQ+1).ptr(), Lmq->cptr(), mr->col(iQ0+1).cptr(), npts);
      }
    }
  }
#else

  // Make three DiagMatrix to hold Lmq's during recurrence calculations
  tmv::DiagMatrix<double>* Lmq = new tmv::DiagMatrix<double>(npts);
  tmv::DiagMatrix<double>* Lmqm1 = new tmv::DiagMatrix<double>(npts);
  tmv::DiagMatrix<double>* Lmqm2 = new tmv::DiagMatrix<double>(npts);
  for (int m=0; m<=N; m++) {
    PQIndex pq(m,0);
    int iQ0 = pq.rIndex();
    // Go to q=1:
    pq.incN();
    if (pq.pastOrder(N)) continue;
    int iQ = pq.rIndex();
    Lmqm1->setAllTo(1.);
    *Lmq = Rsq - (pq.getP()+pq.getQ()-1.);
    *Lmq *= (isK ? -1. : 1.) / sqrtn(pq.getP())/sqrtn(pq.getQ());
    if (isK) {
      if (m%2==0) {
	// even m's have real transforms
	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	if (m>0) mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
      } else {
	// odd m's have imag transforms
	mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
      }
    } else {
      mr->col(iQ) = (*Lmq) * mr->col(iQ0);
      if (m>0) mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
    }

    // do q=2,...
    for (pq.incN(); !pq.pastOrder(N); pq.incN()) {
      {
	// cycle the Lmq vectors
	tmv::DiagMatrix<double>* tmp = Lmqm2;
	Lmqm2 = Lmqm1;
	Lmqm1 = Lmq;
	Lmq = tmp;
      }
      double invsqrtpq = 1./sqrtn(pq.getP())/sqrtn(pq.getQ());
      *Lmq = (Rsq - (pq.getP()+pq.getQ()-1.)) * (*Lmqm1) * (isK ? -invsqrtpq : invsqrtpq);
      *Lmq -= (sqrtn(pq.getP()-1)*sqrtn(pq.getQ()-1)*invsqrtpq) * (*Lmqm2);

      iQ = pq.rIndex();
      if (isK) {
	if (m%2==0) {
	  // even m's have real transforms
	  mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	  if (m>0) mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
	} else {
	  // odd m's have imag transforms
	  mi->col(iQ) = (*Lmq) * mi->col(iQ0);
	  mi->col(iQ+1) = (*Lmq) * mi->col(iQ0+1);
	}
      } else {
	mr->col(iQ) = (*Lmq) * mr->col(iQ0);
	if (m>0) mr->col(iQ+1) = (*Lmq) * mr->col(iQ0+1);
      }
    }
  }
#endif  // PTRLOOPS
  delete Lmq; delete Lmqm1; delete Lmqm2;
}


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// Flux determinations
double 
LVector::flux(int maxP) const {
  if (maxP<0) maxP = getOrder()/2;
  if (maxP > getOrder()/2) maxP=getOrder()/2;
  double retval=0.;
  for (int p=0; p<=maxP; p++)
    retval += (*v)[PQIndex(p,p).rIndex()];
  return retval;
}
double 
LVector::apertureFlux(double R_, int maxP) const {
  static DVector *fp=0;
  static double R=-1.;
  static double psize=-1;

  Assert(R_>=0.);

  if (maxP<0) maxP= getOrder()/2;
  if (maxP > getOrder()/2) maxP=getOrder()/2;

  if (!fp || R_ != R || maxP>psize) {
    if (!fp) {
      fp = new DVector(maxP);
      psize = maxP;
    } else if (maxP != psize) {
      delete fp; 
      fp = new DVector(maxP);
      psize = maxP;
    }
    R = R_;
    DVector Lp(maxP+1);
    DVector Qp(maxP+1);
    double x = R*R;
    double efact = exp(-0.5*x);
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
    flux += (*v)[PQIndex(p,p).rIndex()] * (*fp)[p];
  return flux;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// I/O Routines

ostream& laguerre::operator<<(ostream& os, const laguerre::LVector& lv) {
  lv.write(os); return os;
}

void laguerre::LVector::write(ostream& os, int maxorder) const
{
  int oldprec = os.precision(8);
  std::ios::fmtflags oldf = os.setf(std::ios::scientific,std::ios::floatfield);
  if (maxorder < 0 || maxorder > *order)
    maxorder = *order;
  os << *order << endl;
  for (int n=0; n<=maxorder; n++) 
    for(PQIndex pq(n,0); !pq.needsConjugation(); pq.decm()) {
      os << " " << std::setw(2) << pq.getP() 
	 << " " << std::setw(2) << pq.getQ() ;
      if (pq.isReal()) 
	os << " " << std::setw(15) << (*this)[pq].real() << endl;
      else
	os << " " << std::setw(15) << (*this)[pq].real() 
	   << " " << std::setw(15) << (*this)[pq].imag() << endl;
    }
  os.precision(oldprec);
  os.flags(oldf);
}

istream& laguerre::operator>>(istream& is, laguerre::LVector& lv) {
  lv.read(is); return is;
}

void laguerre::LVector::read(istream& is)
{
  // get order
  int order;
  is >> order;
  resize(order);
  // discard p,q info, read into rVector
  int p, q; 
  double re, im;
  for (int n=0; n<=order; n++) 
    for(PQIndex pq(n,0); !pq.needsConjugation(); pq.decm()) {
      is >> p >> q;
      if (pq.isReal()) {
	is >> re;  im = 0.;
	(*this)[pq]=DComplex(re,im);
      } else {
	is >> re >> im;
	(*this)[pq]=DComplex(re,im);
      }
    }
}

void laguerre::PQIndex::write(ostream& os) const
{
  os << std::setw(2) << getP() 
     << "," << std::setw(2) << getQ() ;
}

// Transformation generators - these return a view into static quantities:
const 
tmv::ConstMatrixView<double>
LVector::Generator(GType iparam, int orderOut, int orderIn) {
  static DMatrix *gmu;
  static DMatrix *gx;
  static DMatrix *gy;
  static DMatrix *ge1;
  static DMatrix *ge2;
  static DMatrix *grot;

  const int sizeIn = PQIndex::size(orderIn);
  const int sizeOut = PQIndex::size(orderOut);

  const int order = MAX(orderOut, orderIn);
  if (iparam==iMu) {
    if (!gmu || gmu->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);

      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int p=pq.getP();
	int q=pq.getQ();
	DComplex zz(-1.,0.);
	if (pq.isReal())
	  lt.set(pq,pq,zz, zz);
	else
	  lt.set(pq,pq,zz, 0.);
	PQIndex pqprime(p+1, q+1);
	if (!pqprime.pastOrder(order)) {
	    zz = DComplex(-sqrtn(p+1)*sqrtn(q+1), 0.);
	    if (pq.isReal())
	      lt.set(pq,pqprime,zz, zz);
	    else
	      lt.set(pq,pqprime,zz, 0.);
	}
	if (q>0) {
	  pqprime.setPQ(p-1,q-1);
	  zz = DComplex(sqrtn(p)*sqrtn(q), 0.);
	  if (pq.isReal())
	    lt.set(pq,pqprime,zz, zz);
	  else
	    lt.set(pq,pqprime,zz, 0.);
	}
      }
      if (gmu) delete gmu;
      gmu = new DMatrix(lt.rMatrix());
    }
    return gmu->subMatrix(0, sizeOut, 0, sizeIn);
  }
  if (iparam==iX) {
    if (!gx || gx->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);

      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int p=pq.getP();
	int q=pq.getQ();
	PQIndex pqprime(p+1, q);
	DComplex zz(-0.5*sqrtn(p+1),0.);
	if (pq.isReal()) {
	  if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
	  if (p>0) {
	    zz = DComplex(0.5*sqrtn(p), 0.);
	    pqprime.setPQ(p-1,q);
	    lt.set(pq,pqprime,zz, zz);
	  }
	} else {
	  if (!pqprime.pastOrder(order)) {	  
	    lt.set(pq,pqprime,zz, 0.);
	    pqprime.setPQ(p, q+1);
	    zz = DComplex(-0.5*sqrtn(q+1),0.);
	    if (pq.m()==1) {
	      lt.set(pq,pqprime, zz, zz);
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	  pqprime.setPQ(p-1,q);
	  zz = DComplex(0.5*sqrtn(p), 0.);
	  if (pq.m()==1) {
	    lt.set(pq,pqprime, zz, zz);
	  } else {
	    lt.set(pq,pqprime, zz, 0.);
	  }
	  if (q>0) {
	    pqprime.setPQ(p,q-1);
	    zz = DComplex(0.5*sqrtn(q), 0.);
	    lt.set(pq,pqprime, zz, 0.);
	  }
	}
      }
      if (gx) delete gx;
      gx = new DMatrix(lt.rMatrix());
    }
    return gx->subMatrix(0, sizeOut, 0, sizeIn);
  }

  if (iparam==iY) {
    if (!gy || gy->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);

      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int p=pq.getP();
	int q=pq.getQ();
	PQIndex pqprime(p+1, q);
	DComplex zz(0.,-0.5*sqrtn(p+1));
	if (pq.isReal()) {
	  if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
	  if (p>0) {
	    zz = DComplex(0.,0.5*sqrtn(q));
	    pqprime.setPQ(p,q-1);
	    lt.set(pq,pqprime,zz, zz);
	  }
	} else {
	  if (!pqprime.pastOrder(order)) {
	    lt.set(pq,pqprime,zz, 0.);
	    pqprime.setPQ(p, q+1);
	    zz = DComplex(0.,0.5*sqrtn(q+1));
	    if (pq.m()==1) {
	      lt.set(pq,pqprime, zz, conj(zz));
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	  pqprime.setPQ(p-1,q);
	  zz = DComplex(0.,-0.5*sqrtn(p));
	  if (pq.m()==1) {
	    lt.set(pq,pqprime, zz, conj(zz));
	  } else {
	    lt.set(pq,pqprime, zz, 0.);
	  }
	  if (q>0) {
	    pqprime.setPQ(p,q-1);
	    zz = DComplex(0.,0.5*sqrtn(q));
	    lt.set(pq,pqprime, zz, 0.);
	  }
	}
      }
      if (gy) delete gy;
      gy = new DMatrix(lt.rMatrix());
    }
    return gy->subMatrix(0, sizeOut, 0, sizeIn);
  }

  if (iparam==iE1) {
    if (!ge1 || ge1->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);

      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int p=pq.getP();
	int q=pq.getQ();
	PQIndex pqprime(p+2, q);
	DComplex zz(-0.25*sqrtn(p+1)*sqrtn(p+2),0.);
	if (pq.isReal()) {
	  if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
	  if (p>1) {
	    zz = DComplex(0.25*sqrtn(p)*sqrtn(p-1),0.);
	    pqprime.setPQ(p-2,q);
	    lt.set(pq,pqprime,zz, zz);
	  }
	} else {
	  if (!pqprime.pastOrder(order)) {
	    lt.set(pq,pqprime,zz, 0.);
	    pqprime.setPQ(p, q+2);
	    zz = DComplex(-0.25*sqrtn(q+1)*sqrtn(q+2),0.);
	    if (pq.m()==2) {
	      lt.set(pq,pqprime, zz, zz);
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	  if (p>1) {
	    pqprime.setPQ(p-2,q);
	    zz = DComplex(0.25*sqrtn(p)*sqrtn(p-1),0.);
	    if (pq.m()==2) {
	      lt.set(pq,pqprime, zz, zz);
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	    if (q>1) {
	      pqprime.setPQ(p,q-2);
	      zz = DComplex(0.25*sqrtn(q)*sqrtn(q-1),0.);
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	}
      }
      if (ge1) delete ge1;
      ge1 = new DMatrix(lt.rMatrix());
    }
    return ge1->subMatrix(0, sizeOut, 0, sizeIn);
  }

  if (iparam==iE2) {
    if (!ge2 || ge2->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);

      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int p=pq.getP();
	int q=pq.getQ();
	PQIndex pqprime(p+2, q);
	DComplex zz(0., -0.25*sqrtn(p+1)*sqrtn(p+2));
	if (pq.isReal()) {
	  if (!pqprime.pastOrder(order)) lt.set(pq,pqprime,zz, zz);
	  if (p>1) {
	    zz = DComplex(0.,-0.25*sqrtn(p)*sqrtn(p-1));
	    pqprime.setPQ(p-2,q);
	    lt.set(pq,pqprime,zz, zz);
	  }
	} else {
	  if (!pqprime.pastOrder(order)) {
	    lt.set(pq,pqprime,zz, 0.);
	    pqprime.setPQ(p, q+2);
	    zz = DComplex(0.,0.25*sqrtn(q+1)*sqrtn(q+2));
	    if (pq.m()==2) {
	      lt.set(pq,pqprime, zz, conj(zz));
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	  if (p>1) {
	    pqprime.setPQ(p-2,q);
	    zz = DComplex(0.,-0.25*sqrtn(p)*sqrtn(p-1));
	    if (pq.m()==2) {
	      lt.set(pq,pqprime, zz, conj(zz));
	    } else {
	      lt.set(pq,pqprime, zz, 0.);
	    }
	    if (q>1) {
	      pqprime.setPQ(p,q-2);
	      zz = DComplex(0.,0.25*sqrtn(q)*sqrtn(q-1));
	      lt.set(pq,pqprime, zz, 0.);
	    }
	  }
	}
      }
      if (ge2) delete ge2;
      ge2 = new DMatrix(lt.rMatrix());
    }
    return ge2->subMatrix(0, sizeOut, 0, sizeIn);
  }

  if (iparam==iRot) {
    // Rotation is diagonal - could use a DiagMatrix perhaps
    if (!grot || grot->nrows()<PQIndex::size(order)) {
      LTransform lt(order, order);
      for (PQIndex pq(0,0); !pq.pastOrder(order); pq.nextDistinct()) {
	int m = pq.m();
	if (m>0) lt.set(pq,pq, DComplex(0.,-m), 0.);
      }
      if (grot) delete grot;
      grot = new DMatrix(lt.rMatrix());
    }
    return grot->subMatrix(0, sizeOut, 0, sizeIn);
  }

  else
    throw (LaguerreError("Unknown parameter for LVector::Generator()"));
}

// Function to solve for radius enclosing a specified flux.
// Return negative radius if no root is apparent.
class FRSolve {
public:
  FRSolve(const LVector& lv_, double thresh_, int maxP_): 
    lv(lv_), maxP(maxP_), thresh(thresh_) {
    Assert(lv.getOrder() >= 2*maxP);
  }
  double operator()(double u) const {return lv.apertureFlux(u,maxP)-thresh;}
private:
  const LVector& lv;
  double thresh;
  int maxP;
};

double
laguerre::fluxRadius(const LVector& lv, double threshold, int maxP) {
  if (maxP<0) maxP= lv.getOrder()/2;
  if (maxP > lv.getOrder()/2) maxP=lv.getOrder()/2;
  FRSolve func(lv, threshold, maxP);

  // First we step through manually at intervals roughly the smallest that 
  // a function of this order can oscillate, in order to bracket the root
  // closest to the origin.

  const double TOLERANCE=0.001;	//radius accuracy required
  const double maxR = 5.;
  double ustep=0.5/sqrt(static_cast<double> (maxP+1));
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
  solve::Solve<FRSolve> s(func, u1, u2);
  s.setXTolerance(TOLERANCE);
  return s.root();
}
