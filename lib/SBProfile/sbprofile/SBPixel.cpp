// $Id: SBPixel.cpp,v 1.8 2012/02/17 01:59:08 garyb Exp $

#include "SBPixel.h"

using namespace sbp;

const double TPI = 2.*PI;
const double OVERSAMPLE_X = 4.;  // FT must be at least this much larger than input

// Default k-space interpolant is quintic:
Quintic
defaultKInterpolant1d(1e-4);

InterpolantXY
SBPixel::defaultKInterpolant2d(defaultKInterpolant1d);

SBPixel::SBPixel(int Npix, double dx_, const Interpolant2d& i, int Nimages_): 
  Ninitial(Npix+Npix%2), dx(dx_), Nimages(Nimages_),
  xInterp(&i), kInterp(&defaultKInterpolant2d),
  wts(Nimages, 1.), fluxes(Nimages, 1.), 
  xFluxes(Nimages, 0.), yFluxes(Nimages,0.),
  xsum(0), ksum(0), xsumValid(false),ksumValid(false),
  ready(false) 
{
  Assert(Ninitial%2==0);
  Assert(Ninitial>=2);
  // Choose the padded size for input array - size 2^N or 3*2^N
  // Make FFT either 2^n or 3x2^n
  Nk = fft::goodFFTSize(OVERSAMPLE_X*Ninitial);
  dk = TPI / (Nk*dx);

  // allocate xTables
  for (int i=0; i<Nimages; i++) 
    vx.push_back(new XTable(Nk, dx));
}

#ifdef USE_IMAGES
SBPixel::SBPixel(Image<> img, const Interpolant2d& i, double dx_, double padFactor): 
  dx(dx_), Nimages(1),
  xInterp(&i), kInterp(&defaultKInterpolant2d),
  wts(Nimages, 1.), fluxes(Nimages, 1.), 
  xFluxes(Nimages, 0.), yFluxes(Nimages,0.),
  xsum(0), ksum(0), xsumValid(false),ksumValid(false),
  ready(false) 
{
  Ninitial = MAX( img.YMax()-img.YMin()+1, img.XMax()-img.XMin()+1);
  Ninitial = Ninitial + Ninitial%2;
  Assert(Ninitial%2==0);
  Assert(Ninitial>=2);
  if (dx<=0.) {
  // If dx was not specified, see if the header has a value, if not just dx=1.
    if (!img.header()->getValue("DX", dx))
      dx = 1.;
  }
  if (padFactor <= 0.) padFactor = OVERSAMPLE_X;
  // Choose the padded size for input array - size 2^N or 3*2^N
  // Make FFT either 2^n or 3x2^n
  Nk = fft::goodFFTSize(static_cast<int> (floor(padFactor*Ninitial)));
  dk = TPI / (Nk*dx);

  // allocate xTables
  for (int i=0; i<Nimages; i++) 
    vx.push_back(new XTable(Nk, dx));
  // fill data from image, shifting to center the image in the table
  int xStart = -((img.XMax()-img.XMin()+1)/2);
  int yTab = -((img.YMax()-img.YMin()+1)/2);
  for (int iy = img.YMin(); iy<= img.YMax(); iy++, yTab++) {
    int xTab = xStart;
    for (int ix = img.XMin(); ix<= img.XMax(); ix++, xTab++) 
      vx.front()->xSet(xTab, yTab, img(ix,iy));
  }
}
#endif

SBPixel::SBPixel(const SBPixel& rhs):
  Ninitial(rhs.Ninitial), dx(rhs.dx), Nk(rhs.Nk), Nimages(rhs.Nimages),
  xInterp(rhs.xInterp), kInterp(rhs.kInterp),
  wts(rhs.wts), fluxes(rhs.fluxes), 
  xFluxes(rhs.xFluxes), yFluxes(rhs.yFluxes),
  xsum(0), ksum(0), xsumValid(false),ksumValid(false),
  ready(rhs.ready) 
{
  // copy tables
  for (int i=0; i<Nimages; i++) {
    vx.push_back(new XTable(*rhs.vx[i]));
    if (ready)     vk.push_back(new KTable(*rhs.vk[i]));
  }
}

SBPixel::~SBPixel() {
  for (int i=0; i<vx.size(); i++)
    if (vx[i]) {delete vx[i]; vx[i]=0;}
  for (int i=0; i<vk.size(); i++)
    if (vk[i]) {delete vk[i]; vk[i]=0;}
  if (xsum) {delete xsum; xsum=0;}
  if (ksum) {delete ksum; ksum=0;}
}

double 
SBPixel::getFlux() const {
  checkReady();
  return wts * fluxes;
}

void
SBPixel::setFlux(double flux) {
  checkReady();
  double factor = flux/getFlux();
  wts *= factor;
  if (xsumValid) *xsum *= factor;
  if (ksumValid) *ksum *= factor;
}

double 
SBPixel::centroidX() const {
  checkReady();
  return (wts * xFluxes) / (wts*fluxes);
}
double 
SBPixel::centroidY() const {
  checkReady();
  return (wts * yFluxes) / (wts*fluxes);
}

void
SBPixel::setPixel(double value, int ix, int iy, int iz) {
  if (iz < 0 || iz>=Nimages)
    FormatAndThrow<SBError>() << "SBPixel::setPixel image number " << iz << " out of bounds";
  if (ix < -Ninitial/2 || ix >= Ninitial / 2)
    FormatAndThrow<SBError>() << "SBPixel::setPixel x coordinate " << ix << " out of bounds";
  if (iy < -Ninitial/2 || iy >= Ninitial / 2)
    FormatAndThrow<SBError>() << "SBPixel::setPixel x coordinate " << iy << " out of bounds";

  ready = false;
  vx[iz]->xSet(ix, iy, value);
}

double
SBPixel::getPixel(int ix, int iy, int iz) const {
  if (iz < 0 || iz>=Nimages)
    FormatAndThrow<SBError>() << "SBPixel::getPixel image number " << iz << " out of bounds";
  
  return vx[iz]->xval(ix, iy);
}

void
SBPixel::setWeights(const DVector& wts_) {
  Assert(wts_.size()==Nimages);
  wts = wts_;
  xsumValid = false;
  ksumValid = false;
}

void
SBPixel::checkReady() const {
  if (ready) return;
  // Flush old kTables if any;
  for (int i=0; i<vk.size(); i++) { delete vk[i]; vk[i]=0;}
  vk.clear();

  for (int i=0; i<Nimages; i++) {
    // Get sums:
    double sum = 0.;
    double sumx = 0.;
    double sumy = 0.;
    for (int iy=-Ninitial/2; iy<Ninitial/2; iy++)
      for (int ix=-Ninitial/2; ix<Ninitial/2; ix++) {
	double value = vx[i]->xval(ix, iy);
	sum += value;
	sumx += value*ix;
	sumy += value*iy;
      }
    fluxes[i] = sum*dx*dx;
    xFluxes[i] = sumx * pow(dx, 3.);
    yFluxes[i] = sumy * pow(dx, 3.);

    // Conduct FFT
    vk.push_back( vx[i]->transform());
  }
  ready = true;
  xsumValid = false;
  ksumValid = false;
  Assert(vk.size()==Nimages);
}

void
SBPixel::checkXsum() const {
  checkReady();
  if (xsumValid) return;
  if (!xsum) {
    xsum = new XTable(*vx[0]);
    *xsum *= wts[0];
  } else {
    xsum->clear();
    xsum->accumulate(*vx[0], wts[0]);
  }
  for (int i=1; i<Nimages; i++)
    xsum->accumulate(*vx[i], wts[i]);
  xsumValid = true;
}

void
SBPixel::checkKsum() const {
  checkReady();
  if (ksumValid) return;
  if (!ksum) {
    ksum = new KTable(*vk[0]);
    *ksum *= wts[0];
  } else {
    ksum->clear();
    ksum->accumulate(*vk[0], wts[0]);
  }
  for (int i=1; i<Nimages; i++)
    ksum->accumulate(*vk[i], wts[i]);
  ksumValid = true;
}

void
SBPixel::fillKGrid(KTable& kt) const {
  // This override of base class is to permit potential efficiency gain from
  // separable interpolant kernel.  If so, the fft::KTable interpolation routine
  // will go faster if we make y iteration the inner loop.
  if ( dynamic_cast<const InterpolantXY*> (kInterp)) {
    int N = kt.getN();
    double dk = kt.getDk();
    // Only need ix>=0 because it's Hermitian:
    for (int ix = 0; ix <= N/2; ix++) {
      for (int iy = -N/2; iy < N/2; iy++) {
	Position<double> k(ix*dk,iy*dk);
	kt.kSet(ix,iy,kValue(k));
      }
    }
  } else {
    // Otherwise just use the normal routine to fill the grid:
    SBProfile::fillKGrid(kt);
  }
}

// Same deal: reverse axis order if we have separable interpolant in X domain
void
SBPixel::fillXGrid(XTable& xt) const {
#ifdef DANIELS_TRACING
  cout << "SBPixel::fillXGrid called" << endl;
#endif
  if ( dynamic_cast<const InterpolantXY*> (xInterp)) {
    int N = xt.getN();
    double dx = xt.getDx();
    // Only need ix>=0 because it's Hermitian:
    for (int ix = 0; ix <= N/2; ix++) {
      for (int iy = -N/2; iy < N/2; iy++) {
	Position<double> x(ix*dx,iy*dx);
	xt.xSet(ix,iy,xValue(x));
      }
    }
  } else {
    // Otherwise just use the normal routine to fill the grid:
    SBProfile::fillXGrid(xt);
  }
}

#ifdef USE_IMAGES
// One more time: for images now
double
SBPixel::fillXImage(Image<> I, double dx) const {
#ifdef DANIELS_TRACING
  cout << "SBPixel::fillXImage called" << endl;
#endif
  double sum=0.;
  if ( dynamic_cast<const InterpolantXY*> (xInterp)) {
    for (int ix = I.XMin(); ix <= I.XMax(); ix++) {
      for (int iy = I.YMin(); iy <= I.YMax(); iy++) {
	Position<double> x(ix*dx,iy*dx);
	double val = xValue(x);
	sum += val;
	I(ix,iy) = (ix,iy,val);
      }
    }
  } else {
    // Otherwise just use the normal routine to fill the grid:
    return SBProfile::fillXImage(I,dx);
  }
}
#endif

#ifndef OLD_WAY
double
SBPixel::xValue(Position<double> p) const {
#ifdef DANIELS_TRACING
  cout << "getting xValue at " << p << endl;
#endif
  checkXsum();
  return xsum->interpolate(p.x, p.y, *xInterp);
}

DComplex
SBPixel::kValue(Position<double> p) const {
  // Don't bother if the desired k value is cut off by the x interpolant:
  double ux = p.x*dx/TPI;
  if (abs(ux) > xInterp->urange()) return DComplex(0.,0.);
  double uy = p.y*dx/TPI;
  if (abs(uy) > xInterp->urange()) return DComplex(0.,0.);
  double xKernelTransform = xInterp->uval(ux, uy);

  checkKsum();
  return xKernelTransform * ksum->interpolate(p.x, p.y, *kInterp);
}

#else
double
SBPixel::xValue(Position<double> p) const {
  // Interpolate WITHOUT wrapping the image.
  int ixMin = static_cast<int> ( ceil(p.x/dx - xInterp->xrange()));
  ixMin = MAX(ixMin, -Ninitial/2);
  int ixMax = static_cast<int> ( floor(p.x/dx + xInterp->xrange()));
  ixMax = MIN(ixMax, Ninitial/2-1);
  int iyMin = static_cast<int> ( ceil(p.y/dx - xInterp->xrange()));
  iyMin = MAX(iyMin, -Ninitial/2);
  int iyMax = static_cast<int> ( floor(p.y/dx + xInterp->xrange()));
  iyMax = MIN(iyMax, Ninitial/2-1);

  if (ixMax < ixMin || iyMax < iyMin) return 0.;  // kernel does not overlap data
  int npts = (ixMax - ixMin+1)*(iyMax-iyMin+1);
  DVector kernel(npts,0.);
  DMatrix data(Nimages, npts, 0.);
  int ipt = 0;
  for (int iy = iyMin; iy <= iyMax; iy++) {
    double deltaY = p.y/dx - iy;
    for (int ix = ixMin; ix <= ixMax; ix++, ipt++) {
      double deltaX = p.x/dx - ix;
      kernel[ipt] = xInterp->xval(deltaX, deltaY);
      for (int iz=0; iz<Nimages; iz++) {
	data(iz, ipt) = vx[iz]->xval(ix, iy);
      }
    }
  }
  return wts * data * kernel;
}

DComplex
SBPixel::kValue(Position<double> p) const {
  checkReady();
  // Interpolate in k space, first apply kInterp kernel to wrapped
  // k-space data, then multiply by FT of xInterp kernel.

  // Don't bother if the desired k value is cut off by the x interpolant:
  double ux = p.x*dx/TPI;
  if (abs(ux) > xInterp->urange()) return DComplex(0.,0.);
  double uy = p.y*dx/TPI;
  if (abs(uy) > xInterp->urange()) return DComplex(0.,0.);
  double xKernelTransform = xInterp->uval(ux, uy);

  // Range of k points within kernel
  int ixMin = static_cast<int> ( ceil(p.x/dk - kInterp->xrange()));
  int ixMax = static_cast<int> ( floor(p.x/dk + kInterp->xrange()));
  int iyMin = static_cast<int> ( ceil(p.y/dk - kInterp->xrange()));
  int iyMax = static_cast<int> ( floor(p.y/dk + kInterp->xrange()));

  int ixLast = MIN(ixMax, ixMin+Nk-1);
  int iyLast = MIN(iyMax, iyMin+Nk-1);
  int npts = (ixLast-ixMin+1) * (iyLast-iyMin+1);
  DVector kernel(npts, 0.);
  CMatrix data(Nimages, npts, DComplex(0.,0.));

  int ipt = 0;
  for (int iy = iyMin; iy <= iyLast; iy++) {
    for (int ix = ixMin; ix <= ixLast; ix++) {
      // sum kernel values for all aliases of this frequency
      double sumk = 0.;
      int iyy=iy;
      while (iyy <= iyMax) {
	double deltaY = p.y/dk - iyy;
	int ixx = ix;
	while (ixx <= ixMax) {
	  double deltaX = p.x/dk - ixx;
	  sumk += kInterp->xval(deltaX, deltaY);
	  ixx += Nk;
	}
	iyy += Nk;
      }
      // Shift ix,iy into un-aliased zone to get k value
      iyy = iy % Nk;  
      if(iyy>=Nk/2) iyy-=Nk; 
      if(iyy<-Nk/2) iyy+=Nk;
      int ixx = ix % Nk;  
      if(ixx>=Nk/2) ixx-=Nk; 
      if(ixx<-Nk/2) ixx+=Nk;
      for (int iz=0; iz<Nimages; iz++) 
	data(iz, ipt) = vk[iz]->kval(ixx, iyy);
      kernel[ipt] = sumk;
      ipt++;
    }
  }
  return xKernelTransform*(wts * data * kernel);
}
#endif

