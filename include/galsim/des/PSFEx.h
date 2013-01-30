//
// This file was originally written by Daniel Gruen and modified by Peter Melchior before
// being ported into GalSim.
// 
// Comments from Peter:
//
// Attached you'll find the PSFEx class I've implemented for my shape pipeline. It derived from 
// Daniel's implementation, but uses some modifications to speed things up or make them more 
// obvious to me. It also uses data structures from my shapelens library, but you should find 
// them fairly obvious. The API docs of shapelens are here:
// http://www.physics.ohio-state.edu/~melchior.12/docs/shapelens/classes.html
//
// One thing I changed for speed it to replace the pow() function by my own template pow_int() 
// function.
//
// Daniel has agreed to make it public as part of galsim (with proper attribution, of course).
//


#include <shapelens/ShapeLens.h>
#include <shapelens/MathHelper.h>
#include <shapelens/Object.h>
#include <vector>
#include <string>

using namespace shapelens;

class PSFEx {

public:
  PSFEx(std::string filename) :
    polzero(2), polscale(2), psfaxis(3) {

    fitsfile* fptr = FITS::openFile(filename);
    FITS::moveToExtension(fptr, 2);
    int polnaxis;
    FITS::readKeyword(fptr, "POLNAXIS", polnaxis);
    if (polnaxis != 2)
      throw std::invalid_argument("PSFEx: POLNAXIS != 2");
    FITS::readKeyword(fptr, "POLZERO1", polzero[0]);
    FITS::readKeyword(fptr, "POLSCAL1", polscale[0]);
    FITS::readKeyword(fptr, "POLZERO2", polzero[1]);
    FITS::readKeyword(fptr, "POLSCAL2", polscale[1]);
    std::vector<std::string> polname(2);
    FITS::readKeyword(fptr, "POLNAME1", polname[0]);
    FITS::readKeyword(fptr, "POLNAME2", polname[1]);
    int psfnaxis;
    FITS::readKeyword(fptr, "PSFNAXIS", psfnaxis);
    if (psfnaxis != 3)
      throw std::invalid_argument("PSFEx: PSFNAXIS != 3");
    FITS::readKeyword(fptr, "PSFAXIS1", psfaxis[0]);
    FITS::readKeyword(fptr, "PSFAXIS2", psfaxis[1]);
    FITS::readKeyword(fptr, "PSFAXIS3", psfaxis[2]);
    FITS::readKeyword(fptr, "POLDEG1", poldeg);
    if (psfaxis[2] != ((poldeg+1)*(poldeg+2))/2)
      throw std::invalid_argument("PSFEx: POLDEG and PSFAXIS3 disagree");
    std::cout << "# " << psfaxis[0] << "\t" << psfaxis[1] << "\t" << poldeg << " -> " << psfaxis[2] << std::endl;
    FITS::readKeyword(fptr, "PSF_SAMP", psf_samp);

    // read basis functin shapes into images
    int status = 0, anynull;
    basis = std::vector<Image<data_t> >(psfaxis[2], Image<data_t>(psfaxis[0], psfaxis[1]));
    for (int k = 0; k < psfaxis[2]; k++)
      fits_read_col(fptr, FITS::getDataType(data_t(0)), 1, 1, 1 + k*psfaxis[0]*psfaxis[1], psfaxis[0]*psfaxis[1], NULL, basis[k].ptr(), &anynull, &status);

    FITS::closeFile(fptr);
  }

  data_t maxsize() {
    return ((psfaxis[0]-1)/2.-INTERPFAC)*psf_samp;
  }

  
  // image sampled psf pixel at position relx,rely relative to the psf center at centerx,centery
  // brightest pixel is at relx=rely=0
  data_t pixel_lanczos3(data_t relx, data_t rely, data_t centerx, data_t centery)  {
    data_t relxp = relx/psf_samp;
    data_t relyp = rely/psf_samp;

    
    if (fabs(relxp) > (psfaxis[0]-1)/2.-INTERPFAC || fabs(relyp) > (psfaxis[1]-1)/2.-INTERPFAC) {
      std::cerr << "interpolating out of bounds at (" << relx << "," << rely << ")." << std::endl;
      return 0;
    }
    
    data_t sum = 0.;
    for(int i=0; i<psfaxis[0]; i++) { // x coordinate on psf_sample scale
      for(int j=0; j<psfaxis[0]; j++) { // y coordinate on psf_sample scale
	data_t dx = fabs(i - 0.5*psfaxis[0] - relxp);
	data_t dy = fabs(j - 0.5*psfaxis[1] - relyp);
	
	if(dx<INTERPFAC && dy<INTERPFAC) {
	  data_t interpolant = sinc(dx)*sinc(dx/INTERPFAC)*sinc(dy)*sinc(dy/INTERPFAC);
	  data_t value = pixel_sampled_interpolated(i,j,centerx,centery);
	  sum += value*interpolant;
	}
      }
    }
    return sum/pow2(psf_samp);
  }


  // i-th psf (i.e. sampled in psf_samp size pixels) pixel in x direction,
  // j-th psf pixel in y direction, interpolated at (x,y) central coordinate of psf
  // caveat: only works when there's just one polngrp
  double pixel_sampled_interpolated(int i, int j, double x, double y) {
    if (i>=0 && j>=0 && i<psfaxis[0] && j<psfaxis[1]) {
      double dx = (x-polzero[0])/polscale[0];
      double dy = (y-polzero[1])/polscale[1];
      double result = basis[0](i,j);
      
      // iterate through all orders from 1 to n (zero done above already)
      for(int n=1; n<=poldeg; n++) {
	for(int ny=0; ny <= n; ny++) { // orders in y
	  int nx = n-ny; // order in x so x^(nx)*y^(ny) is of order n=nx+ny
	  int k = nx+ny*(poldeg+1)-(ny*(ny-1))/2;
	  result += pow_int(dx,nx) * pow_int(dy,ny)* basis[k](i,j);
	}
      }
      return result;
    }
    
    if(i==psfaxis[0] || j==psfaxis[1] || i==-1 || j==-1)
      return 0.; // silently ignore
  
    std::cerr << "warning: requesting pixel outside valid range " << i << " " << j << std::endl;
    return 0;
  }

  void fillObject(Object& psf) {
    data_t sum = 0.;
    data_t m = maxsize();
    Point<data_t> P;
    for (int i=0; i < psf.size(); i++) {
      P = psf.grid(i);
      if (fabs(P(0)-psf.centroid(0)) < m && fabs(P(1) - psf.centroid(1)) < m) {
	sum += psf(i) = pixel_lanczos3(P(0)-psf.centroid(0), P(1) - psf.centroid(1), psf.centroid(0), psf.centroid(1));
      }
    }
    psf *= 1./sum;
  }

private:
  double sinc(double x) { // normalized sinc function, see PSFEx manual
    if (x<1e-5 && x>-1e-5)
      return 1.;
    return sin(x*M_PI)/(x*M_PI);
  }
  
  static const data_t INTERPFAC = 3.0;
  std::vector<data_t> polzero, polscale;
  std::vector<int> psfaxis;
  int poldeg;
  data_t psf_samp;
  std::vector<Image<data_t> > basis;
};
