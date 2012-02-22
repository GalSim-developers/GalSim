// $Id$
// Mimic what Rachel has done in SHERA code and compare to her results.
#include "SBPixel.h"
#include "FITSImage.h"
#include "SBDeconvolve.h"
#include <iostream>

using namespace img;
using namespace sbp;
using namespace std;

int main(int argc,
	 char *argv[])
{
  try {
    Lanczos l3(3, true, 1e-4);
    InterpolantXY l32d(l3);

    const double dxHST = 0.03;
    const double dxSDSS = 0.396;
    const double g1 = 0.02;
    const double g2 = 0.;
    const double psfSky = 1000.;

    const string rootname = argv[1];
    const double xshift = argc>2 ? atof(argv[2]) : 0.;
    const double yshift = argc>3 ? atof(argv[3]) : 0.;
    Shear s;
    s.setG1G2(g1, g2);
    // Rachel is probably using the (1+g, 1-g) form of shear matrix,
    // which means there is some (de)magnification, by my definition:
    Ellipse e(s, -(g1*g1+g2*g2), Position<double>(xshift,yshift));

      FITSImage<> galaxyFits(rootname+"_masknoise.fits");
      Image<> galaxyImg = galaxyFits.extract();
      SBPixel galaxy(galaxyImg, l32d, dxHST, 1.);
      galaxy.setFlux(0.804*1000.*dxSDSS*dxSDSS);

      FITSImage<> psf1Fits(rootname+".psf.fits");
      Image<> psf1Img = psf1Fits.extract();
      SBPixel psf1(psf1Img, l32d, dxHST, 2.);
      psf1.setFlux(1.);

      FITSImage<> psf2Fits(rootname+".sdsspsf.fits");
      Image<> psf2Img = psf2Fits.extract();
      psf2Img -= psfSky;
      SBPixel psf2(psf2Img, l32d, dxSDSS, 2.);
      psf2.setFlux(1.);

      FITSImage<> outFits(rootname+".g1_0.02.g2_0.00.fits");
      Image<> outImg = outFits.extract();
      Image<> result = outImg.duplicate();

      SBDeconvolve psfInv(psf1);
      SBConvolve deconv(galaxy, psfInv);
      SBProfile* sheared = deconv.distort(e);
      SBConvolve out(*sheared, psf2);

      out.draw(result, dxSDSS);
      result.shift(1,1);
      FITSImage<>::writeToFITS(rootname+".gary.fits", result);
      result += psfSky;
      result -= outImg;
      FITSImage<>::writeToFITS(rootname+".diff.fits", result);

  } catch (std::runtime_error &m) {
    quit(m,1);
  }
}
