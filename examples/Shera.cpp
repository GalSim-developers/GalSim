// This file is deprecated in favor of Shera.py, which does the same thing in Python
// (and actually works; C++ FITS output has been removed, so this program does not
// compile and is ignored by scons).

// Mimic what Rachel has done in SHERA code and compare to her results.
#include <iostream>
#include "GalSim.h"

int main(int argc, char *argv[]) try 
{
    galsim::Lanczos l3(3, true, 1e-4);
    galsim::InterpolantXY l32d(l3);

    const double dxHST = 0.03;
    const double dxSDSS = 0.396;
    const double g1 = 0.02;
    const double g2 = 0.;
    const double psfSky = 1000.;

    const std::string rootname = argv[1];
    const double xshift = argc>2 ? atof(argv[2]) : 0.;
    const double yshift = argc>3 ? atof(argv[3]) : 0.;
    galsim::Shear s;
    s.setG1G2(g1, g2);
    // Rachel is probably using the (1+g, 1-g) form of shear matrix,
    // which means there is some (de)magnification, by my definition:
    galsim::Ellipse e(s, -(g1*g1+g2*g2), galsim::Position<double>(xshift,yshift));

    galsim::FITSImage<float> galaxyFits(rootname+"_masknoise.fits");
    galsim::Image<float> galaxyImg = galaxyFits.extract();
    galsim::SBPixel galaxy(galaxyImg, l32d, dxHST, 1.);
    galaxy.setFlux(0.804*1000.*dxSDSS*dxSDSS);

    galsim::FITSImage<float> psf1Fits(rootname+".psf.fits");
    galsim::Image<float> psf1Img = psf1Fits.extract();
    galsim::SBPixel psf1(psf1Img, l32d, dxHST, 2.);
    psf1.setFlux(1.);

    galsim::FITSImage<float> psf2Fits(rootname+".sdsspsf.fits");
    galsim::Image<float> psf2Img = psf2Fits.extract();
    psf2Img -= psfSky;
    galsim::SBPixel psf2(psf2Img, l32d, dxSDSS, 2.);
    psf2.setFlux(1.);

    galsim::FITSImage<float> outFits(rootname+".g1_0.02.g2_0.00.fits");
    galsim::Image<float> outImg = outFits.extract();
    galsim::Image<float> result = outImg.duplicate();

    galsim::SBDeconvolve psfInv(psf1);
    galsim::SBConvolve deconv(galaxy, psfInv);
    galsim::SBProfile* sheared = deconv.distort(e);
    galsim::SBConvolve out(*sheared, psf2);

    out.draw(result, dxSDSS);
    result.shift(1,1);
    galsim::FITSImage<float>::writeToFITS(rootname+".gary.fits", result);
    result += psfSky;
    result -= outImg;
    galsim::FITSImage<float>::writeToFITS(rootname+".diff.fits", result);

    return 0;

} catch (std::runtime_error &err) {
    dbg << err.what() << std::endl;
    std::cerr << err.what() << std::endl;
    return 1;
}
