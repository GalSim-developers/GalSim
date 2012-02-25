
// Mimic what Rachel has done in SHERA code and compare to her results.
#include <iostream>
#include "GalSim.h"

int main(int argc, char *argv[]) try 
{
    sbp::Lanczos l3(3, true, 1e-4);
    sbp::InterpolantXY l32d(l3);

    const double dxHST = 0.03;
    const double dxSDSS = 0.396;
    const double g1 = 0.02;
    const double g2 = 0.;
    const double psfSky = 1000.;

    const std::string rootname = argv[1];
    const double xshift = argc>2 ? atof(argv[2]) : 0.;
    const double yshift = argc>3 ? atof(argv[3]) : 0.;
    sbp::Shear s;
    s.setG1G2(g1, g2);
    // Rachel is probably using the (1+g, 1-g) form of shear matrix,
    // which means there is some (de)magnification, by my definition:
    sbp::Ellipse e(s, -(g1*g1+g2*g2), sbp::Position<double>(xshift,yshift));

    sbp::FITSImage<float> galaxyFits(rootname+"_masknoise.fits");
    sbp::Image<float> galaxyImg = galaxyFits.extract();
    sbp::SBPixel galaxy(galaxyImg, l32d, dxHST, 1.);
    galaxy.setFlux(0.804*1000.*dxSDSS*dxSDSS);

    sbp::FITSImage<float> psf1Fits(rootname+".psf.fits");
    sbp::Image<float> psf1Img = psf1Fits.extract();
    sbp::SBPixel psf1(psf1Img, l32d, dxHST, 2.);
    psf1.setFlux(1.);

    sbp::FITSImage<float> psf2Fits(rootname+".sdsspsf.fits");
    sbp::Image<float> psf2Img = psf2Fits.extract();
    psf2Img -= psfSky;
    sbp::SBPixel psf2(psf2Img, l32d, dxSDSS, 2.);
    psf2.setFlux(1.);

    sbp::FITSImage<float> outFits(rootname+".g1_0.02.g2_0.00.fits");
    sbp::Image<float> outImg = outFits.extract();
    sbp::Image<float> result = outImg.duplicate();

    sbp::SBDeconvolve psfInv(psf1);
    sbp::SBConvolve deconv(galaxy, psfInv);
    sbp::SBProfile* sheared = deconv.distort(e);
    sbp::SBConvolve out(*sheared, psf2);

    out.draw(result, dxSDSS);
    result.shift(1,1);
    sbp::FITSImage<float>::writeToFITS(rootname+".gary.fits", result);
    result += psfSky;
    result -= outImg;
    sbp::FITSImage<float>::writeToFITS(rootname+".diff.fits", result);

    return 0;

} catch (std::runtime_error &m) {
    sbp::quit(m,1);
}
