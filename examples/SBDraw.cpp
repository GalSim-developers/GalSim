
#include "GalSim.h"

int main(int argc, char *argv[])
{
    if (argc>5 || argc<3) {
        std::cerr << "Make a FITS file of a surface brightness pattern\n"
            "Usage:  SBDraw <sb_string> <fitsname> [dx] [wmult=1]\n"
            "  sb_string is parsed to define the pattern, enclose in quotes on cmd line\n"
            "  fitsname  is name of output FITS file\n"
            "  dx is pixel scale for output image, default is to choose automatically\n"
            "  wmult is optional integral factor by which to expand image size beyond default\n" 
            << std::endl;
        exit(1);
    }

    try {
        std::string sbs=argv[1];
        double dx = argc>3 ? atof(argv[3]) : 0.;
        int wmult = argc>4 ? atoi(argv[4]) : 1;
        galsim::SBProfile* sbp = galsim::SBParse(sbs);
        galsim::Image<float> img=sbp->draw(dx, wmult);
        std::cout << "Pixel scale chosen: " << dx << std::endl;
        img.shift(1,1);
        galsim::FITSImage<float>::writeToFITS(argv[2],img);
        delete sbp;
    } catch (std::runtime_error& err) {
        dbg << err.what() << std::endl;
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
