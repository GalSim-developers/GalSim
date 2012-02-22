//
#include "SBParse.h"
#include "StringStuff.h"
#include "Std.h"
#include "FITSImage.h"

using namespace sbp;

int
main(int argc, char *argv[])
{
  if (argc>5 || argc<3) {
    cerr << "Make a FITS file of a surface brightness pattern\n"
      "Usage:  SBDraw <sb_string> <fitsname> [dx] [wmult=1]\n"
      "  sb_string is parsed to define the pattern, enclose in quotes on cmd line\n"
      "  fitsname  is name of output FITS file\n"
      "  dx is pixel scale for output image, default is to choose automatically\n"
      "  wmult is optional integral factor by which to expand image size beyond default\n" << endl;
    exit(1);
  }
  try {
    string sbs=argv[1];
    double dx = argc>3 ? atof(argv[3]) : 0.;
    int wmult = argc>4 ? atoi(argv[4]) : 1;
    SBProfile* sbp = SBParse(sbs);
    Image<> img=sbp->draw(dx, wmult);
    cout << "Pixel scale chosen: " << dx << endl;
    img.shift(1,1);
    FITSImage<>::writeToFITS(argv[2],img);
    delete sbp;
  } catch (std::runtime_error& m) {
    quit(m,1);
  }
  exit(0);
}
