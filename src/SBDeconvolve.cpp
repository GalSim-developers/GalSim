
//#define DEBUGLOGGING

#include "SBDeconvolve.h"
#include "SBDeconvolveImpl.h"

#ifdef DEBUGLOGGING
#include <fstream>
std::ostream* dbgout = new std::ofstream("debug.out");
int verbose_level = 2;
#endif

namespace galsim {

    SBDeconvolve::SBDeconvolve(const SBProfile& adaptee) :
        SBProfile(new SBDeconvolveImpl(adaptee)) {}

    SBDeconvolve::SBDeconvolve(const SBDeconvolve& rhs) : SBProfile(rhs) {}

    SBDeconvolve::~SBDeconvolve() {}

}
