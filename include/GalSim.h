// This just includes the relevant header files from the galsim directory.

#ifndef GALSIM_H
#define GALSIM_H

// The basic SBProfile stuff:
#include "galsim/SBProfile.h"
#include "galsim/SBDeconvolve.h"
#include "galsim/SBParse.h"
#include "galsim/SBInterpolatedImage.h"

// An interface for dealing with images
#include "galsim/Image.h"

// FFT's
#include "galsim/FFT.h"

// Noise stuff
#include "galsim/Random.h"
#include "galsim/CCDNoise.h"

// An integration package by Mike Jarvis
#include "galsim/integ/Int.h"

// Adaptive moments code by Hirata, Seljak, and Mandelbaum
#include "galsim/hsm/PSFCorr.h"

#endif
