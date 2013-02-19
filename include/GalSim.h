/*
 * Copyright 2012, 2013 The GalSim developers:
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 *
 * GalSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GalSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GalSim.  If not, see <http://www.gnu.org/licenses/>
 */
// This just includes the relevant header files from the galsim directory.

#ifndef GALSIM_H
#define GALSIM_H

// The basic SBProfile stuff:
#include "galsim/SBProfile.h"
#include "galsim/SBDeconvolve.h"
#include "galsim/SBInterpolatedImage.h"

// An interface for dealing with images
#include "galsim/Image.h"

// FFT's
#include "galsim/FFT.h"

// Noise stuff
#include "galsim/Random.h"
#include "galsim/Noise.h"

// An integration package by Mike Jarvis
#include "galsim/integ/Int.h"

// Adaptive moments code by Hirata, Seljak, and Mandelbaum
#include "galsim/hsm/PSFCorr.h"

#endif
