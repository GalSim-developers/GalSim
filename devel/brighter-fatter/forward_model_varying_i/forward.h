/* -*- c++ -*-
 * Copyright (c) 2012-2022 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */
/*
Craig Lage - 15-Jul-15

C++ code to calculate forward model fit
for Gaussian spots

file: forward.h

*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>            // for min_element
#include <string.h>
#include <math.h>
#define pi 4.0 * atan(1.0)     // Pi
#define max(x,y) (x>y?x:y)      // max macro definition
#define min(x,y) (x>y?y:x)      // min macro definition

using namespace std;

//  DATA STRUCTURES:  


class Array //This packages the 2D data sets which are brought from Python
{
 public:
  long nx, ny, nstamps;
  double xmin, xmax, ymin, ymax, dx, dy, *x, *y, *xoffset, *yoffset, *imax, *data;
  Array() {};
  ~Array(){};
};

// FUNCTION PROTOTYPES


double FOM(Array*, double, double);
double Area(double, double, double, double, double, double);




