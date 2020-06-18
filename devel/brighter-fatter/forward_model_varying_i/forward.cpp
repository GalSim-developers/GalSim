/* -*- c++ -*-
 * Copyright (c) 2012-2020 by the GalSim developers team on GitHub
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
for Gaussian spots.
This version allows different intensities for each spot.

file: forward.cpp

*/

#include "forward.h"


double FOM(Array* spotlist, double sigmax, double sigmay)
{
  int n, i, j, index;
  double xl, xh, yl, yh;
  double a2, b2, ab, area, result = 0.0;
  
  for (n=0; n<spotlist->nstamps; n++)
    {
      a2 = 0; b2 = 0; ab = 0;
      for (i=0; i<spotlist->nx; i++)
	{
	  for (j=0; j<spotlist->ny; j++)
	    {
	      index = n  +  j * spotlist->nstamps + i * (spotlist->nstamps * spotlist->ny);
	      xl = spotlist->x[i] - spotlist->xoffset[n] - 0.5;
	      xh = xl + spotlist->dx;
	      yl = spotlist->y[j] - spotlist->yoffset[n] - 0.5;
	      yh = yl + spotlist->dy;
              area = Area(xl, xh, yl, yh, sigmax, sigmay);
	      a2 += spotlist->data[index] * spotlist->data[index];
	      b2 += area * area;
	      ab += spotlist->data[index] * area;
	    }
	}
      spotlist->imax[n] = ab / b2;
      result += a2 - (ab * ab) / b2;
    }
  return result;
}

//  SUBROUTINES:

double Area(double xl, double xh, double yl, double yh, double sigmax, double sigmay)
{
  //Calculates how much of a 2D Gaussian falls within a rectangular box
  double ssigx, ssigy, i;
  ssigx = sqrt(2.0) * sigmax;
  ssigy = sqrt(2.0) * sigmay;    
  i = (erf(xh/ssigx)-erf(xl/ssigx))*(erf(yh/ssigy)-erf(yl/ssigy));
  return i / 4.0;
}
