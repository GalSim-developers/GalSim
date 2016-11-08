/* -*- c++ -*-
 * Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
 * ------------------------------------------------------------------------------
 * Author: Craig Lage, UC Davis
 * Date: Nov 8, 2016
 * Routines for integrating the CCD simulations into GalSim
 */

//****************** Silicon.cpp **************************

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <algorithm>

// Uncomment this for debugging output
//#define DEBUGLOGGING

#include "Silicon.h"
#include "Image.h"
#include "PhotonArray.h"


namespace galsim {

  Silicon::Silicon (int NumVertices, int NumElec, int Nx, int Ny, int QDist, double DiffStep, double** vertex_data)
  {
    // This constructor reads in the distorted pixel shapes from the Poisson solver
    // and builds an array of polygons for calculating the distorted pixel shapes
    // as a function of charge in the surrounding pixels.

    /* Next, we build the polygons. We have an array of Nx*Ny + 2 polygons,
       with 0 to Nx*Ny-1 holding the pixel distortions, Nx*Ny being the polygon
       being tested, and Nx*Ny+1 being an undistorted polygon.  First, we build all
       of them as undistorted polygons. */

  int NumPolys, Nv;
  double theta, theta0, dtheta;
  dtheta = M_PI / (2.0 * ((double)NumVertices + 1.0));
  theta0 = - M_PI / 4.0;
  NumPolys = Nx * Ny + 2;
  Nv = 4 * NumVertices + 4; // Number of vertices in each pixel
  int xpix, ypix, n, p, pointcounter = 0;
  polylist = new Polygon*[NumPolys];
  Point **point = new Point*[Nv * NumPolys];
  for (p=0; p<NumPolys; p++)
    {
      polylist[p] = new Polygon(Nv);
      // First the corners
      for (xpix=0; xpix<2; xpix++)
        {
	  for (ypix=0; ypix<2; ypix++)
            {
	      point[pointcounter] = new Point((double)xpix, (double)ypix);
	      polylist[p]->AddPoint(point[pointcounter]);
	      pointcounter += 1;
            }
        }
      // Next the edges
      for (xpix=0; xpix<2; xpix++)
        {
	  for (n=0; n<NumVertices; n++)
            {
	      theta = theta0 + ((double)n + 1.0) * dtheta;
	      point[pointcounter] = new Point((double)xpix, (tan(theta) + 1.0) / 2.0);
	      polylist[p]->AddPoint(point[pointcounter]);
	      pointcounter += 1;
            }
        }
      for (ypix=0; ypix<2; ypix++)
        {
	  for (n=0; n<NumVertices; n++)
            {
	      theta = theta0 + ((double)n + 1.0) * dtheta;
	      point[pointcounter] = new Point((tan(theta) + 1.0) / 2.0, (double)ypix);
	      polylist[p]->AddPoint(point[pointcounter]);
	      pointcounter += 1;
            }
        }
      polylist[p]->Sort(); // Sort the vertices in CCW order
    }

  //Next, we read in the pixel distortions from the Poisson_CCD simulations

  double x0, y0, th, x1, y1;
  int index, i, j;
  for (index=0; index<Nv*(NumPolys - 2); index++)
    {
      n = (index % (Ny * Nv)) % Nv;
      j = (index - n) / Nv;
      i = (index - n - j * Nv) / (Ny * Nv);
      x0 = vertex_data[index][0];
      y0 = vertex_data[index][1];
      th = vertex_data[index][2];
      x1 = vertex_data[index][3];
      y1 = vertex_data[index][4];	
#ifdef DEBUGLOGGING
      if (index == 73) // Test print out of read in
        {
	  dbg<<"Successfully reading the Pixel vertex file\n";
	  dbg<<"line = "<<line<<std::endl;
	  dbg<<"n = "<<n<<", i = "<<i<<", j = "<<j<<", x0 = "<<x0<<", y0 = "<<y0
	     <<", th = "<<th<<", x1 = "<<x1<<", y1 = "<<y1<<std::endl;
        }
#endif
      
      // The following captures the pixel displacement. These are translated into
      // coordinates compatible with (x,y). These are per electron.
      polylist[i * Ny + j]->pointlist[n]->x = (((x1 - x0) / PixelSize + 0.5) - polylist[i * Ny + j]->pointlist[n]->x) / (double)NumElec;
      polylist[i * Ny + j]->pointlist[n]->y = (((y1 - y0) / PixelSize + 0.5) - polylist[i * Ny + j]->pointlist[n]->y) / (double)NumElec;
    }
  //Test print out of distortion for central pixel
#ifdef DEBUGLOGGING
  i = 4; j = 4;
  for (n=0; n<Nv; n++)
    xdbg<<"n = "<<n<<", x = "<<polylist[i * Ny + j]->pointlist[n]->x*(double)NumElec
	<<", y = "<<polylist[i * Ny + j]->pointlist[n]->y*(double)NumElec<<std::endl;
#endif
  // We generate a testpoint for testing whether a point is inside or outside the array
  testpoint = new Point(0.0,0.0);
  }
  
  Silicon::~Silicon ()
  {
    int p, NumPolys = Nx * Ny + 2;
    for (p=0; p<NumPolys; p++)
      {
        delete polylist[p];
      }
    delete[] polylist;
  }
  
  template <typename T>
  bool Silicon::InsidePixel(int ix, int iy, double x, double y, double zconv,
			    ImageView<T> target) const
  {
    // This builds the polygon under test based on the charge in the nearby pixels
    // and tests to see if the delivered position is inside it.
    // (ix,iy) is the pixel being tested, and (x,y) is the coordiante of the
    // photon within the pixel, with (0,0) in the lower left

    int TestPoly, EmptyPoly, i, j, ii, jj, chargei, chargej, n;
    double charge;
    int NxCenter, NyCenter;
    double dx, dy;
    // The term zfactor decreases the pixel shifts as we get closer to the bottom
    // It is an empirical fit to the Poisson solver simulations, and only matters
    // when we get quite close to the bottom.

    double zfit = 12.0;
    double zfactor = 0.0;
    zfactor = tanh(zconv / zfit);

    TestPoly = Nx * Ny; // Index of polygon being tested
    EmptyPoly = Nx * Ny + 1; // Index of undistorted polygon
    NxCenter = (Nx - 1) / 2;
    NyCenter = (Ny - 1) / 2;
    // First set the test polygon to an undistorted polygon
    for (n=0; n<Nv; n++)
    {
        polylist[TestPoly]->pointlist[n]->x = polylist[EmptyPoly]->pointlist[n]->x;
        polylist[TestPoly]->pointlist[n]->y = polylist[EmptyPoly]->pointlist[n]->y;
    }
    // Now add in the displacements
    int minx, miny, maxx, maxy;
    minx = target.getXMin();
    miny = target.getYMin();
    maxx = target.getXMax();
    maxy = target.getYMax();
    xdbg<<"minx = "<<minx<<", miny = "<<miny<<", maxx = "<<maxx<<", maxy = "<<maxy<<std::endl;
    xdbg<<"ix = "<<ix<<", iy = "<<iy<<", target(ix,iy) = "<<target(ix,iy)
        <<", x = "<<x<<", y = "<<y<<std::endl;

    for (i=NxCenter-QDist; i<NxCenter+QDist+1; i++)
    {
        for (j=NyCenter-QDist; j<NyCenter+QDist+1; j++)
        {
            chargei = ix + i - NxCenter;
            chargej = iy + j - NyCenter;
            if ((chargei < minx) || (chargei > maxx) || (chargej < miny) || (chargej > maxy))
                continue;

            charge = target(chargei,chargej);
            if (charge < 10.0)
                continue;

            for (n=0; n<Nv; n++)
            {
                ii = 2 * NxCenter - i;
                jj = 2 * NyCenter - j;
                dx = polylist[ii * Ny + jj]->pointlist[n]->x * charge * zfactor;
                dy = polylist[ii * Ny + jj]->pointlist[n]->y * charge * zfactor;
                polylist[TestPoly]->pointlist[n]->x += dx;
                polylist[TestPoly]->pointlist[n]->y += dy;
            }
        }
    }

#ifdef DEBUGLOGGING
    for (n=0; n<Nv; n++) // test printout of distorted pixel vertices.
        xdbg<<"n = "<<n<<", x = "<<polylist[TestPoly]->pointlist[n]->x
            <<", y = "<<polylist[TestPoly]->pointlist[n]->y<<std::endl;
#endif

    // Now test to see if the point is inside
    testpoint->x = x;
    testpoint->y = y;
    if (polylist[TestPoly]->PointInside(testpoint))
    {
        return true;
    }
    else
    {
        return false;
    }
}


template <typename T>
double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                           ImageView<T> target) const
{
    // Create and read in pixel distortions
    int xoff[9] = {0,1,1,0,-1,-1,-1,0,1}; // Displacements to neighboring pixels
    int yoff[9] = {0,0,1,1,1,0,-1,-1,-1}; // Displacements to neighboring pixels
    int n=0;
    double zconv = 95.0; // Z coordinate of photoconversion in microns
                         // Will add more detail later by drawing this from a distribution
                         // once we have wavelength information.
    GaussianDeviate gd(ud,0,1); // Random variable from Standard Normal dist.

    double DiffStep; // Mean diffusion step size in microns
    if (zconv <= 10.0)
    { DiffStep = 0.0; }
    else
    { DiffStep = this->DiffStep * (zconv - 10.0) / 100.0; }

    int zerocount = 0, nearestcount = 0, othercount = 0, misscount = 0;

    Bounds<int> b = target.getBounds();

    if (!b.isDefined())
        throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                 " undefined Bounds");

    // Factor to turn flux into surface brightness in an Image pixel
    dbg<<"In Silicon::accumulate\n";
    dbg<<"bounds = "<<b<<std::endl;
    dbg<<"total nphotons = "<<photons.size()<<std::endl;

    double addedFlux = 0.;
    double Irr = 0.;
    double Irr0 = 0.;
    for (int i=0; i<int(photons.size()); i++)
    {
        // First we add in a displacement due to diffusion
        double x0 = photons.getX(i) + DiffStep * gd() / 10.0;
        double y0 = photons.getY(i) + DiffStep * gd() / 10.0;
        double flux = photons.getFlux(i);

        // Now we find the undistorted pixel
        int ix = int(floor(x0 + 0.5));
        int iy = int(floor(y0 + 0.5));
        int ix0 = ix;
        int iy0 = iy;

        double x = x0 - (double) ix + 0.5;
        double y = y0 - (double) iy + 0.5;
        // (ix,iy) are the undistorted pixel coordinates.
        // (x,y) are the coordinates within the pixel, centered at the lower left

        // The following code finds which pixel we are in given
        // pixel distortion due to the brighter-fatter effect
        bool FoundPixel = false;
        // The following are set up to start the search in the undistorted pixel, then
        // search in the nearest neighbor first if it's not in the undistorted pixel.
        int step;
        if      ((x > y) && (x > 1.0 - y)) step = 1;
        else if ((x < y) && (x < 1.0 - y)) step = 7;
        else if ((x < y) && (x > 1.0 - y)) step = 3;
        else step = 5;
        int n=0;
        int m_found;
        for (int m=0; m<9; m++)
        {
            int ix_off = ix + xoff[n];
            int iy_off = iy + yoff[n];
            double x_off = x - (double)xoff[n];
            double y_off = y - (double)yoff[n];
            if (this->InsidePixel(ix_off, iy_off, x_off, y_off, zconv, target))
            {
                xdbg<<"Found in pixel "<<n<<", ix = "<<ix<<", iy = "<<iy
                    <<", x="<<x<<", y = "<<y<<", target(ix,iy)="<<target(ix,iy)<<std::endl;
                if (m == 0) zerocount += 1;
                else if (m == 1) nearestcount += 1;
                else othercount +=1;
                ix = ix_off;
                iy = iy_off;
                m_found = m;
                FoundPixel = true;
                break;
            }
            n = ((n-1)+step) % 8 + 1;
            // This is intended to start with the nearest neighbor, then cycle through the others.
        }
        if (!FoundPixel)
        {
            dbg<<"Not found in any pixel\n";
            dbg<<"ix,iy = "<<ix<<','<<iy<<"  x,y = "<<x<<','<<y<<std::endl;
            // We should never arrive here, since this means we didn't find it in the undistorted
            // pixel or any of the neighboring pixels.  However, sometimes (about 0.01% of the
            // time) we do arrive here due to roundoff error of the pixel boundary.  When this
            // happens, I put the electron in the undistorted pixel or the nearest neighbor with
            // equal probability.
            misscount += 1;
            if (ud() > 0.5)
            {
                n = 0;
                zerocount +=1;
            }
            else
            {
                n = step;
                nearestcount +=1;
            }
            ix = ix + xoff[n];
            iy = iy + yoff[n];
            FoundPixel = true;
        }
#if 0
        // (ix, iy) now give the actual pixel which will receive the charge
        if (ix != ix0 || iy != iy0) {
            dbg<<"("<<ix0<<","<<iy0<<") -> ("<<ix<<","<<iy<<")\n";
            double r0 = sqrt((ix0+0.5)*(ix0+0.5)+(iy0+0.5)*(iy0+0.5));
            double r = sqrt((ix+0.5)*(ix+0.5)+(iy+0.5)*(iy+0.5));
            dbg<<"r = "<<r0<<" -> "<<r;
            if (r < r0) { dbg<<"  *****"; }
            dbg<<"\nstep = "<<step<<", n = "<<n<<", m_found = "<<m_found<<std::endl;
            dbg<<"flux = "<<photons.getFlux(i)<<std::endl;
        }
#endif

        if (b.includes(ix,iy)) {
            double rsq = (ix+0.5)*(ix+0.5)+(iy+0.5)*(iy+0.5);
            Irr += flux * rsq;
            rsq = (ix0+0.5)*(ix0+0.5)+(iy0+0.5)*(iy0+0.5);
            Irr0 += flux * rsq;
            target(ix,iy) += flux;
            addedFlux += flux;
        }
    }
    Irr /= addedFlux;
    Irr0 /= addedFlux;
    dbg<<"Irr = "<<Irr<<"  cf. Irr0 = "<<Irr0<<std::endl;
    dbg << "Found "<< zerocount << " photons in undistorted pixel, " << nearestcount;
    dbg << " in closest neighbor, " << othercount << " in other neighbor. " << misscount;
    dbg << " not in any pixel\n" << std::endl;
    return addedFlux;
}

template bool Silicon::InsidePixel(int ix, int iy, double x, double y, double zconv,
                                   ImageView<double> target) const;
template bool Silicon::InsidePixel(int ix, int iy, double x, double y, double zconv,
                                   ImageView<float> target) const;

template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                    ImageView<double> target) const;
template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                    ImageView<float> target) const;

} // ends namespace galsim
