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

  Silicon::Silicon (int NumVertices, int NumElec, int Nx, int Ny, int QDist, int Nrecalc, double DiffStep,
                    double PixelSize, double* vertex_data) :
      _NumVertices(NumVertices), _NumElect(NumElec), _Nx(Nx), _Ny(Ny), _QDist(QDist),
      _DiffStep(DiffStep), _PixelSize(PixelSize), _Nrecalc(Nrecalc)
  {
    // This constructor reads in the distorted pixel shapes from the Poisson solver
    // and builds an array of polygons for calculating the distorted pixel shapes
    // as a function of charge in the surrounding pixels.

    /* Next, we build the distorted polygons. We have an array of Nx*Ny polygons,
       an undistorted polygon, and a polygon for test. */

  _Nv = 4 * _NumVertices + 4; // Number of vertices in each pixel

  _distortions = new Polygon*[_Nx * _Ny]; // This carries the distorted pixel information
  BuildPolylist(_distortions, _Nx, _Ny);
  _emptypoly = new Polygon*[1];
  BuildPolylist(_emptypoly, 1, 1);
  _testpoly = new Polygon*[1];
  BuildPolylist(_testpoly, 1, 1);
  
  //Next, we read in the pixel distortions from the Poisson_CCD simulations

  double x0, y0, th, x1, y1;
  int index, i, j, n;
  for (index=0; index<_Nv*_Nx*_Ny; index++)
    {
      n = (index % (_Ny * _Nv)) % _Nv;
      j = (index - n) / _Nv;
      i = (index - n - j * _Nv) / (_Ny * _Nv);
      x0 = vertex_data[5*index+0];
      y0 = vertex_data[5*index+1];
      th = vertex_data[5*index+2];
      x1 = vertex_data[5*index+3];
      y1 = vertex_data[5*index+4];
#ifdef DEBUGLOGGING
      if (index == 73) // Test print out of read in
        {
	  dbg<<"Successfully reading the Pixel vertex file\n";
	  //dbg<<"line = "<<line<<std::endl;
	  dbg<<"n = "<<n<<", i = "<<i<<", j = "<<j<<", x0 = "<<x0<<", y0 = "<<y0
	     <<", th = "<<th<<", x1 = "<<x1<<", y1 = "<<y1<<std::endl;
        }
#endif
      
      // The following captures the pixel displacement. These are translated into
      // coordinates compatible with (x,y). These are per electron.
      _distortions[i * _Ny + j]->pointlist[n]->x = (((x1 - x0) / _PixelSize + 0.5) - _distortions[i * _Ny + j]->pointlist[n]->x) / (double)NumElec;
      _distortions[i * _Ny + j]->pointlist[n]->y = (((y1 - y0) / _PixelSize + 0.5) - _distortions[i * _Ny + j]->pointlist[n]->y) / (double)NumElec;
    }
  //Test print out of distortion for central pixel
#ifdef DEBUGLOGGING
  i = 4; j = 4;
  for (n=0; n<_Nv; n++)
    xdbg<<"n = "<<n<<", x = "<<_distortions[i * _Ny + j]->pointlist[n]->x*(double)NumElec
	<<", y = "<<_distortions[i * _Ny + j]->pointlist[n]->y*(double)NumElec<<std::endl;
#endif
  // We generate a testpoint for testing whether a point is inside or outside the array
  _testpoint = new Point(0.0,0.0);
  }
  
  Silicon::~Silicon ()
  {
    int p;
    for (p=0; p<_Nx*_Ny; p++)
      {
            delete _distortions[p];
      }
    delete[] _distortions;
    delete _emptypoly[0];
    delete[] _emptypoly;    
    delete _testpoly[0];
    delete[] _testpoly;    
  }


  void Silicon::BuildPolylist(Polygon** polylist, int nx, int ny)
  {
    int xpix, ypix, n, p, pointcounter = 0;
    double theta, theta0, dtheta;
    dtheta = M_PI / (2.0 * ((double)_NumVertices + 1.0));
    theta0 = - M_PI / 4.0;
    Point **point = new Point*[_Nv * nx * ny];
    
    for (p=0; p<nx*ny; p++)
      {
	polylist[p] = new Polygon(_Nv);
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
	    for (n=0; n<_NumVertices; n++)
	      {
		theta = theta0 + ((double)n + 1.0) * dtheta;
		point[pointcounter] = new Point((double)xpix, (tan(theta) + 1.0) / 2.0);
		polylist[p]->AddPoint(point[pointcounter]);
		pointcounter += 1;
	      }
	  }
	for (ypix=0; ypix<2; ypix++)
	  {
	    for (n=0; n<_NumVertices; n++)
	      {
		theta = theta0 + ((double)n + 1.0) * dtheta;
		point[pointcounter] = new Point((tan(theta) + 1.0) / 2.0, (double)ypix);
		polylist[p]->AddPoint(point[pointcounter]);
		pointcounter += 1;
	      }
	  }
	polylist[p]->Sort(); // Sort the vertices in CCW order
      }
    return;
  }

  template <typename T>
  void Silicon::UpdatePixelDistortions(ImageView<T> target) const
  {
    // This updates the pixel distortions in the _imagepolys
    // pixel list based on the amount of charge in each pixel
    // This distortion assumes the electron is created at the
    // top of the silicon.  It mus be scaled based on the conversion depth
    // This is handled in InsidePixel.

    int i, j, ii, jj, ix, iy, index, chargei, chargej, n;
    double charge;
    int NxCenter, NyCenter;
    double dx, dy;

    NxCenter = (_Nx - 1) / 2;
    NyCenter = (_Ny - 1) / 2;

    // Now add in the displacements
    int minx, miny, maxx, maxy;
    minx = target.getXMin();
    miny = target.getYMin();
    maxx = target.getXMax();
    maxy = target.getYMax();

    // Now we cycle through the _imagepolys array
    // and update the pixel shapes
    for (ix=minx; ix<maxx; ix++)
      {
	for (iy=miny; iy<maxy; iy++)
	  {
	    index = (ix - minx) * (maxy - miny + 1) + (iy - miny);
	    // First set the _imagepoly polygon to an undistorted polygon
	    for (n=0; n<_Nv; n++)
	      {
		_imagepolys[index]->pointlist[n]->x = _emptypoly[0]->pointlist[n]->x;
		_imagepolys[index]->pointlist[n]->y = _emptypoly[0]->pointlist[n]->y;		
	      }
	    // Now add in the distortions
	    for (i=NxCenter-_QDist; i<NxCenter+_QDist+1; i++)
	      {
		for (j=NyCenter-_QDist; j<NyCenter+_QDist+1; j++)
		  {
		    chargei = ix + i - NxCenter;
		    chargej = iy + j - NyCenter;
		    if ((chargei < minx) || (chargei > maxx) || (chargej < miny) || (chargej > maxy))
		      continue;
		    
		    charge = target(chargei,chargej);
		    if (charge < 100.0) // Don't bother if less than 100 electrons
		      continue;
		    
		    for (n=0; n<_Nv; n++)
		      {
			ii = 2 * NxCenter - i;
			jj = 2 * NyCenter - j;
			dx = _distortions[ii * _Ny + jj]->pointlist[n]->x * charge;
			dy = _distortions[ii * _Ny + jj]->pointlist[n]->y * charge;
			_imagepolys[index]->pointlist[n]->x += dx;
			_imagepolys[index]->pointlist[n]->y += dy;
		      }
		  }
	      }
	  }
      }
    return;
  }

  
  template <typename T>
  bool Silicon::InsidePixel(int ix, int iy, double x, double y, double zconv,
			    ImageView<T> target) const
  {
    // This scales the pixel distortion based on the zconv, which is the depth
    // at which the electron is created, and then tests to see if the delivered
    // point is inside the pixel.
    // (ix,iy) is the pixel being tested, and (x,y) is the coordinate of the
    // photon within the pixel, with (0,0) in the lower left

    int index, n;
    double dx, dy;
    // The term zfactor decreases the pixel shifts as we get closer to the bottom
    // It is an empirical fit to the Poisson solver simulations, and only matters
    // when we get quite close to the bottom.  This could be more accurate by making
    // the Vertices files have an additional look-up variable (z), but this doesn't
    // seem necessary at this point

    double zfit = 12.0;
    double zfactor = 0.0;
    zfactor = tanh(zconv / zfit);
    int minx, miny, maxx, maxy;
    minx = target.getXMin();
    miny = target.getYMin();
    maxx = target.getXMax();
    maxy = target.getYMax();
    index = (ix - minx) * (maxy - miny + 1) + (iy - miny);

    // Scale the _testpoly vertices by zfactor
    for (n=0; n<_Nv; n++)
    {
      _testpoly[0]->pointlist[n]->x = _emptypoly[0]->pointlist[n]->x + (_imagepolys[index]->pointlist[n]->x - _emptypoly[0]->pointlist[n]->x) * zfactor;
      _testpoly[0]->pointlist[n]->y = _emptypoly[0]->pointlist[n]->y + (_imagepolys[index]->pointlist[n]->y - _emptypoly[0]->pointlist[n]->y) * zfactor;
    }
    // Now test to see if the point is inside
    _testpoint->x = x;
    _testpoint->y = y;
    if (_testpoly[0]->PointInside(_testpoint))
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
                           ImageView<T> target)
{
    // Create and read in pixel distortions
    int xoff[9] = {0,1,1,0,-1,-1,-1,0,1}; // Displacements to neighboring pixels
    int yoff[9] = {0,0,1,1,1,0,-1,-1,-1}; // Displacements to neighboring pixels
    int n=0;

    // PhotonArray
    GaussianDeviate gd(ud,0,1); // Random variable from Standard Normal dist.
    int zerocount = 0, nearestcount = 0, othercount = 0, misscount = 0;

    Bounds<int> b = target.getBounds();

    if (!b.isDefined())
        throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                 " undefined Bounds");

    // Factor to turn flux into surface brightness in an Image pixel
#ifdef DEBUGLOGGING
    dbg<<"In Silicon::accumulate\n";
    dbg<<"bounds = "<<b<<std::endl;
    dbg<<"total nphotons = "<<photons.size()<<std::endl;
    dbg<<"hasAllocatedWavelengths = "<<photons.hasAllocatedWavelengths()<<std::endl;
    dbg<<"hasAllocatedAngles = "<<photons.hasAllocatedAngles()<<std::endl;    
#endif

    int minx, miny, maxx, maxy, nx, ny;
    minx = target.getXMin();
    miny = target.getYMin();
    maxx = target.getXMax();
    maxy = target.getYMax();
    nx = maxx - minx + 1;
    ny = maxy - miny + 1;    
    _imagepolys = new Polygon*[nx * ny]; // These pixels span the target image
    BuildPolylist(_imagepolys, nx, ny);
    
    double addedFlux = 0.;
    double Irr = 0.;
    double Irr0 = 0.;
    for (int i=0; i<int(photons.size()); i++)
    {
      if (i % _Nrecalc == 0) UpdatePixelDistortions(target); // Update shapes every _Nrecalc electrons

      // First we get the location where the photon strikes the silicon:
      double x0 = photons.getX(i); // in pixels
      double y0 = photons.getY(i); // in pixels
      // Next we determine the distance the photon travels into the silicon
      double si_length, zconv;
      if (photons.hasAllocatedWavelengths())
	{
	  double lambda = photons.getWavelength(i); // in nm
	  // The below is an approximation.  TODO: Need to do implement a look-up table
	  double abs_length = pow(10.0,((lambda - 500.0) / 250.0)); // in microns
	  si_length = -abs_length * log(1.0 - ud()); // in microns
#ifdef DEBUGLOGGING
      if (i % 1000 == 0)
	{
	  dbg<<"lambda = "<<lambda<<std::endl;
	  dbg<<"si_length = "<<si_length<<std::endl;	  	  
	}
#endif
	}
      else
	{
	si_length = 5.0; // If no wavelength info, assume conversion takes place near the top.
	}
      // Next we partition the si_length into x,y,z.  Assuming dz is positive downward
      if (photons.hasAllocatedAngles())
	{
	  double dxdz = photons.getDXDZ(i);
	  double dydz = photons.getDYDZ(i);
	  double deltaz = fmin(95.0, si_length / sqrt(1.0 + dxdz*dxdz + dydz*dydz));// in microns
	  x0 += dxdz * deltaz / 10.0; // in pixels
	  y0 += dydz * deltaz / 10.0; // in pixels
	  zconv = 100.0 - deltaz; // Conversion depth in microns
#ifdef DEBUGLOGGING
      if (i % 1000 == 0)
	{
	  dbg<<"dxdz = "<<dxdz<<std::endl;
	  dbg<<"dydz = "<<dydz<<std::endl;
	  dbg<<"deltaz = "<<deltaz<<std::endl;	  	  
	}
#endif

	}
      else
	{
	  zconv = 100.0 - si_length;
	}
      if (zconv < 0.0) continue; // Throw photon away if it hits the bottom
      // TODO: Do something more realistic if it hits the bottom.
      
      // Now we add in a displacement due to diffusion
      double DiffStep = fmax(0.0, _DiffStep * (zconv - 10.0) / 100.0); // in microns
      x0 += DiffStep * gd() / 10.0; // in pixels
      y0 += DiffStep * gd() / 10.0; // in pixels
      double flux = photons.getFlux(i);

#ifdef DEBUGLOGGING
      if (i % 1000 == 0)
	{
	  xdbg<<"DiffStep = "<<DiffStep<<std::endl;	  
	  xdbg<<"zconv = "<<zconv<<std::endl;
	  xdbg<<"x0 = "<<x0<<std::endl;
	  xdbg<<"y0 = "<<y0<<std::endl;
	}
#endif
      
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
            if (InsidePixel(ix_off, iy_off, x_off, y_off, zconv, target))
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
	  xdbg<<"Not found in any pixel\n";
	  xdbg<<"ix,iy = "<<ix<<','<<iy<<"  x,y = "<<x<<','<<y<<std::endl;
            // We should never arrive here, since this means we didn't find it in the undistorted
            // pixel or any of the neighboring pixels.  However, sometimes (about 0.1% of the
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
    for (int p=0; p<nx*ny; p++)
      {
            delete _imagepolys[p];
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

template void Silicon::UpdatePixelDistortions(ImageView<double> target) const;
template void Silicon::UpdatePixelDistortions(ImageView<float> target) const;
  
template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                    ImageView<double> target);
template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                    ImageView<float> target);

} // ends namespace galsim
