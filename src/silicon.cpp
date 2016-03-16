/*
  ------------------------------------------------------------------------------
  Author: Craig Lage, UC Davis
  Date: Mar 14, 2016
  Routines for integrating the CCD simulations into GalSim
*/

//****************** silicon.cpp **************************

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <algorithm>

#include "silicon.h"
#include "Image.h"

namespace galsim {

Silicon::Silicon (std::string inname)
{
    // This consructor reads in the distorted pixel shapes from the Poisson solver
    // and builds an array of polygons for calculating the distorted pixel shapes
    // as a function of charge in the surrounding pixels.
    int DistributedCharge;
    double PixelSize, ChannelStopWidth;
    double Vbb, Vparallel_lo, Vparallel_hi, Vdiff=50.0;

    // These values need to be read in from the bf.cfg Poisson solver
    // configuration file, but for now I am hard coding them.
    NumVertices = 8; // Number per edge
    NumElec = 160000; // Number of electrons in central pixel in Poisson simulation
    Nx = 9; // Size of Poisson postage stamp
    Ny = 9; // Size of Poisson postage stamp
    DistributedCharge = 2; // Number of collecting gates
    PixelSize = 10.0; // Pixel size in microns
    ChannelStopWidth = 2.0; // in microns
    Vbb = -65.0; // Back-bias voltage
    Vparallel_lo = -8.0; // Barrier gate volatge
    Vparallel_hi = 4.0; // Collecting gate voltage
      
    // Set up the collection area and the diffusion step size at 100 C
    collXmin = ChannelStopWidth / (2.0 * PixelSize);
    collXwidth = (PixelSize - ChannelStopWidth) / PixelSize;
    if (DistributedCharge == 1)
      {
	// This is one collecting gate
	collYmin = 1.0 / 3.0;
	collYwidth = 1.0 / 3.0;
	Vdiff = (2.0 * Vparallel_lo + Vparallel_hi) / 3.0 - Vbb;
      }
    else if (DistributedCharge == 2)
      {
	// This is two collecting gates
	collYmin = 1.0 / 6.0;
	collYwidth = 2.0 / 3.0;
	Vdiff = (Vparallel_lo + 2.0 * Vparallel_hi) / 3.0 - Vbb;	
      }
    else
      {
	printf("Error setting collecting region");
      }

    // The following is the lateral diffusion step size in microns, assuming the entire silicon thickness (100 microns)
    // and -100 C temperature.  We adjust it for deviations from this in photonmanipulate.cpp

    DiffStep = 100.0 * sqrt(2.0 * 0.015 / Vdiff);
    
    /* Next, we build the polygons. We have an array of Nx*Ny + 2 polygons,
       with 0 to Nx*Ny-1 holding the pixel distortions, Nx*Ny being the polygon
       being tested, and Nx*Ny+1 being an undistorted polygon.  First, we build all
       of them as undistorted polygons. */

    int NumPolys;
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
    std::ifstream infile;
    infile.open(inname);
    std::string line;
    std::getline(infile,line);// Skip the first line, which is column labels
    for (index=0; index<Nv*(NumPolys - 2); index++)
      {
	std::getline(infile,line);
	std::istringstream iss(line);
	n = (index % (Ny * Nv)) % Nv;
	j = (index - n) / Nv;
	i = (index - n - j * Nv) / (Ny * Nv);
	iss>>x0>>y0>>th>>x1>>y1;
	/*
	if (index == 73) // Test print out of read in 
	  {
	    printf("Successfully reading the Pixel vertex file\n");
	    printf("line = %s\n",line.c_str());
	    printf("n = %d, i = %d, j = %d, x0 = %f, y0 = %f, th = %f, x1 = %f, y1 = %f\n",n,i,j,x0,y0,th,x1,y1);
	  }
	*/
	
	// The following captures the pixel displacement. These are translated into
	// coordinates compatible with (x,y). These are per electron.
	polylist[i * Ny + j]->pointlist[n]->x = (((x1 - x0) / PixelSize + 0.5) - polylist[i * Ny + j]->pointlist[n]->x) / (double)NumElec;
	polylist[i * Ny + j]->pointlist[n]->y = (((y1 - y0) / PixelSize + 0.5) - polylist[i * Ny + j]->pointlist[n]->y) / (double)NumElec;	
      }
    //Test print out of distortion for central pixel
    /*
    i = 4; j = 4;
    for (n=0; n<Nv; n++)
      {
	printf("n = %d, x = %f, y = %f\n",n,polylist[i * Ny + j]->pointlist[n]->x*(double)NumElec,polylist[i * Ny + j]->pointlist[n]->y*(double)NumElec);
	fflush(stdout);
	}*/
    // We generate a testpoint for testing whether a point is inside or outside the array 
    testpoint = new Point(0.0,0.0);
    return;
}

Silicon::~Silicon () {
  int p, NumPolys = Nx * Ny + 2;
  for (p=0; p<NumPolys; p++)
      {
	delete polylist[p];
      }
  delete[] polylist;
}

  bool Silicon::InsidePixel(int ix, int iy, double x, double y, double zconv, ImageView<float>& target)
{
  // This builds the polygon under test based on the charge in the nearby pixels
  // and tests to see if the delivered position is inside it.
  // (ix,iy) is the pixel being tested, and (x,y) is the coordiante of the
  // photon within the pixel, with (0,0) in the lower left
  
  int TestPoly, EmptyPoly, i, j, ii, jj, chargei, chargej, n; 
  double charge;
  int NxCenter, NyCenter, QDist = 3; // QDist is how many pixels away to include neighboring charge.
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
  //printf("minx = %d, miny = %d, maxx = %d, maxy = %d\n",minx,miny,maxx,maxy);
  //printf("ix = %d, iy = %d, target(ix,iy) = %f, x = %f, y = %f\n",ix, iy, target(ix,iy), x, y);
  for (i=NxCenter-QDist; i<NxCenter+QDist+1; i++)
    {
      for (j=NyCenter-QDist; j<NyCenter+QDist+1; j++)
	{
	  chargei = ix + i - NxCenter;
	  chargej = iy + j - NyCenter;	  
	  if ((chargei < minx) || (chargei > maxx) || (chargej < miny) || (chargej > maxy)) continue;
	  charge = target(chargei,chargej);
	  if (charge < 10.0)
	    {
	      continue;
	    }
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

  /*
  for (n=0; n<Nv; n++) // test printout of distorted pixe vertices.
    {
      printf("n = %d, x = %f, y = %f\n",n,polylist[TestPoly]->pointlist[n]->x,polylist[TestPoly]->pointlist[n]->y);
      fflush(stdout);
      }*/

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


  double Silicon::random_gaussian(void)
  {
    // Copied from PhoSim
    double u1 = 0.0, u2 = 0.0, v1 = 0.0, v2 = 0.0, s = 2.0;
    while(s >= 1)
      {
	u1 = drand48();
	u2 = drand48();
	v1 = 2.0*u1-1.0;
	v2 = 2.0*u2-1.0;
	s = pow(v1,2) + pow(v2,2);
      }
    double x1 = v1*sqrt((-2.0*log(s))/s);
    return x1;
  }

  
} // ends namespace galsim
