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
 * Date: Mar 14, 2016
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
        dbg<<"n = "<<n<<", x = "<<polylist[i * Ny + j]->pointlist[n]->x*(double)NumElec
            <<", y = "<<polylist[i * Ny + j]->pointlist[n]->y*(double)NumElec<<std::endl;
#endif
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

#ifdef DEBUGLOGGING
    for (n=0; n<Nv; n++) // test printout of distorted pixe vertices.
        dbg<<"n = "<<n<<", x = "<<polylist[TestPoly]->pointlist[n]->x
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

double Silicon::random_gaussian(void)
{
    // Copied from PhoSim
    // - SHOULD NOT BE USED!
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

template <typename T>
double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                           ImageView<T> target) const
{
    // Modified by Craig Lage - UC Davis to incorporate the brighter-fatter effect
    // 16-Mar-16
    // 'silicon' must be passed to the function, optionally
    // Silicon* silicon = new Silicon("../poisson/BF_256_9x9_0_Vertices"); // Create and read in pixel distortions
    bool FoundPixel;
    int xoff[9] = {0,1,1,0,-1,-1,-1,0,1};// Displacements to neighboring pixels
    int yoff[9] = {0,0,1,1,1,0,-1,-1,-1};// Displacements to neighboring pixels
    int n=0, step, ix_off, iy_off;
    double x, y, x_off, y_off;
    double zconv = 95.0; // Z coordinate of photoconversion in microns
    // Will add more detail later
    double ccdtemp =  173; // CCD temp in K <- THIS SHOULD COME FROM silicon
    double DiffStep; // Mean diffusion step size in microns
    GaussianDeviate gd(ud,0,1); // Random variable from Standard Normal dist.

    if (zconv <= 10.0)
    { DiffStep = 0.0; }
    else
    { DiffStep = this->DiffStep * (zconv - 10.0) / 100.0 * sqrt(ccdtemp / 173.0); }

    int zerocount = 0, nearestcount = 0, othercount = 0, misscount = 0;

    Bounds<int> b = target.getBounds();

    if (!b.isDefined())
        throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                 " undefined Bounds");

    // Factor to turn flux into surface brightness in an Image pixel
    dbg<<"In PhotonArray::addTo\n";
    dbg<<"bounds = "<<b<<std::endl;

    double addedFlux = 0.;
    for (int i=0; i<int(photons.size()); i++)
    {
        int ix, iy;
        // First we add in a displacement due to diffusion
        x = photons.getX(i) + DiffStep * gd() / 10.0;
        y = photons.getY(i) + DiffStep * gd() / 10.0;

        // Now we find the undistorted pixel
        ix = int(floor(x + 0.5));
        iy = int(floor(y + 0.5));

        int n=0, step, ix_off, iy_off;
        x = x - (double) ix + 0.5;
        y = y - (double) iy + 0.5;
        // (ix,iy) are the undistorted pixel coordinates.
        // (x,y) are the coordinates within the pixel, centered at the lower left - CRAIG, BETTER TO
        // CALL IT (dx,dy)

        // The following code finds which pixel we are in given
        // pixel distortion due to the brighter-fatter effect
        FoundPixel = false;
        // The following are set up to start the search in the undistorted pixel, then
        // search in the nearest neighbor first if it's not in the undistorted pixel.
        if      ((x > y) && (x > 1.0 - y)) step = 1;
        else if ((x > y) && (x < 1.0 - y)) step = 7;
        else if ((x < y) && (x > 1.0 - y)) step = 3;
        else step = 5;
        for (int m=0; m<9; m++)
        {
            ix_off = ix + xoff[n];
            iy_off = iy + yoff[n];
            x_off = x - (double)xoff[n];
            y_off = y - (double)yoff[n];
            if (this->InsidePixel(ix_off, iy_off, x_off, y_off, zconv, target))
            {
                xdbg<<"Found in pixel "<<n<<", ix = "<<ix<<", iy = "<<iy
                    <<", x="<<x<<", y = "<<y<<", target(ix,iy)="<<target(ix,iy)<<std::endl;
                if (m == 0) zerocount += 1;
                else if (m == 1) nearestcount += 1;
                else othercount +=1;
                ix = ix_off;
                iy = iy_off;
                FoundPixel = true;
                break;
            }
            n = ((n-1)+step) % 8 + 1;
            // This is intended to start with the nearest neighbor, then cycle through the others.
        }
        if (!FoundPixel)
        {
            // We should never arrive here, since this means we didn't find it in the undistorted pixel
            // or any of the neighboring pixels.  However, sometimes (about 0.01% of the time) we do
            // arrive here due to roundoff error of the pixel boundary.  When this happens, I put
            // the electron in the undistorted pixel or the nearest neighbor with equal probability.
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
            dbg<<"Not found in any pixel\n";
        }
        // (ix, iy) now give the actual pixel which will receive the charge

        if (b.includes(ix,iy)) {
            target(ix,iy) += photons.getFlux(i);
            addedFlux += photons.getFlux(i);
        }
    }
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
