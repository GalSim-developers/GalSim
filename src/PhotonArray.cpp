/* -*- c++ -*-
 * Copyright (c) 2012-2015 by the GalSim developers team on GitHub
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
//
// PhotonArray Class members
//

//#define DEBUGLOGGING

#include <algorithm>
#include <numeric>
#include "PhotonArray.h"
#include "silicon.h"

#ifdef DEBUGLOGGING
#include <fstream>
//std::ostream* dbgout = new std::ofstream("debug.out");
//int verbose_level = 2;
#endif

namespace galsim {

    PhotonArray::PhotonArray(
        std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vflux) :
        _is_correlated(false)
    {
        if (vx.size() != vy.size() || vx.size() != vflux.size())
            throw std::runtime_error("Size mismatch of input vectors to PhotonArray");
        _x = vx;
        _y = vy;
        _flux = vflux;
    }

    double PhotonArray::getTotalFlux() const 
    {
        double total = 0.;
        return std::accumulate(_flux.begin(), _flux.end(), total);
    }

    void PhotonArray::setTotalFlux(double flux) 
    {
        double oldFlux = getTotalFlux();
        if (oldFlux==0.) return; // Do nothing if the flux is zero to start with
        scaleFlux(flux / oldFlux);
    }

    void PhotonArray::scaleFlux(double scale)
    {
        for (std::vector<double>::size_type i=0; i<_flux.size(); i++) {
            _flux[i] *= scale;
        }
    }

    void PhotonArray::scaleXY(double scale)
    {
        for (std::vector<double>::size_type i=0; i<_x.size(); i++) {
            _x[i] *= scale;
        }
        for (std::vector<double>::size_type i=0; i<_y.size(); i++) {
            _y[i] *= scale;
        }
    }

    void PhotonArray::append(const PhotonArray& rhs) 
    {
        if (rhs.size()==0) return;      // Nothing needed for empty RHS.
        int oldSize = size();
        int finalSize = oldSize + rhs.size();
        _x.resize(finalSize);
        _y.resize(finalSize);
        _flux.resize(finalSize);
        std::vector<double>::iterator destination=_x.begin()+oldSize;
        std::copy(rhs._x.begin(), rhs._x.end(), destination);
        destination=_y.begin()+oldSize;
        std::copy(rhs._y.begin(), rhs._y.end(), destination);
        destination=_flux.begin()+oldSize;
        std::copy(rhs._flux.begin(), rhs._flux.end(), destination);
    }

    void PhotonArray::convolve(const PhotonArray& rhs, UniformDeviate ud) 
    {
        // If both arrays have correlated photons, then we need to shuffle the photons
        // as we convolve them.
        if (_is_correlated && rhs._is_correlated) return convolveShuffle(rhs,ud);

        // If neither or only one is correlated, we are ok to just use them in order.
        int N = size();
        if (rhs.size() != N) 
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        // Add x coordinates:
        std::vector<double>::iterator lIter = _x.begin();
        std::vector<double>::const_iterator rIter = rhs._x.begin();
        for ( ; lIter!=_x.end(); ++lIter, ++rIter) *lIter += *rIter;
        // Add y coordinates:
        lIter = _y.begin();
        rIter = rhs._y.begin();
        for ( ; lIter!=_y.end(); ++lIter, ++rIter) *lIter += *rIter;
        // Multiply fluxes, with a factor of N needed:
        lIter = _flux.begin();
        rIter = rhs._flux.begin();
        for ( ; lIter!=_flux.end(); ++lIter, ++rIter) *lIter *= *rIter*N;

        // If rhs was correlated, then the output will be correlated.
        // This is ok, but we need to mark it as such.
        if (rhs._is_correlated) _is_correlated = true;
    }

    void PhotonArray::convolveShuffle(const PhotonArray& rhs, UniformDeviate ud) 
    {
        int N = size();
        if (rhs.size() != N) 
            throw std::runtime_error("PhotonArray::convolve with unequal size arrays");
        double xSave=0.;
        double ySave=0.;
        double fluxSave=0.;

        for (int iOut = N-1; iOut>=0; iOut--) {
            // Randomly select an input photon to use at this output
            // NB: don't need floor, since rhs is positive, so floor is superfluous.
            int iIn = int((iOut+1)*ud());
            if (iIn > iOut) iIn=iOut;  // should not happen, but be safe
            if (iIn < iOut) {
                // Save input information
                xSave = _x[iOut];
                ySave = _y[iOut];
                fluxSave = _flux[iOut];
            }
            _x[iOut] = _x[iIn] + rhs._x[iOut];
            _y[iOut] = _y[iIn] + rhs._y[iOut];
            _flux[iOut] = _flux[iIn] * rhs._flux[iOut] * N;
            if (iIn < iOut) {
                // Move saved info to new location in array
                _x[iIn] = xSave;
                _y[iIn] = ySave ;
                _flux[iIn] = fluxSave;
            }
        }
    }

    void PhotonArray::takeYFrom(const PhotonArray& rhs) 
    {
        int N = size();
        assert(rhs.size()==N);
        for (int i=0; i<N; i++) {
            _y[i] = rhs._x[i];
            _flux[i] *= rhs._flux[i]*N;
        }
    }

    template <class T>
    double PhotonArray::addTo(ImageView<T>& target) const 
    {
      // Modified by Craig Lage - UC Davis to incorporate the brighter-fatter effect
      // 16-Mar-16
      Silicon* silicon = new Silicon("../poisson/BF_256_9x9_0_Vertices"); // Create and read in pixel distortions
      bool FoundPixel;
      int xoff[9] = {0,1,1,0,-1,-1,-1,0,1};// Displacements to neighboring pixels
      int yoff[9] = {0,0,1,1,1,0,-1,-1,-1};// Displacements to neighboring pixels
      int n=0, step, ix_off, iy_off;
      double x, y, x_off, y_off;
      double zconv = 95.0; // Z coordinate of photoconversion in microns
                           // Will add more detail later
      double ccdtemp = 173; // CCD temp in K
      double DiffStep; // Mean diffusion step size in microns
      if (zconv <= 10.0)
	{
	  DiffStep = 0.0;
	}
      else
	{
	  DiffStep = silicon->DiffStep * (zconv - 10.0) / 100.0 * sqrt(ccdtemp / 173.0);
	}
      
      Bounds<int> b = target.getBounds();

        if (!b.isDefined()) 
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        // Factor to turn flux into surface brightness in an Image pixel
        dbg<<"In PhotonArray::addTo\n";
        dbg<<"bounds = "<<b<<std::endl;

        double addedFlux = 0.;
#ifdef DEBUGLOGGING
        double totalFlux = 0.;
        double lostFlux = 0.;
        int nx = target.getXMax()-target.getXMin()+1;
        int ny = target.getYMax()-target.getYMin()+1;
        std::vector<std::vector<double> > posFlux(nx,std::vector<double>(ny,0.));
        std::vector<std::vector<double> > negFlux(nx,std::vector<double>(ny,0.));
#endif
	int zerocount = 0, nearestcount = 0, othercount = 0, misscount = 0;
        for (int i=0; i<int(size()); i++)
	  {
	    double x, y, x_off, y_off;
	    bool FoundPixel;
	    int ix, iy;
	    // First we add in a displacement due to diffusion
	    x = _x[i] + DiffStep * silicon->random_gaussian() / 10.0;
	    y = _y[i] + DiffStep * silicon->random_gaussian() / 10.0;
	    // Now we find the undistorted pixel
	    ix = int(floor(x + 0.5));
            iy = int(floor(y + 0.5));
	    int n=0, step, ix_off, iy_off;
	    x = x - (double) ix + 0.5;
	    y = y - (double) iy + 0.5;
	    // (ix,iy) are the undistorted pixel coordinates.
	    // (x,y) are the coordinates within the pixel, centered at the lower left

	    // The following code finds which pixel we are in given
	    // pixel distortion due to the brighter-fatter effect
	    FoundPixel = false;
	    // The following are set up to start the search in the undistorted pixel, then
	    // search in the nearest neighbor first if it's not in the undistorted pixel.
	    if      ((x > y) && (x > 1.0 - y)) step = 1;
	    else if ((x > y) && (x < 1.0 - y)) step = 7;
	    else if ((x < y) && (x > 1.0 - y)) step = 3;
	    else                                               step = 5;
	    for (int m=0; m<9; m++)
	      {
		ix_off = ix + xoff[n];
		iy_off = iy + yoff[n];	
		x_off = x - (double)xoff[n];
		y_off = y - (double)yoff[n];
		if (silicon->InsidePixel(ix_off, iy_off, x_off, y_off, zconv, (ImageView<float>&)target))
		  {
		    //printf("Found in pixel %d, ix = %d, iy = %d, x=%f, y = %f, target(ix,iy)=%f\n",n, ix, iy, x, y, target(ix,iy));
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
		if (drand48() > 0.5)
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
		//printf("Not found in any pixel\n");
		}
		// (ix, iy) now give the actual pixel which will receive the charge

#ifdef DEBUGLOGGING
            totalFlux += _flux[i];
            xdbg<<"  photon: ("<<_x[i]<<','<<_y[i]<<")  f = "<<_flux[i]<<std::endl;
#endif
            if (b.includes(ix,iy)) {
#ifdef DEBUGLOGGING
                if (_flux[i] > 0.) posFlux[ix-target.getXMin()][iy-target.getXMin()] += _flux[i];
                else negFlux[ix-target.getXMin()][iy-target.getXMin()] -= _flux[i];
#endif
                target(ix,iy) += _flux[i];
                addedFlux += _flux[i];
            } else {
#ifdef DEBUGLOGGING
                xdbg<<"lost flux at ix = "<<ix<<", iy = "<<iy<<" with flux = "<<_flux[i]<<std::endl;
                lostFlux += _flux[i];
#endif
            }
        }
#ifdef DEBUGLOGGING
        dbg<<"totalFlux = "<<totalFlux<<std::endl;
        dbg<<"addedlFlux = "<<addedFlux<<std::endl;
        dbg<<"lostFlux = "<<lostFlux<<std::endl;
        for(int ix=0;ix<nx;++ix) {
            for(int iy=0;iy<ny;++iy) {
                double pos = posFlux[ix][iy];
                double neg = negFlux[ix][iy];
                double tot = pos + neg;
                if (tot > 0.) {
                    xdbg<<"eta("<<ix+target.getXMin()<<','<<iy+target.getXMin()<<") = "<<
                        neg<<" / "<<tot<<" = "<<neg/tot<<std::endl;
                }
            }
        }
#endif
	// These counts are mainly for debug purposes and can be removed later.
	printf("Found %d photons in undistorted pixel, %d in closest neighbor, %d in other neighbor. %d not in any pixel\n",zerocount, nearestcount, othercount, misscount);
	delete silicon;
        return addedFlux;
    }

    // instantiate template functions for expected image types
    template double PhotonArray::addTo(ImageView<float>& image) const;
    template double PhotonArray::addTo(ImageView<double>& image) const;

}
