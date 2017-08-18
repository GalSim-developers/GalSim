/* -*- c++ -*-
 * Copyright (c) 2012-2017 by the GalSim developers team on GitHub
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
 * Date: Feb 17, 2017
 * Routines for integrating the CCD simulations into GalSim
 */

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

    // Helper function used in a few places below.
    void buildEmptyPoly(Polygon& poly, int numVertices)
    {
        double dtheta = M_PI / (2.0 * (numVertices + 1.0));
        double theta0 = - M_PI / 4.0;

        poly.reserve(numVertices*4 + 4);
        // First the corners
        dbg<<"corners:\n";
        for (int xpix=0; xpix<2; xpix++) {
            for (int ypix=0; ypix<2; ypix++) {
                poly.add(Point(xpix, ypix));
            }
        }
        // Next the edges
        dbg<<"x edges:\n";
        for (int xpix=0; xpix<2; xpix++) {
            for (int n=0; n<numVertices; n++) {
                double theta = theta0 + (n + 1.0) * dtheta;
                poly.add(Point(xpix, (std::tan(theta) + 1.0) / 2.0));
            }
        }
        dbg<<"y edges:\n";
        for (int ypix=0; ypix<2; ypix++) {
            for (int n=0; n<numVertices; n++) {
                double theta = theta0 + (n + 1.0) * dtheta;
                poly.add(Point((std::tan(theta) + 1.0) / 2.0, ypix));
            }
        }
        poly.sort();
    }

    Silicon::Silicon(int numVertices, double numElec, int nx, int ny, int qDist, double nrecalc,
                     double diffStep, double pixelSize, double sensorThickness,
                     double* vertex_data) :
        _numVertices(numVertices), _nx(nx), _ny(ny), _nrecalc(nrecalc),
        _qDist(qDist), _diffStep(diffStep), _pixelSize(pixelSize),
        _sensorThickness(sensorThickness)
    {
        // This constructor reads in the distorted pixel shapes from the Poisson solver
        // and builds an array of polygons for calculating the distorted pixel shapes
        // as a function of charge in the surrounding pixels.

        // First build the distorted polygons. We have an array of nx*ny polygons,
        // an undistorted polygon, and a polygon for test.

        _nv = 4 * _numVertices + 4; // Number of vertices in each pixel

        buildEmptyPoly(_emptypoly, _numVertices);
        _testpoly = _emptypoly;  // This is a mutable Polygon we'll use as scratch space
        _distortions.resize(_nx*_ny);
        for (int i=0; i<(_nx*_ny); ++i)
            _distortions[i] = _emptypoly;  // These will accumulated the distortions over time.

        // Next, we read in the pixel distortions from the Poisson_CCD simulations
        for (int index=0; index < _nv*_nx*_ny; index++) {
            int n = (index % (_ny * _nv)) % _nv;
            int j = (index - n) / _nv;
            int i = (index - n - j * _nv) / (_ny * _nv);
            double x0 = vertex_data[5*index+0];
            double y0 = vertex_data[5*index+1];
            double th = vertex_data[5*index+2];
            double x1 = vertex_data[5*index+3];
            double y1 = vertex_data[5*index+4];
#ifdef DEBUGLOGGING
            if (index == 73) { // Test print out of read in
                dbg<<"Successfully reading the Pixel vertex file\n";
                //dbg<<"line = "<<line<<std::endl;
                dbg<<"n = "<<n<<", i = "<<i<<", j = "<<j<<", x0 = "<<x0<<", y0 = "<<y0
                    <<", th = "<<th<<", x1 = "<<x1<<", y1 = "<<y1<<std::endl;
            }
#endif

            // The following captures the pixel displacement. These are translated into
            // coordinates compatible with (x,y). These are per electron.
            double x = _distortions[i * _ny + j][n].x;
            x = ((x1 - x0) / _pixelSize + 0.5 - x) / numElec;
            _distortions[i * _ny + j][n].x = x;
            double y = _distortions[i * _ny + j][n].y;
            y = ((y1 - y0) / _pixelSize + 0.5 - y) / numElec;
            _distortions[i * _ny + j][n].y = y;
        }
#ifdef DEBUGLOGGING
        //Test print out of distortion for central pixel
        int i = 4;
        int j = 4;
        for (int n=0; n < _nv; n++) {
            xdbg<<"n = "<<n<<", x = "<<_distortions[i * _ny + j][n].x * numElec
                <<", y = "<<_distortions[i * _ny + j][n].y * numElec<<std::endl;
        }
#endif
    }

#if 0
    // Got a start on implementing the absorption length lookup,
    // but couldn't get the python - C++ wrapper working, so I went
    // back to the analytic function - Craig Lage 17-Feb-17
    double Silicon::AbsLength(double lambda)
    {
        // Looks up the absorption length and returns
        // an interpolated value
        int nminus;
        double aminus, aplus, dlambda;
        dlambda = _abs_data[2] - _abs_data[0];
        // index of
        nminus = (int) ((lambda - _abs_data[0]) / dlambda) * 2;
        if (nminus < 0) return _abs_data[1];
        else if (nminus > _nabs - 4) return _abs_data[_nabs - 1];
        else return _abs_data[nminus+1] +
            (lambda - _abs_data[nminus]) * (_abs_data[nminus+3] - _abs_data[nminus+1]);
    }
#endif

    template <typename T>
    void Silicon::updatePixelDistortions(ImageView<T> target)
    {
        // This updates the pixel distortions in the _imagepolys
        // pixel list based on the amount of additional charge in each pixel
        // This distortion assumes the electron is created at the
        // top of the silicon.  It mus be scaled based on the conversion depth
        // This is handled in insidePixel.

        int nxCenter = (_nx - 1) / 2;
        int nyCenter = (_ny - 1) / 2;

        // Now add in the displacements
        int minx = target.getXMin();
        int miny = target.getYMin();
        int maxx = target.getXMax();
        int maxy = target.getYMax();

        // Now we cycle through the pixels in the target image and update any affected
        // pixel shapes.
        std::vector<bool> changed(_imagepolys.size(), false);
        for (int i=minx; i<maxx; ++i) {
            for (int j=miny; j<maxy; ++j) {
                double charge = target(i,j);
                if (charge == 0.0) continue;

                for (int di=-_qDist; di<=_qDist; ++di) {
                    for (int dj=-_qDist; dj<=_qDist; ++dj) {
                        int polyi = i + di;
                        int polyj = j + dj;
                        if ((polyi < minx) || (polyi > maxx) || (polyj < miny) || (polyj > maxy))
                            continue;
                        int index = (polyi - minx) * (maxy - miny + 1) + (polyj - miny);

                        int disti = nxCenter + di;
                        int distj = nyCenter + dj;
                        int dist_index = disti * _ny + distj;
                        for (int n=0; n<_nv; n++) {
                            double dx = _distortions[dist_index][n].x * charge;
                            double dy = _distortions[dist_index][n].y * charge;
                            _imagepolys[index][n].x += dx;
                            _imagepolys[index][n].y += dy;
                        }
                        changed[index] = true;
                    }
                }
            }
        }
        for (size_t k=0; k<_imagepolys.size(); ++k) {
            if (changed[k]) _imagepolys[k].updateBounds();
        }
    }

    template <typename T>
    bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                              ImageView<T> target) const
    {
        // This scales the pixel distortion based on the zconv, which is the depth
        // at which the electron is created, and then tests to see if the delivered
        // point is inside the pixel.
        // (ix,iy) is the pixel being tested, and (x,y) is the coordinate of the
        // photon within the pixel, with (0,0) in the lower left

        // If test pixel is off the image, return false.  (Avoids seg faults!)
        if (!target.getBounds().includes(Position<int>(ix,iy))) return false;

        const int minx = target.getXMin();
        const int miny = target.getYMin();
        const int maxx = target.getXMax();
        const int maxy = target.getYMax();

        int index = (ix - minx) * (maxy - miny + 1) + (iy - miny);

        // First do some easy checks if the point isn't terribly close to the boundary.
        Point p(x,y);
        if (_imagepolys[index].triviallyContains(p)) return true;
        if (!_imagepolys[index].mightContain(p)) return false;

        // OK, it must be near the boundary, so now be careful.
        // The term zfactor decreases the pixel shifts as we get closer to the bottom
        // It is an empirical fit to the Poisson solver simulations, and only matters
        // when we get quite close to the bottom.  This could be more accurate by making
        // the Vertices files have an additional look-up variable (z), but this doesn't
        // seem necessary at this point
        const double zfit = 12.0;
        const double zfactor = std::tanh(zconv / zfit);

        // Scale the testpoly vertices by zfactor
        _testpoly.scale(_imagepolys[index], _emptypoly, zfactor);

        // Now test to see if the point is inside
        return _testpoly.contains(p);
    }

    // Helper function to calculate how far down into the silicon the photon converts into
    // an electron.

    void Silicon::calculateConversionDepth(const PhotonArray& photons, std::vector<double>& depth,
                                           UniformDeviate ud) const
    {
        const double log10_over_250 = std::log(10.) / 250.;

        const int nphotons = photons.size();
        for (int i=0; i<nphotons; ++i) {
            // Determine the distance the photon travels into the silicon
            double si_length;
            if (photons.hasAllocatedWavelengths()) {
                double lambda = photons.getWavelength(i); // in nm
                // The below is an approximation.  ToDo: replace with lookup table
                //double abs_length = pow(10.0,((lambda - 500.0) / 250.0)); // in microns
                double abs_length = std::exp((lambda - 500.0) * log10_over_250); // in microns
                //double abs_length = AbsLength(lambda); // in microns
                si_length = -abs_length * log(1.0 - ud()); // in microns
#ifdef DEBUGLOGGING
                if (i % 1000 == 0) {
                    dbg<<"lambda = "<<lambda<<std::endl;
                    dbg<<"si_length = "<<si_length<<std::endl;
                }
#endif
            } else {
                // If no wavelength info, assume conversion takes place near the top.
                si_length = 1.0;
            }

            // Next we partition the si_length into x,y,z.  Assuming dz is positive downward
            if (photons.hasAllocatedAngles()) {
                double dxdz = photons.getDXDZ(i);
                double dydz = photons.getDYDZ(i);
                double dz = si_length / std::sqrt(1.0 + dxdz*dxdz + dydz*dydz); // in microns
                depth[i] = std::min(_sensorThickness - 1.0, dz);  // max 1 micron from bottom
#ifdef DEBUGLOGGING
                if (i % 1000 == 0) {
                    dbg<<"dxdz = "<<dxdz<<std::endl;
                    dbg<<"dydz = "<<dydz<<std::endl;
                    dbg<<"dz = "<<dz<<std::endl;
                }
#endif
            } else {
                depth[i] = si_length;
            }
        }
    }

    static const int xoff[9] = {0,1,1,0,-1,-1,-1,0,1}; // Displacements to neighboring pixels
    static const int yoff[9] = {0,0,1,1,1,0,-1,-1,-1}; // Displacements to neighboring pixels

    // Break this bit out mostly to make it easier when profiling to see how much it would help
    // to further optimize this part of the code.
    template <typename T>
    bool searchNeighbors(const Silicon& silicon, int& ix, int& iy, double x, double y, double zconv,
                         ImageView<T> target, int& step)
    {
        // The following code finds which pixel we are in given
        // pixel distortion due to the brighter-fatter effect
        // The following are set up to start the search in the undistorted pixel, then
        // search in the nearest neighbor first if it's not in the undistorted pixel.
        if      ((x > y) && (x > 1.0 - y)) step = 1;
        else if ((x < y) && (x < 1.0 - y)) step = 7;
        else if ((x < y) && (x > 1.0 - y)) step = 3;
        else step = 5;
        int n=step;
        for (int m=1; m<9; m++) {
            int ix_off = ix + xoff[n];
            int iy_off = iy + yoff[n];
            double x_off = x - xoff[n];
            double y_off = y - yoff[n];
            if (silicon.insidePixel(ix_off, iy_off, x_off, y_off, zconv, target)) {
                xdbg<<"Found in pixel "<<n<<", ix = "<<ix<<", iy = "<<iy
                    <<", x="<<x<<", y = "<<y<<", target(ix,iy)="<<target(ix,iy)<<std::endl;
                ix = ix_off;
                iy = iy_off;
                return true;
            }
            n = ((n-1) + step) % 8 + 1;
            // This is intended to start with the nearest neighbor, then cycle through others.
        }
        return false;
    }

    template <typename T>
    double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud, ImageView<T> target)
    {
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
        double Irr = 0.;
        double Irr0 = 0.;
        int zerocount=0, neighborcount=0, misscount=0;
#endif

        const int nx = b.getXMax() - b.getXMin() + 1;
        const int ny = b.getYMax() - b.getYMin() + 1;
        const int nxny = nx * ny;
        dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;
        _imagepolys.resize(nxny);
        for (int i=0; i<nxny; ++i)
            _imagepolys[i] = _emptypoly;
        dbg<<"Built poly list\n";

        const double invPixelSize = 1./_pixelSize; // pixels/micron
        const double diffStep_pixel_z = _diffStep / (_sensorThickness * _pixelSize);

        const int nphotons = photons.size();
        std::vector<double> depth(nphotons);
        calculateConversionDepth(photons, depth, ud);

        GaussianDeviate gd(ud,0,1); // Random variable from Standard Normal dist.

        // Start with the correct distortions for the initial image as it is already
        updatePixelDistortions(target);

        // Keep track of the charge we are accumulating on a separate image for efficiency
        // of the distortion updates.
        ImageAlloc<T> delta(b, 0.);

        double addedFlux = 0.;
        double next_recalc = _nrecalc;
        for (int i=0; i<nphotons; i++) {
            // Update shapes every _nrecalc electrons
            if (addedFlux > next_recalc) {
                updatePixelDistortions(delta.view());
                target += delta;
                delta.setZero();
                next_recalc = addedFlux + _nrecalc;
            }

            // Get the location where the photon strikes the silicon:
            double x0 = photons.getX(i); // in pixels
            double y0 = photons.getY(i); // in pixels

            double dz = depth[i];  // microns
            if (photons.hasAllocatedAngles()) {
                double dxdz = photons.getDXDZ(i);
                double dydz = photons.getDYDZ(i);
                double dz_pixel = dz * invPixelSize;
                x0 += dxdz * dz_pixel; // dx in pixels
                y0 += dydz * dz_pixel; // dy in pixels
            }
            // This is the reverse of depth. zconv is how far above the substrate the e- converts.
            double zconv = _sensorThickness - dz;
            if (zconv < 0.0) continue; // Throw photon away if it hits the bottom
            // TODO: Do something more realistic if it hits the bottom.

            // Now we add in a displacement due to diffusion
            if (_diffStep != 0.) {
                double diffStep = std::max(0.0, diffStep_pixel_z * (zconv - 10.0));
                x0 += diffStep * gd();
                y0 += diffStep * gd();
            }
            double flux = photons.getFlux(i);

#ifdef DEBUGLOGGING
            if (i % 1000 == 0) {
                xdbg<<"diffStep = "<<diffStep<<std::endl;
                xdbg<<"zconv = "<<zconv<<std::endl;
                xdbg<<"x0 = "<<x0<<std::endl;
                xdbg<<"y0 = "<<y0<<std::endl;
            }
#endif

            // Now we find the undistorted pixel
            int ix = int(floor(x0 + 0.5));
            int iy = int(floor(y0 + 0.5));

#ifdef DEBUGLOGGING
            int ix0 = ix;
            int iy0 = iy;
#endif

            double x = x0 - ix + 0.5;
            double y = y0 - iy + 0.5;
            // (ix,iy) are the undistorted pixel coordinates.
            // (x,y) are the coordinates within the pixel, centered at the lower left

            // First check the obvious choice, since this will usually work.
            bool foundPixel = insidePixel(ix, iy, x, y, zconv, target);
#ifdef DEBUGLOGGING
            if (foundPixel) ++zerocount;
#endif

            // Then check neighbors
            int step;  // We might need this below, so let searchNeighbors return it.
            if (!foundPixel) {
                foundPixel = searchNeighbors(*this, ix, iy, x, y, zconv, target, step);
#ifdef DEBUGLOGGING
                if (foundPixel) ++neighborcount;
#endif
            }

            // Rarely, we won't find it in the undistorted pixel or any of the neighboring pixels.
            // If we do arrive here due to roundoff error of the pixel boundary, put the electron
            // in the undistorted pixel or the nearest neighbor with equal probability.
            if (!foundPixel) {
                xdbg<<"Not found in any pixel\n";
                xdbg<<"ix,iy = "<<ix<<','<<iy<<"  x,y = "<<x<<','<<y<<std::endl;
                int n = (ud() > 0.5) ? 0 : step;
                ix = ix + xoff[n];
                iy = iy + yoff[n];
#ifdef DEBUGLOGGING
                ++misscount;
#endif
            }
#if 0
            // (ix, iy) now give the actual pixel which will receive the charge
            if (ix != ix0 || iy != iy0) {
                dbg<<"("<<ix0<<","<<iy0<<") -> ("<<ix<<","<<iy<<")\n";
                double r0 = std::sqrt((ix0+0.5)*(ix0+0.5)+(iy0+0.5)*(iy0+0.5));
                double r = std::sqrt((ix+0.5)*(ix+0.5)+(iy+0.5)*(iy+0.5));
                dbg<<"r = "<<r0<<" -> "<<r;
                if (r < r0) { dbg<<"  *****"; }
                dbg<<"\nstep = "<<step<<", n = "<<n<<", m_found = "<<m_found<<std::endl;
                dbg<<"flux = "<<photons.getFlux(i)<<std::endl;
            }
#endif

            if (b.includes(ix,iy)) {
#ifdef DEBUGLOGGING
                double rsq = (ix+0.5)*(ix+0.5)+(iy+0.5)*(iy+0.5);
                Irr += flux * rsq;
                rsq = (ix0+0.5)*(ix0+0.5)+(iy0+0.5)*(iy0+0.5);
                Irr0 += flux * rsq;
#endif
                delta(ix,iy) += flux;
                addedFlux += flux;
            }
        }
        // No need to update the distortions again, but we do need to add the delta image.
        target += delta;

#ifdef DEBUGLOGGING
        Irr /= addedFlux;
        Irr0 /= addedFlux;
        dbg<<"Irr = "<<Irr<<"  cf. Irr0 = "<<Irr0<<std::endl;
        dbg << "Found "<< zerocount << " photons in undistorted pixel, " << neighborcount;
        dbg << " in one of the neighbors, and "<<misscount;
        dbg << " not in any pixel\n" << std::endl;
#endif
        return addedFlux;
    }

    //template double Silicon::AbsLength(double lambda) const;

    template bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                                       ImageView<double> target) const;
    template bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                                       ImageView<float> target) const;

    template void Silicon::updatePixelDistortions(ImageView<double> target);
    template void Silicon::updatePixelDistortions(ImageView<float> target);

    template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                        ImageView<double> target);
    template double Silicon::accumulate(const PhotonArray& photons, UniformDeviate ud,
                                        ImageView<float> target);

} // namespace galsim
