/* -*- c++ -*-
 * Copyright (c) 2012-2021 by the GalSim developers team on GitHub
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

#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <climits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Uncomment this for debugging output
//#define DEBUGLOGGING

#include "Std.h"
#include "Silicon.h"
#include "Image.h"
#include "PhotonArray.h"


namespace galsim {

    // Helper function used in a few places below.
    void buildEmptyPoly(Polygon& poly, int numVertices)
    {
        dbg<<"buildEmptyPoly\n";
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
                     double diffStep, double pixelSize,
                     double sensorThickness, double* vertex_data,
                     const Table& tr_radial_table, Position<double> treeRingCenter,
                     const Table& abs_length_table, bool transpose) :
        _numVertices(numVertices), _nx(nx), _ny(ny), _qDist(qDist),
        _nrecalc(nrecalc), _diffStep(diffStep), _pixelSize(pixelSize),
        _sensorThickness(sensorThickness),
        _tr_radial_table(tr_radial_table), _treeRingCenter(treeRingCenter),
        _abs_length_table(abs_length_table), _transpose(transpose), _resume_next_recalc(-999)
    {
        dbg<<"Silicon constructor\n";
        // This constructor reads in the distorted pixel shapes from the Poisson solver
        // and builds an array of polygons for calculating the distorted pixel shapes
        // as a function of charge in the surrounding pixels.

        // First build the distorted polygons. We have an array of nx*ny polygons,
        // an undistorted polygon, and a polygon for test.

        _nv = 4 * _numVertices + 4; // Number of vertices in each pixel
        dbg<<"_numVertices = "<<_numVertices<<", _nv = "<<_nv<<std::endl;
        dbg<<"nx,ny = "<<nx<<", "<<ny<<"  ntot = "<<nx*ny<<std::endl;
        dbg<<"total memory = "<<nx*ny*_nv*sizeof(Point)/(1024.*1024.)<<" MBytes"<<std::endl;

        buildEmptyPoly(_emptypoly, _numVertices);
        // These are mutable Polygons we'll use as scratch space
        int numThreads = 1;
#ifdef _OPENMP
        numThreads = omp_get_max_threads();
#endif
        for (int i=0; i < numThreads; i++) {
            _testpoly.push_back(_emptypoly);
        }
        _distortions.resize(_nx*_ny);
        for (int i=0; i<(_nx*_ny); ++i)
            _distortions[i] = _emptypoly;  // These will accumulated the distortions over time.

        // Next, we read in the pixel distortions from the Poisson_CCD simulations
        if (_transpose) std::swap(_nx,_ny);

        for (int index=0; index < _nv*_nx*_ny; index++) {
            xdbg<<"index = "<<index<<std::endl;
            int n = index % _nv;
            int j = (index / _nv) % _ny;
            int i = index / (_nv * _ny);
            xassert(index == (i * _ny + j) * _nv + n);
            double x0 = vertex_data[5*index+0];
            double y0 = vertex_data[5*index+1];
            double x1 = vertex_data[5*index+3];
            double y1 = vertex_data[5*index+4];
            if (_transpose) {
                xdbg<<"Original i,j,n = "<<i<<','<<j<<','<<n<<std::endl;
                std::swap(i,j);
                std::swap(x0,y0);
                std::swap(x1,y1);
                n = (_numVertices - n + _nv) % _nv;
                xdbg<<"    => "<<i<<','<<j<<','<<n<<std::endl;
            }

            // The following captures the pixel displacement. These are translated into
            // coordinates compatible with (x,y). These are per electron.
            double x = _distortions[i * _ny + j][n].x;
            x = ((x1 - x0) / _pixelSize + 0.5 - x) / numElec;
            _distortions[i * _ny + j][n].x = x;
            double y = _distortions[i * _ny + j][n].y;
            y = ((y1 - y0) / _pixelSize + 0.5 - y) / numElec;
            _distortions[i * _ny + j][n].y = y;
#if 0
            if (index == 73) { // Test print out of read in
                dbg<<"Successfully reading the Pixel vertex file\n";
                //dbg<<"line = "<<line<<std::endl;
                dbg<<"n = "<<n<<", i = "<<i<<", j = "<<j<<", x0 = "<<x0<<", y0 = "<<y0
                    <<", th = "<<th<<", x1 = "<<x1<<", y1 = "<<y1<<std::endl;
                dbg<<"x,y = "<<x<<','<<y<<std::endl;
            }
#endif
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

    template <typename T>
    void Silicon::updatePixelDistortions(ImageView<T> target)
    {
        dbg<<"updatePixelDistortions\n";
        // This updates the pixel distortions in the _imagepolys
        // pixel list based on the amount of additional charge in each pixel
        // This distortion assumes the electron is created at the
        // top of the silicon.  It mus be scaled based on the conversion depth
        // This is handled in insidePixel.

        int nxCenter = (_nx - 1) / 2;
        int nyCenter = (_ny - 1) / 2;

        // Now add in the displacements
        const int i1 = target.getXMin();
        const int i2 = target.getXMax();
        const int j1 = target.getYMin();
        const int j2 = target.getYMax();
        const int ny = j2-j1+1;
        const int step = target.getStep();

        // Now we cycle through the pixels in the target image and update any affected
        // pixel shapes.
        std::vector<bool> changed(_imagepolys.size(), false);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j=j1; j<=j2; ++j) {
            const T* ptr = target.getData();
            ptr += (j-j1) * target.getStride();
            for (int i=i1; i<=i2; ++i, ptr+=step) {
                double charge = *ptr;
                if (charge == 0.0) continue;

                int polyi1 = std::max(i - _qDist, i1);
                int polyi2 = std::min(i + _qDist, i2);
                int polyj1 = std::max(j - _qDist, j1);
                int polyj2 = std::min(j + _qDist, j2);
                int disti = nxCenter + polyi1 - i;

                for (int polyi=polyi1; polyi<=polyi2; ++polyi, ++disti) {
                    int distj = nyCenter + polyj1 - j;
                    int index = (polyi - i1) * ny + (polyj1 - j1);
                    int dist_index = disti * _ny + distj;

                    for (int polyj=polyj1; polyj<=polyj2; ++polyj, ++distj, ++index, ++dist_index) {
                        Polygon& distortion = _distortions[dist_index];
                        Polygon& imagepoly = _imagepolys[index];
                        imagepoly.distort(distortion, charge);
                        changed[index] = true;
                    }
                }
            }
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t k=0; k<_imagepolys.size(); ++k) {
            if (changed[k]) _imagepolys[k].updateBounds();
        }
    }

    void Silicon::calculateTreeRingDistortion(int i, int j, Position<int> orig_center,
                                              Polygon& poly) const
    {
        double shift = 0.0;
        for (int n=0; n<_nv; n++) {
            xdbg<<"i,j,n = "<<i<<','<<j<<','<<n<<": x,y = "<<
                poly[n].x <<"  "<< poly[n].y<<std::endl;
            double tx = (double)i + poly[n].x - _treeRingCenter.x + (double)orig_center.x;
            double ty = (double)j + poly[n].y - _treeRingCenter.y + (double)orig_center.y;
            xdbg<<"tx,ty = "<<tx<<','<<ty<<std::endl;
            double r = sqrt(tx * tx + ty * ty);
            shift = _tr_radial_table.lookup(r);
            xdbg<<"r = "<<r<<", shift = "<<shift<<std::endl;
            // Shifts are along the radial vector in direction of the doping gradient
            double dx = shift * tx / r;
            double dy = shift * ty / r;
            xdbg<<"dx,dy = "<<dx<<','<<dy<<std::endl;
            poly[n].x += dx;
            poly[n].y += dy;
            xdbg<<"    x,y => "<<poly[n].x <<"  "<< poly[n].y;
        }
    }

    template <typename T>
    void Silicon::addTreeRingDistortions(ImageView<T> target, Position<int> orig_center)
    {
        if (_tr_radial_table.size() == 2) {
            //dbg<<"Trivial radial table\n";
            // The no tree rings case is indicated with a table of size 2, which
            // wouldn't make any sense as a user input.
            return;
        }
        dbg<<"addTreeRings\n";
        // This updates the pixel distortions in the _imagepolys
        // pixel list based on a model of tree rings.
        // The coordinates _treeRingCenter are the coordinates
        // of the tree ring center, shifted to compensate for the
        // fact that target has its origin shifted to (0,0).
        Bounds<int> b = target.getBounds();
        const int i1 = b.getXMin();
        const int i2 = b.getXMax();
        const int j1 = b.getYMin();
        const int j2 = b.getYMax();
        const int ny = j2-j1+1;
        // Now we cycle through the pixels in the target image and add
        // the (small) distortions due to tree rings
        std::vector<bool> changed(_imagepolys.size(), false);
        for (int i=i1; i<=i2; ++i) {
            for (int j=j1; j<=j2; ++j) {
                int index = (i - i1) * ny + (j - j1);
                calculateTreeRingDistortion(i, j, orig_center, _imagepolys[index]);
                changed[index] = true;
            }
        }
        for (size_t k=0; k<_imagepolys.size(); ++k) {
            if (changed[k]) _imagepolys[k].updateBounds();
        }
    }

    template <typename T>
    bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                              ImageView<T> target, bool* off_edge) const
    {
        // This scales the pixel distortion based on the zconv, which is the depth
        // at which the electron is created, and then tests to see if the delivered
        // point is inside the pixel.
        // (ix,iy) is the pixel being tested, and (x,y) is the coordinate of the
        // photon within the pixel, with (0,0) in the lower left

        // If test pixel is off the image, return false.  (Avoids seg faults!)
        if (!target.getBounds().includes(Position<int>(ix,iy))) {
            if (off_edge) *off_edge = true;
            return false;
        }
        xdbg<<"insidePixel: "<<ix<<','<<iy<<','<<x<<','<<y<<','<<off_edge<<std::endl;

        const int i1 = target.getXMin();
        const int i2 = target.getXMax();
        const int j1 = target.getYMin();
        const int j2 = target.getYMax();

        int index = (ix - i1) * (j2 - j1 + 1) + (iy - j1);
        xdbg<<"index = "<<index<<std::endl;
        const Polygon& poly = _imagepolys[index];
        xdbg<<"p = "<<x<<','<<y<<std::endl;
        xdbg<<"inner = "<<poly.getInnerBounds()<<std::endl;
        xdbg<<"outer = "<<poly.getOuterBounds()<<std::endl;

        // First do some easy checks if the point isn't terribly close to the boundary.
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t  = 0;
#endif
        Point p(x,y);
        bool inside;
        if (poly.triviallyContains(p)) {
            xdbg<<"trivial\n";
            inside = true;
        } else if (!poly.mightContain(p)) {
            xdbg<<"trivially not\n";
            inside = false;
        } else {
            xdbg<<"maybe\n";
            // OK, it must be near the boundary, so now be careful.
            // The term zfactor decreases the pixel shifts as we get closer to the bottom
            // It is an empirical fit to the Poisson solver simulations, and only matters
            // when we get quite close to the bottom.  This could be more accurate by making
            // the Vertices files have an additional look-up variable (z), but this doesn't
            // seem necessary at this point
            const double zfit = 12.0;
            const double zfactor = std::tanh(zconv / zfit);

            // Scale the testpoly vertices by zfactor
            _testpoly[t].scale(poly, _emptypoly, zfactor);

            // Now test to see if the point is inside
            inside = _testpoly[t].contains(p);
        }

        // If the nominal pixel is on the edge of the image and the photon misses in the
        // direction of falling off the image, (possibly) report that in off_edge.
        if (!inside && off_edge) {
            xdbg<<"Check for off_edge\n";
            xdbg<<"inner = "<<poly.getInnerBounds()<<std::endl;
            xdbg<<"ix,i1,i2 = "<<ix<<','<<i1<<','<<i2<<std::endl;
            xdbg<<"iy,j1,j2 = "<<iy<<','<<j1<<','<<j2<<std::endl;
            *off_edge = false;
            xdbg<<"ix == i1 ? "<<(ix == i1)<<std::endl;
            xdbg<<"x < inner.xmin? "<<(x < _testpoly[t].getInnerBounds().getXMin())<<std::endl;
            if ((ix == i1) && (x < poly.getInnerBounds().getXMin())) *off_edge = true;
            if ((ix == i2) && (x > poly.getInnerBounds().getXMax())) *off_edge = true;
            if ((iy == j1) && (y < poly.getInnerBounds().getYMin())) *off_edge = true;
            if ((iy == j2) && (y > poly.getInnerBounds().getYMax())) *off_edge = true;
            xdbg<<"off_edge = "<<*off_edge<<std::endl;
        }
        return inside;
    }

    // Helper function to calculate how far down into the silicon the photon converts into
    // an electron.

    double Silicon::calculateConversionDepth(const PhotonArray& photons, int i,
                                             double randomNumber) const
    {
        // Determine the distance the photon travels into the silicon
        double si_length;
        if (photons.hasAllocatedWavelengths()) {
            double lambda = photons.getWavelength(i); // in nm
            // Lookup the absorption length in the imported table
            double abs_length = _abs_length_table.lookup(lambda); // in microns
            si_length = -abs_length * log(1.0 - randomNumber); // in microns
#ifdef DEBUGLOGGING
            if (i % 1000 == 0) {
                xdbg<<"lambda = "<<lambda<<std::endl;
                xdbg<<"si_length = "<<si_length<<std::endl;
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
#ifdef DEBUGLOGGING
            if (i % 1000 == 0) {
                xdbg<<"dxdz = "<<dxdz<<std::endl;
                xdbg<<"dydz = "<<dydz<<std::endl;
                xdbg<<"dz = "<<dz<<std::endl;
            }
#endif
            return std::min(_sensorThickness - 1.0, dz);  // max 1 micron from bottom
        } else {
            return si_length;
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
        xdbg<<"searchNeighbors for "<<ix<<','<<iy<<','<<x<<','<<y<<std::endl;
        // The following code finds which pixel we are in given
        // pixel distortion due to the brighter-fatter effect
        // The following are set up to start the search in the undistorted pixel, then
        // search in the nearest neighbor first if it's not in the undistorted pixel.
        if      ((x > y) && (x > 1.0 - y)) step = 1;
        else if ((x < y) && (x < 1.0 - y)) step = 7;
        else if ((x < y) && (x > 1.0 - y)) step = 3;
        else step = 5;
        int n=step;
        xdbg<<"step = "<<step<<std::endl;
        for (int m=1; m<9; m++) {
            int ix_off = ix + xoff[n];
            int iy_off = iy + yoff[n];
            double x_off = x - xoff[n];
            double y_off = y - yoff[n];
            xdbg<<n<<"  "<<ix_off<<"  "<<iy_off<<"  "<<x_off<<"  "<<y_off<<std::endl;
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
    void Silicon::fillWithPixelAreas(ImageView<T> target, Position<int> orig_center,
                                     bool use_flux)
    {
        Bounds<int> b = target.getBounds();
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        const int i1 = b.getXMin();
        const int i2 = b.getXMax();
        const int j1 = b.getYMin();
        const int j2 = b.getYMax();
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;
        const int nxny = nx * ny;

        if (use_flux) {
            dbg<<"Start full pixel area calculation\n";
            dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;
            dbg<<"total memory = "<<nxny*_nv*sizeof(Point)/(1024.*1024.)<<" MBytes"<<std::endl;

            _imagepolys.resize(nxny);
            for (int i=0; i<nxny; ++i)
                _imagepolys[i] = _emptypoly;

            // Set up the pixel information according to the current flux in the image.
            addTreeRingDistortions(target, orig_center);
            updatePixelDistortions(target);

            // Fill target with the area in each pixel.
            const int skip = target.getNSkip();
            const int step = target.getStep();
            T* ptr = target.getData();

            for (int j=j1; j<=j2; ++j, ptr+=skip) {
                for (int i=i1; i<=i2; ++i, ptr+=step) {
                    int index = (i - i1) * ny + (j - j1);
                    *ptr = _imagepolys[index].area();
                }
            }
        } else {
            // If we don't care about respecting the flux in the image (which we usually don't
            // since this is generally used for making sky images with the tree ring patterns),
            // we can save a lot of the memory required for the above algorithm and just
            // calculate one polygon at a time and get the area.
            if (_tr_radial_table.size() == 2) {
                //dbg<<"Trivial radial table\n";
                // The no tree rings case is indicated with a table of size 2, which
                // wouldn't make any sense as a user input.
                target.fill(1.);
                return;
            }
            dbg<<"Start no-flux pixel area calculation\n";

            // Cycle through the pixels in the target image and add
            // the (small) distortions due to tree rings.
            // Then write the area to the target image.
            const int skip = target.getNSkip();
            const int step = target.getStep();
            T* ptr = target.getData();

            // Temporary space.
            Polygon poly;

            for (int j=j1; j<=j2; ++j, ptr+=skip) {
                for (int i=i1; i<=i2; ++i, ptr+=step) {
                    poly = _emptypoly;
                    calculateTreeRingDistortion(i, j, orig_center, poly);
                    *ptr = poly.area();
                }
            }
        }
    }

    template <typename T>
    double Silicon::accumulate(const PhotonArray& photons, BaseDeviate rng, ImageView<T> target,
                               Position<int> orig_center, bool resume)
    {
        const int nphotons = photons.size();

        // Generate random numbers in advance
        std::vector<double> conversionDepthRandom(nphotons);
        std::vector<double> pixelNotFoundRandom(nphotons);
        std::vector<double> diffStepRandom(nphotons * 2);

        UniformDeviate ud(rng);
        GaussianDeviate gd(ud, 0, 1);

        for (int i = 0; i < nphotons; i++) {
            diffStepRandom[i*2] = gd();
            diffStepRandom[i*2+1] = gd();
            pixelNotFoundRandom[i] = ud();
            conversionDepthRandom[i] = ud();
        }

        Bounds<int> b = target.getBounds();
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

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
        double next_recalc;
        if (resume) {
            // These two tests are now done at the python layer.
            // However, Jim is reporting that he's hitting the assert statement below, which
            // I (MJ) thought shouldn't happen.  So rather than bomb out, raise an exception.
#if 1
            // _resume_next_recalc initialized to -1, so this is our sign that we haven't run
            // accumulate yet.
            if (_resume_next_recalc == -999)
                throw std::runtime_error(
                    "Silicon::accumulate called with resume, but accumulate hasn't been run yet.");

            // This isn't a complete check that it is the same image.  But it prevents
            // seg faults if the user messes up.
            if (int(_imagepolys.size()) != nxny)
                throw std::runtime_error(
                    "Silicon::accumulate called with resume, but image is not the same shape as "
                    "the previous run.");
#endif
            assert(_resume_next_recalc != -999);
            assert(int(_imagepolys.size()) == nxny);

            next_recalc = _resume_next_recalc;
            // We already added delta to target.  But to get the right values when we next
            // updatePixelDistortions, we want delta to have everything that's been added since
            // the last update.  The easiest way to do that is to just subtract off what has
            // been added so far now and just keep adding to the existing _delta image.
            // It will all be added back at the end of this call to accumulate.
            target -= _delta;
            dbg<<"resume=True.  Use saved next_recalc = "<<next_recalc<<std::endl;
        } else {
            _imagepolys.resize(nxny);
            for (int i=0; i<nxny; ++i)
                _imagepolys[i] = _emptypoly;
            dbg<<"Built poly list\n";
            // Now we add in the tree ring distortions
            addTreeRingDistortions(target, orig_center);

            // Start with the correct distortions for the initial image as it is already
            dbg<<"Initial updatePixelDistortions\n";
            updatePixelDistortions(target);
            next_recalc = _nrecalc;

            // Keep track of the charge we are accumulating on a separate image for efficiency
            // of the distortion updates.
            _delta.resize(b);
            _delta.setZero();
        }
        const double invPixelSize = 1./_pixelSize; // pixels/micron
        const double diffStep_pixel_z = _diffStep / (_sensorThickness * _pixelSize);

        double addedFlux = 0.;
        int startPhoton = 0;

        while (startPhoton < nphotons) {
            // new parallel version of code

            // count up how many photos we can use before recalc is needed
            int photonsUntilRecalc = startPhoton;
            while ((photonsUntilRecalc < nphotons) && (addedFlux <= next_recalc)) {
                addedFlux += photons.getFlux(photonsUntilRecalc);
                photonsUntilRecalc++;
            }

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = startPhoton; i < photonsUntilRecalc; i++) {
                // Get the location where the photon strikes the silicon:
                double x0 = photons.getX(i); // in pixels
                double y0 = photons.getY(i); // in pixels
                xdbg<<"x0,y0 = "<<x0<<','<<y0;

                double dz = calculateConversionDepth(photons, i, conversionDepthRandom[i]);
                if (photons.hasAllocatedAngles()) {
                    double dxdz = photons.getDXDZ(i);
                    double dydz = photons.getDYDZ(i);
                    double dz_pixel = dz * invPixelSize;
                    x0 += dxdz * dz_pixel; // dx in pixels
                    y0 += dydz * dz_pixel; // dy in pixels
                }
                xdbg<<" => "<<x0<<','<<y0;
                // This is the reverse of depth. zconv is how far above the substrate the e- converts.
                double zconv = _sensorThickness - dz;
                if (zconv < 0.0) continue; // Throw photon away if it hits the bottom
                // TODO: Do something more realistic if it hits the bottom.

                // Now we add in a displacement due to diffusion
                if (_diffStep != 0.) {
                    double diffStep = std::max(0.0, diffStep_pixel_z * std::sqrt(zconv * _sensorThickness));
                    x0 += diffStep * diffStepRandom[i*2];
                    y0 += diffStep * diffStepRandom[i*2+1];
                }
                xdbg<<" => "<<x0<<','<<y0<<std::endl;
                double flux = photons.getFlux(i);

#ifdef DEBUGLOGGING
                if (i % 1000 == 0) {
                    xdbg<<"diffStep = "<<_diffStep<<std::endl;
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
                bool off_edge;
                bool foundPixel = insidePixel(ix, iy, x, y, zconv, target, &off_edge);
#ifdef DEBUGLOGGING
                if (foundPixel) ++zerocount;
#endif

                // If the nominal position is on the edge of the image, off_edge reports whether
                // the photon has fallen off the edge of the image. In this case, we won't find it in
                // any of the neighbors either.  Just let the photon fall off the edge in this case.
                if (!foundPixel && off_edge) continue;

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
#ifdef DEBUGLOGGING
                    dbg<<"Not found in any pixel\n";
                    dbg<<"x0,y0 = "<<x0<<','<<y0<<std::endl;
                    dbg<<"b = "<<b<<std::endl;
                    dbg<<"ix,iy = "<<ix<<','<<iy<<"  x,y = "<<x<<','<<y<<std::endl;
                    set_verbose(2);
                    bool off_edge;
                    insidePixel(ix, iy, x, y, zconv, target, &off_edge);
                    searchNeighbors(*this, ix, iy, x, y, zconv, target, step);
                    set_verbose(1);
                    ++misscount;
#endif
                    int n = (pixelNotFoundRandom[i] > 0.5) ? 0 : step;
                    ix = ix + xoff[n];
                    iy = iy + yoff[n];
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
#ifdef _OPENMP
#pragma omp atomic
#endif
                    _delta(ix,iy) += flux;

                    // no longer need to update addedFlux as it's done before this loop
                }
            }

            // Update shapes every _nrecalc electrons
            if (addedFlux > next_recalc) {
                dbg<<"updatePixelDistortions because "<<addedFlux<<" > "<<next_recalc<<std::endl;
                updatePixelDistortions(_delta.view());
                target += _delta;
                _delta.setZero();
                next_recalc = addedFlux + _nrecalc;
            }

            startPhoton = photonsUntilRecalc;
        }

        // No need to update the distortions again, but we do need to add the delta image.
        target += _delta;
        _resume_next_recalc = next_recalc - addedFlux;
        dbg<<"All done.  Added flux "<<addedFlux<<".  Save next_recalc = "<<_resume_next_recalc<<std::endl;

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

    int SetOMPThreads(int num_threads)
    {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    int GetOMPThreads()
    {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    template bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                                       ImageView<double> target, bool*) const;
    template bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                                       ImageView<float> target, bool*) const;

    template void Silicon::updatePixelDistortions(ImageView<double> target);
    template void Silicon::updatePixelDistortions(ImageView<float> target);

    template void Silicon::addTreeRingDistortions(ImageView<double> target,
                                                  Position<int> orig_center);
    template void Silicon::addTreeRingDistortions(ImageView<float> target,
                                                  Position<int> orig_center);

    template double Silicon::accumulate(const PhotonArray& photons, BaseDeviate rng,
                                        ImageView<double> target, Position<int> orig_center,
                                        bool resume);
    template double Silicon::accumulate(const PhotonArray& photons, BaseDeviate rng,
                                        ImageView<float> target, Position<int> orig_center,
                                        bool resume);

    template void Silicon::fillWithPixelAreas(ImageView<double> target, Position<int> orig_center,
                                              bool);
    template void Silicon::fillWithPixelAreas(ImageView<float> target, Position<int> orig_center,
                                              bool);

} // namespace galsim
