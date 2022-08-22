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

// Uncomment this for debugging output
//#define DEBUGLOGGING

#ifdef DEBUGLOGGING
#undef _OPENMP
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

        poly.reserve(numVertices*4 + 8);
        // First the corners
        dbg<<"corners:\n";
        for (int xpix=0; xpix<2; xpix++) {
            for (int ypix=0; ypix<2; ypix++) {
                poly.add(Position<double>(xpix, ypix));
                // Two copies of the corner to be consistent with new code
                // that has two corner points.
                poly.add(Position<double>(xpix, ypix));
            }
        }
        // Next the edges
        dbg<<"x edges:\n";
        for (int xpix=0; xpix<2; xpix++) {
            for (int n=0; n<numVertices; n++) {
                double theta = theta0 + (n + 1.0) * dtheta;
                poly.add(Position<double>(xpix, (std::tan(theta) + 1.0) / 2.0));
            }
        }
        dbg<<"y edges:\n";
        for (int ypix=0; ypix<2; ypix++) {
            for (int n=0; n<numVertices; n++) {
                double theta = theta0 + (n + 1.0) * dtheta;
                poly.add(Position<double>((std::tan(theta) + 1.0) / 2.0, ypix));
            }
        }
        poly.sort();
    }

    Silicon::Silicon(int numVertices, double numElec, int nx, int ny, int qDist,
                     double diffStep, double pixelSize,
                     double sensorThickness, double* vertex_data,
                     const Table& tr_radial_table, Position<double> treeRingCenter,
                     const Table& abs_length_table, bool transpose) :
        _numVertices(numVertices), _nx(nx), _ny(ny), _qDist(qDist),
        _diffStep(diffStep), _pixelSize(pixelSize),
        _sensorThickness(sensorThickness),
        _tr_radial_table(tr_radial_table), _treeRingCenter(treeRingCenter),
        _abs_length_table(abs_length_table), _transpose(transpose)
    {
        dbg<<"Silicon constructor\n";
        // This constructor reads in the distorted pixel shapes from the Poisson solver
        // and builds arrays of points for calculating the distorted pixel shapes
        // as a function of charge in the surrounding pixels.

        // First build the distorted points. We have linear boundary arrays,
        // an undistorted polygon, and a polygon for test.

        int nv1 = 4 * _numVertices + 4; // Number of vertices in each pixel in input file
        _nv = 4 * _numVertices + 8; // Number of vertices in each pixel
        dbg<<"_numVertices = "<<_numVertices<<", _nv = "<<_nv<<std::endl;
        dbg<<"nx,ny = "<<nx<<", "<<ny<<"  ntot = "<<nx*ny<<std::endl;
        dbg<<"total memory = "<<nx*ny*_nv*sizeof(Position<float>)/(1024.*1024.)<<" MBytes"<<std::endl;

        buildEmptyPoly(_emptypoly, _numVertices);
        // These are mutable Polygons we'll use as scratch space
        int numThreads = 1;
#ifdef _OPENMP
        numThreads = omp_get_max_threads();
#endif
        for (int i=0; i < numThreads; i++) {
            _testpoly.push_back(_emptypoly);
        }

        // Next, we read in the pixel distortions from the Poisson_CCD simulations
        if (_transpose) std::swap(_nx,_ny);

        _horizontalDistortions.resize(horizontalRowStride(_nx) * (_ny + 1));
        _verticalDistortions.resize(verticalColumnStride(_ny) * (_nx + 1));

        for (int index=0; index < nv1*_nx*_ny; index++) {
            int n1 = index % nv1;
            int j = (index / nv1) % _ny;
            int i = index / (nv1 * _ny);
            xdbg<<"index = "<<index<<std::endl;
            xdbg<<"i,j = "<<i<<','<<j<<std::endl;
            xassert(index == (i * _ny + j) * nv1 + n1);
            double x0 = vertex_data[5*index+0];
            double y0 = vertex_data[5*index+1];
            double x1 = vertex_data[5*index+3];
            double y1 = vertex_data[5*index+4];
            if (_transpose) {
                xdbg<<"Original i,j,n = "<<i<<','<<j<<','<<n1<<std::endl;
                std::swap(i,j);
                std::swap(x0,y0);
                std::swap(x1,y1);
                n1 = (_numVertices - n1 + nv1) % nv1;
                xdbg<<"    => "<<i<<','<<j<<','<<n1<<std::endl;
            }

            // Figure out the new index around the pixel polygon when there are two corner points.
            int n = n1;
            if (n >= cornerIndexBottomLeft()) ++n;
            if (n > cornerIndexBottomRight()) ++n;
            if (n >= cornerIndexTopRight()) ++n;
            if (n > cornerIndexTopLeft()) ++n;
            xdbg<<"n1 = "<<n1<<", n = "<<n<<std::endl;

            // The following captures the pixel displacement. These are translated into
            // coordinates compatible with (x,y). These are per electron.
            double x = _emptypoly[n].x;
            x = ((x1 - x0) / _pixelSize + 0.5 - x) / numElec;
            double y = _emptypoly[n].y;
            y = ((y1 - y0) / _pixelSize + 0.5 - y) / numElec;

            // populate the linear distortions arrays
            // make sure to always use values from closest to center pixel
            if ((((n < cornerIndexBottomLeft()) || (n > cornerIndexTopLeft())) && (i <= (_nx / 2))) ||  // LHS
                (((n > cornerIndexBottomRight()) && (n < cornerIndexTopRight())) && (i >= (_nx / 2))) || // RHS
                (((n >= cornerIndexBottomLeft()) && (n <= cornerIndexBottomRight())) && (j <= (_ny / 2))) || // bottom
                (((n >= cornerIndexTopRight()) && (n <= cornerIndexTopLeft())) && (j >= (_ny / 2)))) {  // top
                bool horiz = false;
                int bidx = getBoundaryIndex(i, j, n, &horiz);
                if (horiz) {
                    _horizontalDistortions[bidx].x = x;
                    _horizontalDistortions[bidx].y = y;
                }
                else {
                    _verticalDistortions[bidx].x = x;
                    _verticalDistortions[bidx].y = y;
                }
            }

            // If this is a corner pixel, we need to increment n and add it again.
            // For the old method (using _distortions), this just duplicates the point.
            // But in the new method, this will do different things in the horizontal and
            // vertical directions to make sure we get both possible corner locations
            // in the two directions if appropriate.
            bool corner = ((n == cornerIndexBottomLeft() - 1) ||
                           (n == cornerIndexBottomRight()) ||
                           (n == cornerIndexTopRight() - 1) ||
                           (n == cornerIndexTopLeft()));

            if (corner) {
                // Increment n to the next location around the polygon.
                ++n;
                xdbg<<"corner.  n => "<<n<<std::endl;

                // Do all the same stuff as above again.  (Could consider pulling this little
                // section out into a function that we call twice.)
                if ((((n < cornerIndexBottomLeft()) || (n > cornerIndexTopLeft())) && (i <= (_nx / 2))) ||  // LHS
                    (((n > cornerIndexBottomRight()) && (n < cornerIndexTopRight())) && (i >= (_nx / 2))) || // RHS
                    (((n >= cornerIndexBottomLeft()) && (n <= cornerIndexBottomRight())) && (j <= (_ny / 2))) || // bottom
                    (((n >= cornerIndexTopRight()) && (n <= cornerIndexTopLeft())) && (j >= (_ny / 2)))) {  // top
                    bool horiz = false;
                    int bidx = getBoundaryIndex(i, j, n, &horiz);
                    if (horiz) {
                        _horizontalDistortions[bidx].x = x;
                        _horizontalDistortions[bidx].y = y;
                    }
                    else {
                        _verticalDistortions[bidx].x = x;
                        _verticalDistortions[bidx].y = y;
                    }
                }
            }
        }
    }

    void Silicon::updatePixelBounds(int nx, int ny, size_t k)
    {
        // update the bounding rectangles for pixel k
        // get pixel co-ordinates
        int x = k / ny;
        int y = k % ny;

        // compute outer bounds first
        _pixelOuterBounds[k] = Bounds<double>();

        iteratePixelBoundary(x, y, nx, ny, [this, k](int n, Position<float>& pt, bool rhs, bool top) {
                             Position<double> p = pt;
                             if (rhs) p.x += 1.0;
                             if (top) p.y += 1.0;
                             _pixelOuterBounds[k] += p;
                             });

        Position<double> center = _pixelOuterBounds[k].center();

        // now compute inner bounds manually
        _pixelInnerBounds[k] = _pixelOuterBounds[k];
        Bounds<double>& inner = _pixelInnerBounds[k];

        iteratePixelBoundary(x, y, nx, ny, [&](int n, Position<float>& pt, bool rhs, bool top) {
                             Position<double> p = pt;
                             if (rhs) p.x += 1.0;
                             if (top) p.y += 1.0;

                             if (p.x-center.x >= std::abs(p.y-center.y) && p.x < inner.getXMax()) inner.setXMax(p.x);
                             if (p.x-center.x <= -std::abs(p.y-center.y) && p.x > inner.getXMin()) inner.setXMin(p.x);
                             if (p.y-center.y >= std::abs(p.x-center.x) && p.y < inner.getYMax()) inner.setYMax(p.y);
                             if (p.y-center.y <= -std::abs(p.x-center.x) && p.y > inner.getYMin()) inner.setYMin(p.y);
                             });
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
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;
        const int step = target.getStep();
        const int stride = target.getStride();

        std::vector<bool> changed(nx * ny, false);

        // Loop through the boundary arrays and update any points affected by nearby pixels
        // Horizontal array first
        const T* ptr = target.getData();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int p=0; p < (ny * nx); p++) {
            // Calculate which pixel we are currently below
            int x = p % nx;
            int y = p / nx;

            // Loop over rectangle of pixels that could affect this row of points
            int polyi1 = std::max(x - _qDist, 0);
            int polyi2 = std::min(x + _qDist, nx - 1);
            // NB. We are working between rows y and y-1, so need polyj1 = y-1 - _qDist.
            int polyj1 = std::max(y - (_qDist + 1), 0);
            int polyj2 = std::min(y + _qDist, ny - 1);

            bool change = false;
            for (int j=polyj1; j <= polyj2; j++) {
                for (int i=polyi1; i <= polyi2; i++) {
                    // Check whether this pixel has charge on it
                    double charge = ptr[(j * stride) + (i * step)];

                    if (charge != 0.0) {
                        change = true;

                        // Work out corresponding index into distortions array
                        int dist_index = (((y - j + nyCenter) * _nx) + (x - i + nxCenter)) * horizontalPixelStride();
                        int index = p * horizontalPixelStride();

                        // Loop over boundary points and update them
                        for (int n=0; n < horizontalPixelStride(); ++n, ++index, ++dist_index) {
                            _horizontalBoundaryPoints[index].x =
                                double(_horizontalBoundaryPoints[index].x) +
                                _horizontalDistortions[dist_index].x * charge;
                            _horizontalBoundaryPoints[index].y =
                                double(_horizontalBoundaryPoints[index].y) +
                                _horizontalDistortions[dist_index].y * charge;
                        }
                    }
                }
            }

            // update changed array
            if (change) {
                if (y < ny) changed[(x * ny) + y] = true; // pixel above
                if (y > 0)  changed[(x * ny) + (y - 1)] = true; // pixel below
            }
        }

        // Now vertical array
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int p=0; p < (nx * ny); p++) {
            // Calculate which pixel we are currently on
            int x = p / ny;
            int y = (ny - 1) - (p % ny); // remember vertical points run top-to-bottom

            // Loop over rectangle of pixels that could affect this column of points
            int polyi1 = std::max(x - (_qDist + 1), 0);
            int polyi2 = std::min(x + _qDist, nx - 1);
            int polyj1 = std::max(y - _qDist, 0);
            int polyj2 = std::min(y + _qDist, ny - 1);

            bool change = false;
            for (int j=polyj1; j <= polyj2; j++) {
                for (int i=polyi1; i <= polyi2; i++) {
                    // Check whether this pixel has charge on it
                    double charge = ptr[(j * stride) + (i * step)];

                    if (charge != 0.0) {
                        change = true;

                        // Work out corresponding index into distortions array
                        int dist_index = (((x - i + nxCenter) * _ny) + ((_ny - 1) - (y - j + nyCenter))) * verticalPixelStride() + (verticalPixelStride() - 1);
                        int index = p * verticalPixelStride() + (verticalPixelStride() - 1);

                        // Loop over boundary points and update them
                        for (int n=0; n < verticalPixelStride(); ++n, --index, --dist_index) {
                            _verticalBoundaryPoints[index].x =
                                double(_verticalBoundaryPoints[index].x) +
                                _verticalDistortions[dist_index].x * charge;
                            _verticalBoundaryPoints[index].y =
                                double(_verticalBoundaryPoints[index].y) +
                                _verticalDistortions[dist_index].y * charge;
                        }
                    }
                }
            }

            // update changed array
            if (change) {
                if (x < nx) changed[(x * ny) + y] = true;
                if (x > 0)  changed[((x - 1) * ny) + y] = true;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (size_t k=0; k<changed.size(); ++k) {
            if (changed[k]) {
                updatePixelBounds(nx, ny, k);
            }
        }
    }

    // This version of calculateTreeRingDistortion only distorts a polygon.
    // Used in the no-flux pixel area calculation.
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
            if (r > 0) {
                // Shifts are along the radial vector in direction of the doping gradient
                double dx = shift * tx / r;
                double dy = shift * ty / r;
                xdbg<<"dx,dy = "<<dx<<','<<dy<<std::endl;
                poly[n].x = double(poly[n].x) + dx;
                poly[n].y = double(poly[n].y) + dy;
                xdbg<<"    x,y => "<<poly[n].x <<"  "<< poly[n].y;
            }
        }
    }

    // This version updates the linear boundary
    void Silicon::calculateTreeRingDistortion(int i, int j, Position<int> orig_center,
                                              int nx, int ny, int i1, int j1)
    {
        iteratePixelBoundary(i - i1, j - j1, nx, ny, [&](int n, Position<float>& pt, bool rhs, bool top) {
                             Position<double> p = pt;

                             // only do bottom and left points unless we're on top/right edge
                             if ((rhs) && ((i - i1) < (nx - 1))) return;
                             if ((top) && ((j - j1) < (ny - 1))) return;

                             if (rhs) p.x += 1.0;
                             if (top) p.y += 1.0;
                             //xdbg<<"x,y = "<<p.x<<','<<p.y<<std::endl;

                             double tx = (double)i + p.x - _treeRingCenter.x + (double)orig_center.x;
                             double ty = (double)j + p.y - _treeRingCenter.y + (double)orig_center.y;
                             //xdbg<<"tx,ty = "<<tx<<','<<ty<<std::endl;
                             double r = sqrt(tx * tx + ty * ty);
                             if (r > 0) {
                                double shift = _tr_radial_table.lookup(r);
                                //xdbg<<"r = "<<r<<", shift = "<<shift<<std::endl;
                                // Shifts are along the radial vector in direction of the doping gradient
                                double dx = shift * tx / r;
                                double dy = shift * ty / r;
                                //xdbg<<"dx,dy = "<<dx<<','<<dy<<std::endl;
                                pt.x += dx;
                                pt.y += dy;
                             }
        });
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
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;
        // Now we cycle through the pixels in the target image and add
        // the (small) distortions due to tree rings
        std::vector<bool> changed(nx * ny, false);
        for (int i=i1; i<=i2; ++i) {
            for (int j=j1; j<=j2; ++j) {
                int index = (i - i1) * ny + (j - j1);
                calculateTreeRingDistortion(i, j, orig_center, nx, ny, i1, j1);
                changed[index] = true;
            }
        }
        for (size_t k=0; k<changed.size(); ++k) {
            if (changed[k]) {
                updatePixelBounds(nx, ny, k);
            }
        }
    }

    // Scales a linear pixel boundary into a polygon object.
    void Silicon::scaleBoundsToPoly(int i, int j, int nx, int ny,
                                    const Polygon& emptypoly, Polygon& result,
                                    double factor) const
    {
        result = emptypoly;

        iteratePixelBoundary(i, j, nx, ny, [&](int n, const Position<float>& pt, bool rhs, bool top) {
                             Position<double> p = pt;
                             if (rhs) p.x += 1.0;
                             if (top) p.y += 1.0;
                             result[n].x += (p.x - emptypoly[n].x) * factor;
                             result[n].y += (p.y - emptypoly[n].y) * factor;
                             });

        result.updateBounds();
    }

    // Checks if a point is inside a pixel based on the new linear boundaries.
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
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;

        int index = (ix - i1) * ny + (iy - j1);
        xdbg<<"index = "<<index<<std::endl;
        xdbg<<"p = "<<x<<','<<y<<std::endl;
        xdbg<<"inner = "<<_pixelInnerBounds[index]<<std::endl;
        xdbg<<"outer = "<<_pixelOuterBounds[index]<<std::endl;

        // First do some easy checks if the point isn't terribly close to the boundary.
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t  = 0;
#endif
        Position<double> p(x,y);
        bool inside;
        if (_pixelInnerBounds[index].includes(p)) {
            xdbg<<"trivial\n";
            inside = true;
        } else if (!_pixelOuterBounds[index].includes(p)) {
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
            scaleBoundsToPoly(ix - i1, iy - j1, nx, ny, _emptypoly, _testpoly[t],
                              zfactor);

            // Now test to see if the point is inside
            inside = _testpoly[t].contains(p);
        }

        // If the nominal pixel is on the edge of the image and the photon misses in the
        // direction of falling off the image, (possibly) report that in off_edge.
        if (!inside && off_edge) {
            xdbg<<"Check for off_edge\n";
            xdbg<<"inner = "<<_pixelInnerBounds[index]<<std::endl;
            xdbg<<"ix,i1,i2 = "<<ix<<','<<i1<<','<<i2<<std::endl;
            xdbg<<"iy,j1,j2 = "<<iy<<','<<j1<<','<<j2<<std::endl;
            *off_edge = false;
            xdbg<<"ix == i1 ? "<<(ix == i1)<<std::endl;
            xdbg<<"x < inner.xmin? "<<(x < _pixelInnerBounds[index].getXMin())<<std::endl;
            if ((ix == i1) && (x < _pixelInnerBounds[index].getXMin())) *off_edge = true;
            if ((ix == i2) && (x > _pixelInnerBounds[index].getXMax())) *off_edge = true;
            if ((iy == j1) && (y < _pixelInnerBounds[index].getYMin())) *off_edge = true;
            if ((iy == j2) && (y > _pixelInnerBounds[index].getYMax())) *off_edge = true;
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

    // Calculates the area of a pixel based on the linear boundaries.
    double Silicon::pixelArea(int i, int j, int nx, int ny) const
    {
        double area = 0.0;
        bool horizontal1, horizontal2;

        // compute sum of triangle areas using cross-product rule (shoelace formula)
        for (int n = 0; n < _nv; n++) {
            int pi1 = getBoundaryIndex(i, j, n, &horizontal1, nx, ny);
            Position<double> p1 = horizontal1 ? _horizontalBoundaryPoints[pi1] :
                _verticalBoundaryPoints[pi1];
            if ((n > cornerIndexBottomRight()) && (n < cornerIndexTopRight())) p1.x += 1.0;
            if ((n >= cornerIndexTopRight()) && (n <= cornerIndexTopLeft())) p1.y += 1.0;

            int n2 = (n + 1) % _nv;
            int pi2 = getBoundaryIndex(i, j, n2, &horizontal2, nx, ny);
            Position<double> p2 = horizontal2 ? _horizontalBoundaryPoints[pi2] :
                _verticalBoundaryPoints[pi2];
            if ((n2 > cornerIndexBottomRight()) && (n2 < cornerIndexTopRight())) p2.x += 1.0;
            if ((n2 >= cornerIndexTopRight()) && (n2 <= cornerIndexTopLeft())) p2.y += 1.0;

            area += p1.x * p2.y;
            area -= p2.x * p1.y;
        }

        return std::abs(area) / 2.0;
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
            dbg<<"total memory = "<<nxny*_nv*sizeof(Position<float>)/(1024.*1024.)<<" MBytes"<<std::endl;

            initializeBoundaryPoints(nx, ny);

            // Set up the pixel information according to the current flux in the image.
            addTreeRingDistortions(target, orig_center);
            updatePixelDistortions(target);

            // Fill target with the area in each pixel.
            const int skip = target.getNSkip();
            const int step = target.getStep();
            T* ptr = target.getData();

            for (int j=j1; j<=j2; ++j, ptr+=skip) {
                for (int i=i1; i<=i2; ++i, ptr+=step) {
                    double newArea = pixelArea(i - i1, j - j1, nx, ny);
                    *ptr = newArea;
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

    // Initializes the linear boundary arrays by copying points from _emptypoly.
    void Silicon::initializeBoundaryPoints(int nx, int ny)
    {
        _horizontalBoundaryPoints.resize(horizontalRowStride(nx) * (ny+1));
        _verticalBoundaryPoints.resize(verticalColumnStride(ny) * (nx+1));

        // fill in horizontal boundary points from emptypoly
        int i = 0;
        // loop over rows
        for (int y = 0; y < (ny + 1); y++) {
            // loop over pixels within a row
            for (int x = 0; x < nx; x++) {
                for (int n = cornerIndexBottomLeft(); n <= cornerIndexBottomRight(); n++) {
                    _horizontalBoundaryPoints[i++] = _emptypoly[n];
                }
            }
        }

        // fill in vertical boundary points from emptypoly
        i = 0;
        // loop over columns
        for (int x = 0; x < (nx + 1); x++) {
            // loop over pixels within a column
            for (int y = 0; y < ny; y++) {
                for (int n = cornerIndexTopLeft()+1; n < _nv; n++) {
                    _verticalBoundaryPoints[i++] = _emptypoly[n];
                }
                for (int n = 0; n < cornerIndexBottomLeft(); n++) {
                    _verticalBoundaryPoints[i++] = _emptypoly[n];
                }
            }
        }

        _pixelInnerBounds.resize(nx * ny);
        _pixelOuterBounds.resize(nx * ny);
        for (int k = 0; k < (nx * ny); k++) {
            updatePixelBounds(nx, ny, k);
        }
    }

    template <typename T>
    void Silicon::initialize(ImageView<T> target, Position<int> orig_center)
    {
        Bounds<int> b = target.getBounds();
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        const int nx = b.getXMax() - b.getXMin() + 1;
        const int ny = b.getYMax() - b.getYMin() + 1;
        dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;

        initializeBoundaryPoints(nx, ny);

        dbg<<"Built poly list\n";
        // Now we add in the tree ring distortions
        addTreeRingDistortions(target, orig_center);

        // Start with the correct distortions for the initial image as it is already
        dbg<<"Initial updatePixelDistortions\n";
        updatePixelDistortions(target);

        // Keep track of the charge we are accumulating on a separate image for efficiency
        // of the distortion updates.
        _delta.resize(b);
        _delta.setZero();
    }

    template <typename T>
    void Silicon::subtractDelta(ImageView<T> target)
    {
        // both delta and target and stored on GPU so copy them back
        int imageDataSize = (_delta.getXMax() - _delta.getXMin()) * _delta.getStep() + (_delta.getYMax() - _delta.getYMin()) * _delta.getStride();

        double* deltaData = _delta.getData();
        T* targetData = target.getData();
#pragma omp target update from(deltaData[0:imageDataSize], targetData[0:imageDataSize])
        target -= _delta;
    }

    template <typename T>
    void Silicon::addDelta(ImageView<T> target)
    {
        // both delta and target and stored on GPU so copy them back
        int imageDataSize = (_delta.getXMax() - _delta.getXMin()) * _delta.getStep() + (_delta.getYMax() - _delta.getYMin()) * _delta.getStride();

        double* deltaData = _delta.getData();
        T* targetData = target.getData();
#pragma omp target update from(deltaData[0:imageDataSize], targetData[0:imageDataSize])
        target += _delta;
    }

    template <typename T>
    double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                               BaseDeviate rng, ImageView<T> target)
    {
        const int nphotons = i2 - i1;
        dbg<<"Start accumulate: nphot = "<<nphotons<<std::endl;

        // Generate random numbers in advance
        std::vector<double> conversionDepthRandom(nphotons);
        std::vector<double> pixelNotFoundRandom(nphotons);
        std::vector<double> diffStepRandom(nphotons * 2);

        UniformDeviate ud(rng);
        GaussianDeviate gd(ud, 0, 1);

        for (int i=0; i<nphotons; i++) {
            diffStepRandom[i*2] = gd();
            diffStepRandom[i*2+1] = gd();
            pixelNotFoundRandom[i] = ud();
            conversionDepthRandom[i] = ud();
        }

        const double invPixelSize = 1./_pixelSize; // pixels/micron
        const double diffStep_pixel_z = _diffStep / (_sensorThickness * _pixelSize);
        Bounds<int> b = target.getBounds();
        double addedFlux = 0.;

#ifdef _OPENMP
#pragma omp parallel reduction (+:addedFlux)
        {
#pragma omp for
#endif
        for (int i=i1; i<i2; i++) {
            // Get the location where the photon strikes the silicon:
            double x0 = photons.getX(i); // in pixels
            double y0 = photons.getY(i); // in pixels
            xdbg<<"x0,y0 = "<<x0<<','<<y0;

            double dz = calculateConversionDepth(photons, i, conversionDepthRandom[i-i1]);
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
            xdbg<<"zconv = "<<zconv<<std::endl;
            if (zconv < 0.0) continue; // Throw photon away if it hits the bottom
            // TODO: Do something more realistic if it hits the bottom.

            // Now we add in a displacement due to diffusion
            if (_diffStep != 0.) {
                double diffStep = std::max(0.0, diffStep_pixel_z * std::sqrt(zconv * _sensorThickness));
                x0 += diffStep * diffStepRandom[(i-i1)*2];
                y0 += diffStep * diffStepRandom[(i-i1)*2+1];
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
            bool foundPixel;
            foundPixel = insidePixel(ix, iy, x, y, zconv, target, &off_edge);

            // If the nominal position is on the edge of the image, off_edge reports whether
            // the photon has fallen off the edge of the image. In this case, we won't find it in
            // any of the neighbors either.  Just let the photon fall off the edge in this case.
            if (!foundPixel && off_edge) continue;

            // Then check neighbors
            int step;  // We might need this below, so let searchNeighbors return it.
            if (!foundPixel) {
                foundPixel = searchNeighbors(*this, ix, iy, x, y, zconv, target, step);
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
#endif
                int n = (pixelNotFoundRandom[i-i1] > 0.5) ? 0 : step;
                ix = ix + xoff[n];
                iy = iy + yoff[n];
            }

            if (b.includes(ix,iy)) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                _delta(ix,iy) += flux;

                // This isn't atomic -- openmp is handling the reduction for us.
                addedFlux += flux;
            }
        }
#ifdef _OPENMP
        }
#endif
        return addedFlux;
    }

    template <typename T>
    void Silicon::initializeGPU(ImageView<T> target, Position<int> orig_center)
    {
	// do GPU-specific stuff
        Bounds<int> b = target.getBounds();
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        const int nx = b.getXMax() - b.getXMin() + 1;
        const int ny = b.getYMax() - b.getYMin() + 1;
        dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;

        initializeBoundaryPoints(nx, ny);

        dbg<<"Built poly list\n";
        // Now we add in the tree ring distortions
        addTreeRingDistortions(target, orig_center);

        // Start with the correct distortions for the initial image as it is already
        dbg<<"Initial updatePixelDistortions\n";
        updatePixelDistortions(target);

        // Keep track of the charge we are accumulating on a separate image for efficiency
        // of the distortion updates.
        _delta.resize(b);
        _delta.setZero();


	// convert and map data for GPU
        double* deltaData = _delta.getData();
        int imageDataSize = (_delta.getXMax() - _delta.getXMin()) * _delta.getStep() + (_delta.getYMax() - _delta.getYMin()) * _delta.getStride();

        T* targetData = target.getData();
        
        int pixelInnerBoundsSize = _pixelInnerBounds.size();

        int hbpSize = _horizontalBoundaryPoints.size();
        int vbpSize = _verticalBoundaryPoints.size();

        int hdSize = _horizontalDistortions.size();
        int vdSize = _verticalDistortions.size();
        
	// first item is for lambda=255.0, last for lambda=1450.0
	const double abs_length_table[240] = {
	    0.005376, 0.005181, 0.004950, 0.004673, 0.004444, 0.004292, 0.004237, 0.004348,
	    0.004854, 0.005556, 0.006211, 0.006803, 0.007299, 0.007752, 0.008130, 0.008475,
	    0.008850, 0.009174, 0.009434, 0.009615, 0.009709, 0.009804, 0.010776, 0.013755,
	    0.020243, 0.030769, 0.044843, 0.061728, 0.079365, 0.097087, 0.118273, 0.135230,
	    0.160779, 0.188879, 0.215008, 0.248565, 0.280576, 0.312637, 0.339916, 0.375516,
	    0.421177, 0.462770, 0.519427, 0.532396, 0.586786, 0.638651, 0.678058, 0.724795,
	    0.754888, 0.819471, 0.888573, 0.925497, 1.032652, 1.046835, 1.159474, 1.211754,
	    1.273999, 1.437339, 1.450579, 1.560939, 1.641228, 1.678331, 1.693222, 1.910329,
	    1.965988, 2.107881, 2.183263, 2.338634, 2.302821, 2.578183, 2.540070, 2.812702,
	    2.907146, 2.935392, 3.088994, 3.082139, 3.311807, 3.466084, 3.551767, 3.580123,
	    3.716781, 3.859216, 4.007534, 4.162331, 4.323576, 4.492161, 4.667662, 4.851307,
	    5.042610, 5.243014, 5.451968, 5.670863, 5.899705, 6.139489, 6.390185, 6.652917,
	    6.928086, 7.217090, 7.519928, 7.838219, 8.171938, 8.522969, 8.891260, 9.279881,
	    9.688045, 10.119102, 10.572501, 11.051556, 11.556418, 12.090436, 12.654223,
	    13.251527, 13.883104, 14.553287, 15.263214, 16.017940, 16.818595, 17.671903,
	    18.578727, 19.546903, 20.578249, 21.681627, 22.860278, 24.124288, 25.477707,
	    26.933125, 28.495711, 30.181390, 31.995905, 33.960470, 36.082846, 38.387716,
	    40.886418, 43.610990, 46.578788, 50.147936, 53.455926, 57.267209, 61.599113,
	    66.352598, 71.802973, 77.730276, 84.423808, 91.810503, 100.049024, 109.326777,
	    120.098481, 132.101967, 145.853388, 162.345569, 180.515516, 202.860331,
	    228.060573, 258.191113, 295.011358, 340.808398, 394.960306, 460.893211,
	    541.418517, 640.697078, 760.282825, 912.075885, 1085.116542, 1255.510120,
	    1439.760424, 1647.500741, 1892.004389, 2181.025082, 2509.599217, 2896.955300,
	    3321.155762, 3854.455751, 4470.072862, 5222.477543, 6147.415012, 7263.746641,
	    8802.042074, 10523.214211, 12895.737959, 16091.399147, 20783.582632,
	    26934.575915, 35981.577432, 52750.962705, 90155.066715, 168918.918919,
	    288184.438040, 409836.065574, 534759.358289, 684931.506849, 900900.900901,
	    1190476.190476, 1552795.031056, 2024291.497976, 2673796.791444, 3610108.303249,
	    4830917.874396, 6896551.724138, 10416666.666667, 16920473.773266,
	    27700831.024931, 42918454.935622, 59880239.520958, 79365079.365079,
	    103842159.916926, 135317997.293640, 175746924.428822, 229357798.165138,
	    294117647.058824, 380228136.882129, 497512437.810945, 657894736.842105,
	    877192982.456140, 1204819277.108434, 1680672268.907563, 2518891687.657431,
	    3816793893.129771, 5882352941.176471, 7999999999.999999, 10298661174.047373,
	    14430014430.014431, 17211703958.691910, 21786492374.727669, 27932960893.854748,
	    34482758620.689659, 41666666666.666672, 54347826086.956520, 63694267515.923569,
	    86956521739.130432, 106837606837.606827, 128205128205.128204,
	    185528756957.328400, 182815356489.945160, 263157894736.842072,
	    398406374501.992065, 558659217877.094971, 469483568075.117371,
	    833333333333.333374, 917431192660.550415, 1058201058201.058228
	};

        _abs_length_table_GPU.resize(240);
	//_abs_length_table_GPU = new double[240];
	for (int i = 0; i < 240; i++) {
	    _abs_length_table_GPU[i] = abs_length_table[i];
	}
        double* abs_length_table_data = _abs_length_table_GPU.data();
        
	int emptypolySize = _emptypoly.size();
        _emptypolyGPU.resize(emptypolySize);
	//_emptypolyGPU = new PointDGPU[emptypolySize];
	for (int i = 0; i < emptypolySize; i++) {
	    _emptypolyGPU[i].x = _emptypoly[i].x;
	    _emptypolyGPU[i].y = _emptypoly[i].y;
	}
        Position<double>* emptypolyData = _emptypolyGPU.data();
        
        int nxny = nx * ny;
        //_changedGPU = new bool[nxny];
        //for (int i = 0; i < nxny; i++) _changedGPU[i] = false;
        _changed = std::shared_ptr<bool>(new bool[nxny]);
        for (int i = 0; i < nxny; i++) _changed[i] = false;
        bool* changedData = _changed.get();

        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();

        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

        // map all data to the GPU
#pragma omp target enter data map(to: this[:1], deltaData[0:imageDataSize], targetData[0:imageDataSize], pixelInnerBoundsData[0:pixelInnerBoundsSize], pixelOuterBoundsData[0:pixelInnerBoundsSize], horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize], abs_length_table_data[0:240], emptypolyData[0:emptypolySize], horizontalDistortionsData[0:hdSize], verticalDistortionsData[0:vdSize], changedData[0:nxny])
    }

    template <typename T>
    void Silicon::finalizeGPU(ImageView<T> target)
    {
        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();

        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

        double* abs_length_table_data = _abs_length_table_GPU.data();
        
        Bounds<int> b = target.getBounds();
        const int nx = b.getXMax() - b.getXMin() + 1;
        const int ny = b.getYMax() - b.getYMin() + 1;
        int nxny = nx * ny;

        int pixelInnerBoundsSize = _pixelInnerBounds.size();

        int hbpSize = _horizontalBoundaryPoints.size();
        int vbpSize = _verticalBoundaryPoints.size();

        int hdSize = _horizontalDistortions.size();
        int vdSize = _verticalDistortions.size();
        
	int emptypolySize = _emptypoly.size();
        Position<double>* emptypolyData = _emptypolyGPU.data();

        bool* changedData = _changed.get();
        
        double* deltaData = _delta.getData();
        int imageDataSize = (_delta.getXMax() - _delta.getXMin()) * _delta.getStep() + (_delta.getYMax() - _delta.getYMin()) * _delta.getStride();
        T* targetData = target.getData();
#pragma omp target update from(targetData[0:imageDataSize])

#pragma omp target exit data map(release: this[:1], deltaData[0:imageDataSize], targetData[0:imageDataSize], pixelInnerBoundsData[0:pixelInnerBoundsSize], pixelOuterBoundsData[0:pixelInnerBoundsSize], horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize], abs_length_table_data[0:240], emptypolyData[0:emptypolySize], horizontalDistortionsData[0:hdSize], verticalDistortionsData[0:vdSize], changedData[0:nxny])
        
        //delete[] _abs_length_table_GPU;
        //delete[] _emptypolyGPU;
        //delete[] _changedGPU;
    }
    
    bool Silicon::insidePixelGPU(int ix, int iy, double x, double y, double zconv,
				 BoundsIGPU& targetBounds, bool* off_edge,
                                 int emptypolySize,
                                 Bounds<double>* pixelInnerBoundsData,
                                 Bounds<double>* pixelOuterBoundsData,
                                 Position<float>* horizontalBoundaryPointsData,
                                 Position<float>* verticalBoundaryPointsData) const
    {
        // This scales the pixel distortion based on the zconv, which is the depth
        // at which the electron is created, and then tests to see if the delivered
        // point is inside the pixel.
        // (ix,iy) is the pixel being tested, and (x,y) is the coordinate of the
        // photon within the pixel, with (0,0) in the lower left

        // If test pixel is off the image, return false.  (Avoids seg faults!)
	if ((ix < targetBounds.xmin) || (ix > targetBounds.xmax) ||
	    (iy < targetBounds.ymin) || (iy > targetBounds.ymax)) {
            if (off_edge) *off_edge = true;
            return false;
        }

        const int i1 = targetBounds.xmin;
        const int i2 = targetBounds.xmax;
        const int j1 = targetBounds.ymin;
        const int j2 = targetBounds.ymax;
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;

        int index = (ix - i1) * ny + (iy - j1);

        // First do some easy checks if the point isn't terribly close to the boundary.

        bool inside;
	if ((x >= pixelInnerBoundsData[index].getXMin()) && (x <= pixelInnerBoundsData[index].getXMax()) &&
	    (y >= pixelInnerBoundsData[index].getYMin()) && (y <= pixelInnerBoundsData[index].getYMax())) {
            inside = true;
	} else if ((x < pixelOuterBoundsData[index].getXMin()) || (x > pixelOuterBoundsData[index].getXMax()) ||
		   (y < pixelOuterBoundsData[index].getYMin()) || (y > pixelOuterBoundsData[index].getYMax())) {
            inside = false;
        } else {
            // OK, it must be near the boundary, so now be careful.
            // The term zfactor decreases the pixel shifts as we get closer to the bottom
            // It is an empirical fit to the Poisson solver simulations, and only matters
            // when we get quite close to the bottom.  This could be more accurate by making
            // the Vertices files have an additional look-up variable (z), but this doesn't
            // seem necessary at this point
            const double zfit = 12.0;
            const double zfactor = std::tanh(zconv / zfit);

            Position<double> emptypolyData = _emptypolyGPU.data();
            
            // new version not using testpoly
            // first compute first point of polygon (index 0)
            double x1, y1, xinters = 0.0;
            inside = false;
            for (int n = 0; n <= _nv; n++) {
                double xp = 0.0, yp = 0.0;
                double epx = 0.0, epy = 0.0;
                if (n < _nv) {
                    epx = emptypolyData[n].x;
                    epy = emptypolyData[n].y;
                }
                xp = epx;
                yp = epy;
                int idx;

                // compute this point
                if (n < cornerIndexBottomLeft()) {
                    idx = verticalPixelIndex(ix - i1, iy - j1, ny) + n + cornerIndexBottomLeft();
                    xp += (verticalBoundaryPointsData[idx].x - epx) * zfactor;
                    yp += (verticalBoundaryPointsData[idx].y - epy) * zfactor;
                }
                else if (n <= cornerIndexBottomRight()) {
                    // bottom row including corners
                    idx = horizontalPixelIndex(ix - i1, iy - j1, nx) + (n - cornerIndexBottomLeft());
                    double px = horizontalBoundaryPointsData[idx].x;
                    if (n == cornerIndexBottomRight()) px += 1.0;
                    xp += (px - epx) * zfactor;
                    yp += (horizontalBoundaryPointsData[idx].y - epy) * zfactor;
                }
                // RHS
                else if (n < cornerIndexTopRight()) {
                    idx = verticalPixelIndex(ix - i1 + 1, iy - j1, ny) + (cornerIndexTopRight() - n - 1);
                    xp += ((verticalBoundaryPointsData[idx].x + 1.0) - epx) * zfactor;
                    yp += (verticalBoundaryPointsData[idx].y - epy) * zfactor;
                }
                // top row including corners
                else if (n <= cornerIndexTopLeft()) {
                    idx = horizontalPixelIndex(ix - i1, iy - j1 + 1, nx) + (cornerIndexTopLeft() - n);
                    double px = horizontalBoundaryPointsData[idx].x;
                    if (n == cornerIndexTopRight()) px += 1.0;
                    xp += (px - epx) * zfactor;
                    yp += ((horizontalBoundaryPointsData[idx].y + 1.0) - epy) * zfactor;
                }
                else if (n < _nv) {
                    // LHS upper half
                    idx = verticalPixelIndex(ix - i1, iy - j1, ny) + (n - cornerIndexTopLeft() - 1);
                    xp += (verticalBoundaryPointsData[idx].x - epx) * zfactor;
                    yp += (verticalBoundaryPointsData[idx].y - epy) * zfactor;
                }
                if (n == 0) {
                    // save first point for later
                    x1 = xp;
                    y1 = yp;
                }
                else {
                    // shoelace algorithm
                    double x2 = xp;
                    double y2 = yp;
                    if (n == _nv) {
                        x2 = x1;
                        y2 = y1;
                    }
                    double ymin = y1 < y2 ? y1 : y2;
                    double ymax = y1 > y2 ? y1 : y2;
                    double xmax = x1 > x2 ? x1 : x2;
                    if (y > ymin) {
                        if (y <= ymax) {
                            if (x <= xmax) {
                                if (y1 != y2) {
                                    xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1;
                                }
                                if ((x1 == x2) || (x <= xinters)) {
                                    inside = !inside;
                                }
                            }
                        }
                    }
                    x1 = x2;
                    y1 = y2;
                }
            }
        }
        
        // If the nominal pixel is on the edge of the image and the photon misses in the
        // direction of falling off the image, (possibly) report that in off_edge.
        if (!inside && off_edge) {
            *off_edge = false;
            if ((ix == i1) && (x < pixelInnerBoundsData[index].getXMin())) *off_edge = true;
            if ((ix == i2) && (x > pixelInnerBoundsData[index].getXMax())) *off_edge = true;
            if ((iy == j1) && (y < pixelInnerBoundsData[index].getYMin())) *off_edge = true;
            if ((iy == j2) && (y > pixelInnerBoundsData[index].getYMax())) *off_edge = true;
        }
        return inside;
    }

    bool searchNeighborsGPU(const Silicon& silicon, int& ix, int& iy, double x, double y, double zconv,
			    BoundsIGPU& targetBounds, int& step, int emptypolysize,
                            Bounds<double>* pixelInnerBoundsData,
                            Bounds<double>* pixelOuterBoundsData,
                            Position<float>* horizontalBoundaryPointsData,
                            Position<float>* verticalBoundaryPointsData)
    {
        const int xoff[9] = {0,1,1,0,-1,-1,-1,0,1}; // Displacements to neighboring pixels
        const int yoff[9] = {0,0,1,1,1,0,-1,-1,-1}; // Displacements to neighboring pixels

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
	    if (silicon.insidePixelGPU(ix_off, iy_off, x_off, y_off, zconv,
                                       targetBounds, nullptr, emptypolysize,
                                       pixelInnerBoundsData, pixelOuterBoundsData,
                                       horizontalBoundaryPointsData,
                                       verticalBoundaryPointsData)) {
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
    double Silicon::accumulateGPU(const PhotonArray& photons, int i1, int i2,
				  BaseDeviate rng, ImageView<T> target)
    {
        const int nphotons = i2 - i1;

        // Generate random numbers in advance
        std::vector<double> conversionDepthRandom(nphotons);
        std::vector<double> pixelNotFoundRandom(nphotons);
        std::vector<double> diffStepRandom(nphotons * 2);

        UniformDeviate ud(rng);
        GaussianDeviate gd(ud, 0, 1);

        for (int i=0; i<nphotons; i++) {
            diffStepRandom[i*2] = gd();
            diffStepRandom[i*2+1] = gd();
            pixelNotFoundRandom[i] = ud();
            conversionDepthRandom[i] = ud();
        }

        const double invPixelSize = 1./_pixelSize; // pixels/micron
        const double diffStep_pixel_z = _diffStep / (_sensorThickness * _pixelSize);
        Bounds<int> b = target.getBounds();
        double addedFlux = 0.;

	// Get everything out of C++ classes and into arrays/structures suitable for GPU
	// photons
        // can't get pointers to internal data from a const reference
        PhotonArray& photonsMutable = const_cast<PhotonArray&>(photons);        
	double* photonsX = photonsMutable.getXArray();
	double* photonsY = photonsMutable.getYArray();
	double* photonsDXDZ = photonsMutable.getDXDZArray();
	double* photonsDYDZ = photonsMutable.getDYDZArray();
	double* photonsFlux = photonsMutable.getFluxArray();
	double* photonsWavelength = photonsMutable.getWavelengthArray();
	bool photonsHasAllocatedAngles = photons.hasAllocatedAngles();
	bool photonsHasAllocatedWavelengths = photons.hasAllocatedWavelengths();

        // if no wavelengths allocated, photonsWavelength will be null, but some
        // compilers don't like mapping a null pointer to the GPU, so assign it to
        // something safe instead. (It will never be accessed in this case so the
        // content of what it points to doesn't matter).
        if (!photonsHasAllocatedWavelengths) {
            photonsWavelength = photonsX;
        }
        // same with the angles
        if (!photonsHasAllocatedAngles) {
            photonsDXDZ = photonsX;
            photonsDYDZ = photonsY;
        }
        
	// random arrays
	double* diffStepRandomArray = diffStepRandom.data();
	//double* pixelNotFoundRandomArray = pixelNotFoundRandom.data();
	double* conversionDepthRandomArray = conversionDepthRandom.data();

	// target bounds
	BoundsIGPU targetBounds;
	targetBounds.xmin = target.getXMin();
	targetBounds.xmax = target.getXMax();
	targetBounds.ymin = target.getYMin();
	targetBounds.ymax = target.getYMax();
	
	// delta image
	int deltaXMin = _delta.getXMin();
	int deltaYMin = _delta.getYMin();
        int deltaXMax = _delta.getXMax();
        int deltaYMax = _delta.getYMax();
	int deltaStep = _delta.getStep();
	int deltaStride = _delta.getStride();

	int emptypolySize = _emptypoly.size();

        double* deltaData = _delta.getData();
        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();
        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();

        double* abs_length_table_data = _abs_length_table_GPU.data();
        
#pragma omp target teams distribute parallel for map(to: photonsX[i1:i2-i1], photonsY[i1:i2-i1], photonsDXDZ[i1:i2-i1], photonsDYDZ[i1:i2-i1], photonsFlux[i1:i2-i1], photonsWavelength[i1:i2-i1], diffStepRandomArray[i1*2:(i2-i1)*2], conversionDepthRandomArray[i1:i2-i1]) reduction(+:addedFlux)
	for (int i = i1; i < i2; i++) {
	    double x0 = photonsX[i];
	    double y0 = photonsY[i];

	    // calculateConversionDepth
	    double dz;
	    if (photonsHasAllocatedWavelengths) {
		double lambda = photonsWavelength[i];

		// perform abs_length_table lookup with linear interpolation
		int tableIdx = int((lambda - 255.0) / 5.0);
		double alpha = (lambda - ((float(tableIdx) * 5.0) + 255.0)) / 5.0;
                if (tableIdx < 0) tableIdx = 0;
		int tableIdx1 = tableIdx + 1;
                if (tableIdx > 239) tableIdx = 239;
                if (tableIdx1 > 239) tableIdx1 = 239;
		double abs_length = (abs_length_table_data[tableIdx] * (1.0 - alpha)) +
		    (abs_length_table_data[tableIdx1] * alpha);

		dz = -abs_length * std::log(1.0 - conversionDepthRandomArray[i - i1]);
	    }
	    else {
		dz = 1.0;
	    }

	    if (photonsHasAllocatedAngles) {
		double dxdz = photonsDXDZ[i];
		double dydz = photonsDYDZ[i];
		double pdz = dz / std::sqrt(1.0 + dxdz*dxdz + dydz*dydz);
                dz = _sensorThickness - 1.0;
                if (pdz < dz) dz = pdz;
	    }

	    if (photonsHasAllocatedAngles) {
		double dxdz = photonsDXDZ[i];
		double dydz = photonsDYDZ[i];
		double dz_pixel = dz * invPixelSize;
		x0 += dxdz * dz_pixel;
		y0 += dydz * dz_pixel;
	    }

	    double zconv = _sensorThickness - dz;
	    if (zconv < 0.0) continue;

	    if (_diffStep != 0.) {
		double diffStep = diffStep_pixel_z * std::sqrt(zconv * _sensorThickness);
                if (diffStep < 0.0) diffStep = 0.0;
		x0 += diffStep * diffStepRandomArray[(i-i1)*2];
		y0 += diffStep * diffStepRandomArray[(i-i1)*2+1];
	    }

	    double flux = photonsFlux[i];

	    int ix = int(std::floor(x0 + 0.5));
	    int iy = int(std::floor(y0 + 0.5));

	    double x = x0 - ix + 0.5;
	    double y = y0 - iy + 0.5;

	    bool off_edge;
	    bool foundPixel;

	    foundPixel = insidePixelGPU(ix, iy, x, y, zconv, targetBounds, &off_edge,
                                        emptypolySize, pixelInnerBoundsData,
                                        pixelOuterBoundsData,
                                        horizontalBoundaryPointsData,
                                        verticalBoundaryPointsData);

	    if (!foundPixel && off_edge) continue;

	    int step;
	    if (!foundPixel) {
		foundPixel = searchNeighborsGPU(*this, ix, iy, x, y, zconv,
                                                targetBounds, step, emptypolySize,
                                                pixelInnerBoundsData,
                                                pixelOuterBoundsData,
                                                horizontalBoundaryPointsData,
                                                verticalBoundaryPointsData);
		if (!foundPixel) continue; // ignore ones that have rounding errors for now
            }

	    if ((ix >= targetBounds.xmin) && (ix <= targetBounds.xmax) &&
                (iy >= targetBounds.ymin) && (iy <= targetBounds.ymax)) {

		int deltaIdx = (ix - deltaXMin) * deltaStep + (iy - deltaYMin) * deltaStride;
#pragma omp atomic
                deltaData[deltaIdx] += flux;
		addedFlux += flux;
            }
	}

        return addedFlux;
    }

    void Silicon::updatePixelBoundsGPU(int nx, int ny, size_t k,
                                       Bounds<double>* pixelInnerBoundsData,
                                       Bounds<double>* pixelOuterBoundsData,
                                       Position<float>* horizontalBoundaryPointsData,
                                       Position<float>* verticalBoundaryPointsData)
    {
        // update the bounding rectangles for pixel k
        // get pixel co-ordinates
        int x = k / ny;
        int y = k % ny;

        // compute outer bounds first
        // initialise outer bounds
        double obxmin = 1000000.0, obxmax = -1000000.0;
        double obymin = 1000000.0, obymax = -1000000.0;

        // iterate over pixel boundary
        int n, idx;
        // LHS lower half
        for (n = 0; n < cornerIndexBottomLeft(); n++) {
            idx = verticalPixelIndex(x, y, ny) + n + cornerIndexBottomLeft();
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px < obxmin) obxmin = px;
            if (px > obxmax) obxmax = px;
            if (py < obymin) obymin = py;
            if (py > obymax) obymax = py;
        }
        // bottom row including corners
        for (; n <= cornerIndexBottomRight(); n++) {
            idx = horizontalPixelIndex(x, y, nx) + (n - cornerIndexBottomLeft());
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y;
            if (n == cornerIndexBottomRight()) px += 1.0;
            if (px < obxmin) obxmin = px;
            if (px > obxmax) obxmax = px;
            if (py < obymin) obymin = py;
            if (py > obymax) obymax = py;
        }
        // RHS
        for (; n < cornerIndexTopRight(); n++) {
            idx = verticalPixelIndex(x + 1, y, ny) + (cornerIndexTopRight() - n - 1);
            double px = verticalBoundaryPointsData[idx].x + 1.0;
            double py = verticalBoundaryPointsData[idx].y;
            if (px < obxmin) obxmin = px;
            if (px > obxmax) obxmax = px;
            if (py < obymin) obymin = py;
            if (py > obymax) obymax = py;
        }            
        // top row including corners
        for (; n <= cornerIndexTopLeft(); n++) {
            idx = horizontalPixelIndex(x, y + 1, nx) + (cornerIndexTopLeft() - n);
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y + 1.0;
            if (n == cornerIndexTopRight()) px += 1.0;
            if (px < obxmin) obxmin = px;
            if (px > obxmax) obxmax = px;
            if (py < obymin) obymin = py;
            if (py > obymax) obymax = py;
        }
        // LHS upper half
        for (; n < _nv; n++) {
            idx = verticalPixelIndex(x, y, ny) + (n - cornerIndexTopLeft() - 1);
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px < obxmin) obxmin = px;
            if (px > obxmax) obxmax = px;
            if (py < obymin) obymin = py;
            if (py > obymax) obymax = py;
        }

        // compute center
        double centerx = (obxmin + obxmax) * 0.5;
        double centery = (obymin + obymax) * 0.5;

        // compute inner bounds
        // initialize inner from outer
        double ibxmin = obxmin, ibxmax = obxmax, ibymin = obymin, ibymax = obymax;
        
        // iterate over pixel boundary
        // LHS lower half
        for (n = 0; n < cornerIndexBottomLeft(); n++) {
            idx = verticalPixelIndex(x, y, ny) + n + cornerIndexBottomLeft();
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-centerx >= std::abs(py-centery) && px < ibxmax) ibxmax = px;
            if (px-centerx <= -std::abs(py-centery) && px > ibxmin) ibxmin = px;
            if (py-centery >= std::abs(px-centerx) && py < ibymax) ibymax = py;
            if (py-centery <= -std::abs(px-centerx) && py > ibymin) ibymin = py;
        }
        // bottom row including corners
        for (; n <= cornerIndexBottomRight(); n++) {
            idx = horizontalPixelIndex(x, y, nx) + (n - cornerIndexBottomLeft());
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y;
            if (n == cornerIndexBottomRight()) px += 1.0;
            if (px-centerx >= std::abs(py-centery) && px < ibxmax) ibxmax = px;
            if (px-centerx <= -std::abs(py-centery) && px > ibxmin) ibxmin = px;
            if (py-centery >= std::abs(px-centerx) && py < ibymax) ibymax = py;
            if (py-centery <= -std::abs(px-centerx) && py > ibymin) ibymin = py;
        }
        // RHS
        for (; n < cornerIndexTopRight(); n++) {
            idx = verticalPixelIndex(x + 1, y, ny) + (cornerIndexTopRight() - n - 1);
            double px = verticalBoundaryPointsData[idx].x + 1.0;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-centerx >= std::abs(py-centery) && px < ibxmax) ibxmax = px;
            if (px-centerx <= -std::abs(py-centery) && px > ibxmin) ibxmin = px;
            if (py-centery >= std::abs(px-centerx) && py < ibymax) ibymax = py;
            if (py-centery <= -std::abs(px-centerx) && py > ibymin) ibymin = py;
        }            
        // top row including corners
        for (; n <= cornerIndexTopLeft(); n++) {
            idx = horizontalPixelIndex(x, y + 1, nx) + (cornerIndexTopLeft() - n);
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y + 1.0;
            if (n == cornerIndexTopRight()) px += 1.0;
            if (px-centerx >= std::abs(py-centery) && px < ibxmax) ibxmax = px;
            if (px-centerx <= -std::abs(py-centery) && px > ibxmin) ibxmin = px;
            if (py-centery >= std::abs(px-centerx) && py < ibymax) ibymax = py;
            if (py-centery <= -std::abs(px-centerx) && py > ibymin) ibymin = py;
        }
        // LHS upper half
        for (; n < _nv; n++) {
            idx = verticalPixelIndex(x, y, ny) + (n - cornerIndexTopLeft() - 1);
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-centerx >= std::abs(py-centery) && px < ibxmax) ibxmax = px;
            if (px-centerx <= -std::abs(py-centery) && px > ibxmin) ibxmin = px;
            if (py-centery >= std::abs(px-centerx) && py < ibymax) ibymax = py;
            if (py-centery <= -std::abs(px-centerx) && py > ibymin) ibymin = py;
        }

        // store results in actual bound structures
        pixelInnerBoundsData[k].setXMin(ibxmin);
        pixelInnerBoundsData[k].setXMax(ibxmax);
        pixelInnerBoundsData[k].setYMin(ibymin);
        pixelInnerBoundsData[k].setYMax(ibymax);

        pixelOuterBoundsData[k].setXMin(obxmin);
        pixelOuterBoundsData[k].setXMax(obxmax);
        pixelOuterBoundsData[k].setYMin(obymin);
        pixelOuterBoundsData[k].setYMax(obymax);
    }

    template <typename T>
    void Silicon::updateGPU(ImageView<T> target)
    {
        int nxCenter = (_nx - 1) / 2;
        int nyCenter = (_ny - 1) / 2;

        // Now add in the displacements
        const int i1 = target.getXMin();
        const int i2 = target.getXMax();
        const int j1 = target.getYMin();
        const int j2 = target.getYMax();
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;
        const int step = target.getStep();
        const int stride = target.getStride();

        int nxny = nx * ny;
        
        int imageDataSize = (_delta.getXMax() - _delta.getXMin()) * _delta.getStep() + (_delta.getYMax() - _delta.getYMin()) * _delta.getStride();

        T* targetData = target.getData();
        
        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

        bool* changedData = _changed.get();
        
        // Loop through the boundary arrays and update any points affected by nearby pixels
        // Horizontal array first
        // map image data and changed array throughout all GPU loops

#pragma omp target teams distribute parallel for
        for (int p=0; p < nxny; p++) {
            // Calculate which pixel we are currently below
            int x = p % nx;
            int y = p / nx;

            // Loop over rectangle of pixels that could affect this row of points
            // std::min and std::max are causing this loop to crash, even though
            // the same functions run fine in the accumulate loop.
            int polyi1 = x - _qDist;
            if (polyi1 < 0) polyi1 = 0;
            int polyi2 = x + _qDist;
            if (polyi2 > (nx - 1)) polyi2 = nx - 1;
            int polyj1 = y - (_qDist + 1);
            if (polyj1 < 0) polyj1 = 0;
            int polyj2 = y + _qDist;
            if (polyj2 > (ny - 1)) polyj2 = ny - 1;

            // NB. We are working between rows y and y-1, so need polyj1 = y-1 - _qDist.

            bool change = false;
            for (int j=polyj1; j <= polyj2; j++) {
                for (int i=polyi1; i <= polyi2; i++) {
                    // Check whether this pixel has charge on it
                    double charge = targetData[(j * stride) + (i * step)];

                    if (charge != 0.0) {
                        change = true;

                        // Work out corresponding index into distortions array
                        int dist_index = (((y - j + nyCenter) * _nx) + (x - i + nxCenter)) * horizontalPixelStride();
                        int index = p * horizontalPixelStride();

                        // Loop over boundary points and update them
                        for (int n=0; n < horizontalPixelStride(); ++n, ++index, ++dist_index) {
                            horizontalBoundaryPointsData[index].x =
                                double(horizontalBoundaryPointsData[index].x) +
                                horizontalDistortionsData[dist_index].x * charge;
                            horizontalBoundaryPointsData[index].y =
                                double(horizontalBoundaryPointsData[index].y) +
                                horizontalDistortionsData[dist_index].y * charge;
                        }
                    }
                }
            }
            
            // update changed array
            if (change) {
                if (y < ny) changedData[(x * ny) + y] = true; // pixel above
                if (y > 0)  changedData[(x * ny) + (y - 1)] = true; // pixel below
            }
        }

        // Now vertical array
#pragma omp target teams distribute parallel for
        for (int p=0; p < (nx * ny); p++) {
            // Calculate which pixel we are currently on
            int x = p / ny;
            int y = (ny - 1) - (p % ny); // remember vertical points run top-to-bottom

            // Loop over rectangle of pixels that could affect this column of points
            int polyi1 = x - (_qDist + 1);
            if (polyi1 < 0) polyi1 = 0;
            int polyi2 = x + _qDist;
            if (polyi2 > (nx - 1)) polyi2 = nx - 1;
            int polyj1 = y - _qDist;
            if (polyj1 < 0) polyj1 = 0;
            int polyj2 = y + _qDist;
            if (polyj2 > (ny - 1)) polyj2 = ny - 1;

            bool change = false;
            for (int j=polyj1; j <= polyj2; j++) {
                for (int i=polyi1; i <= polyi2; i++) {
                    // Check whether this pixel has charge on it
                    double charge = targetData[(j * stride) + (i * step)];

                    if (charge != 0.0) {
                        change = true;

                        // Work out corresponding index into distortions array
                        int dist_index = (((x - i + nxCenter) * _ny) + ((_ny - 1) - (y - j + nyCenter))) * verticalPixelStride() + (verticalPixelStride() - 1);
                        int index = p * verticalPixelStride() + (verticalPixelStride() - 1);

                        // Loop over boundary points and update them
                        for (int n=0; n < verticalPixelStride(); ++n, --index, --dist_index) {
                            verticalBoundaryPointsData[index].x =
                                double(verticalBoundaryPointsData[index].x) +
                                verticalDistortionsData[dist_index].x * charge;
                            verticalBoundaryPointsData[index].y =
                                double(verticalBoundaryPointsData[index].y) +
                                verticalDistortionsData[dist_index].y * charge;
                        }
                    }
                }
            }

            // update changed array
            if (change) {
                if (x < nx) changedData[(x * ny) + y] = true;
                if (x > 0)  changedData[((x - 1) * ny) + y] = true;
            }
        }

        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();
#pragma omp target teams distribute parallel for
        for (size_t k=0; k<nxny; ++k) {
            if (changedData[k]) {
                updatePixelBoundsGPU(nx, ny, k, pixelInnerBoundsData,
                                     pixelOuterBoundsData,
                                     horizontalBoundaryPointsData,
                                     verticalBoundaryPointsData);
                changedData[k] = false;
            }
        }
        
        // update target from delta and zero delta on GPU
        // CPU delta is not zeroed but that shouldn't matter
        double* deltaData = _delta.getData();
#pragma omp target teams distribute parallel for
        for (int i = 0; i < imageDataSize; i++) {
            targetData[i] += deltaData[i];
            deltaData[i] = 0.0;
        }
    }

    template <typename T>
    void Silicon::update(ImageView<T> target)
    {
        updatePixelDistortions(_delta.view());
        target += _delta;
        _delta.setZero();
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

    template void Silicon::subtractDelta(ImageView<double> target);
    template void Silicon::subtractDelta(ImageView<float> target);
    template void Silicon::addDelta(ImageView<double> target);
    template void Silicon::addDelta(ImageView<float> target);

    template void Silicon::initialize(ImageView<double> target, Position<int> orig_center);
    template void Silicon::initialize(ImageView<float> target, Position<int> orig_center);

    template void Silicon::initializeGPU(ImageView<double> target, Position<int> orig_center);
    template void Silicon::initializeGPU(ImageView<float> target, Position<int> orig_center);

    template void Silicon::finalizeGPU(ImageView<double> target);
    template void Silicon::finalizeGPU(ImageView<float> target);

    template double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                                        BaseDeviate rng, ImageView<double> target);
    template double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                                        BaseDeviate rng, ImageView<float> target);

    template double Silicon::accumulateGPU(const PhotonArray& photons, int i1, int i2,
					   BaseDeviate rng, ImageView<double> target);
    template double Silicon::accumulateGPU(const PhotonArray& photons, int i1, int i2,
					   BaseDeviate rng, ImageView<float> target);

    template void Silicon::update(ImageView<double> target);
    template void Silicon::update(ImageView<float> target);

    template void Silicon::updateGPU(ImageView<double> target);
    template void Silicon::updateGPU(ImageView<float> target);

    template void Silicon::fillWithPixelAreas(ImageView<double> target, Position<int> orig_center,
                                              bool);
    template void Silicon::fillWithPixelAreas(ImageView<float> target, Position<int> orig_center,
                                              bool);

} // namespace galsim
