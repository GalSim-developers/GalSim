/* -*- c++ -*-
 * Copyright (c) 2012-2023 by the GalSim developers team on GitHub
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

#ifdef DEBUGLOGGING
// This function was used to help track down a memory leak.  Saving it for future resuse.

#include<mach/mach.h>

double RSS() {
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

    if (KERN_SUCCESS != task_info(mach_task_self(),
                                  TASK_BASIC_INFO, (task_info_t)&t_info,
                                  &t_info_count))
    {
        return -1;
    }

    // resident size is in t_info.resident_size;
    // virtual size is in t_info.virtual_size;
    return double(t_info.resident_size) / (1 << 30);  // 2^30 == GB
}
#endif

namespace galsim {
    // std::min and std::max are not supported in GPU offloaded code, so define our own
    // integer versions here. (C standard library provides floating point versions which do
    // work on GPU).
    int imin(int a, int b) { return a < b ? a : b; }
    int imax(int a, int b) { return a > b ? a : b; }

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
        _abs_length_table(abs_length_table), _transpose(transpose),
        _targetData(nullptr)
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

        // Next, we read in the pixel distortions from the Poisson_CCD simulations
        if (_transpose) std::swap(_nx,_ny);

        _horizontalDistortions.resize(horizontalRowStride(_nx) * (_ny + 1));
        _verticalDistortions.resize(verticalColumnStride(_ny) * (_nx + 1));
        _horizontalDistortions.shrink_to_fit();
        _verticalDistortions.shrink_to_fit();

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

        // Process _abs_length_table and _emptypoly ready for GPU
        // this will only be fully accurate for cases where the table uses linear
        // interpolation, and the data points are evenly spaced. Currently this is
        // always the case for _abs_length_table.
        _abs_length_arg_min = _abs_length_table.argMin();
        _abs_length_arg_max = _abs_length_table.argMax();
        _abs_length_size = _abs_length_table.size();

        _abs_length_table_GPU.resize(_abs_length_size);
        _abs_length_increment = (_abs_length_arg_max - _abs_length_arg_min) /
            (double)(_abs_length_size - 1);
        for (int i = 0; i < _abs_length_size; i++) {
            _abs_length_table_GPU[i] =
                _abs_length_table.lookup(_abs_length_arg_min + (((double)i) * _abs_length_increment));
        }

        _emptypolyGPU.resize(_emptypoly.size());
        for (int i=0; i<int(_emptypoly.size()); i++) {
            _emptypolyGPU[i].x = _emptypoly[i].x;
            _emptypolyGPU[i].y = _emptypoly[i].y;
        }
    }

    Silicon::~Silicon()
    {
        if (_targetData != nullptr) {
            finalize();
        }
    }

    void Silicon::updatePixelBounds(int nx, int ny, size_t k,
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
        pixelOuterBoundsData[k] = Bounds<double>();

        // iterate over pixel boundary
        int n, idx;
        // LHS lower half
        for (n = 0; n < cornerIndexBottomLeft(); n++) {
            idx = verticalPixelIndex(x, y, ny) + n + cornerIndexBottomLeft();
            pixelOuterBoundsData[k] += verticalBoundaryPointsData[idx];
        }
        // bottom row including corners
        for (; n <= cornerIndexBottomRight(); n++) {
            idx = horizontalPixelIndex(x, y, nx) + (n - cornerIndexBottomLeft());
            pixelOuterBoundsData[k] += horizontalBoundaryPointsData[idx];
        }
        // RHS
        for (; n < cornerIndexTopRight(); n++) {
            idx = verticalPixelIndex(x + 1, y, ny) + (cornerIndexTopRight() - n - 1);
            Position<double> p = verticalBoundaryPointsData[idx];
            p.x += 1.0;
            pixelOuterBoundsData[k] += p;
        }
        // top row including corners
        for (; n <= cornerIndexTopLeft(); n++) {
            idx = horizontalPixelIndex(x, y + 1, nx) + (cornerIndexTopLeft() - n);
            Position<double> p = horizontalBoundaryPointsData[idx];
            p.y += 1.0;
            pixelOuterBoundsData[k] += p;
        }
        // LHS upper half
        for (; n < _nv; n++) {
            idx = verticalPixelIndex(x, y, ny) + (n - cornerIndexTopLeft() - 1);
            pixelOuterBoundsData[k] += verticalBoundaryPointsData[idx];
        }

        Position<double> center = pixelOuterBoundsData[k].center();

        // compute inner bounds
        // initialize inner from outer
        double ibxmin = pixelOuterBoundsData[k].getXMin();
        double ibxmax = pixelOuterBoundsData[k].getXMax();
        double ibymin = pixelOuterBoundsData[k].getYMin();
        double ibymax = pixelOuterBoundsData[k].getYMax();

        // iterate over pixel boundary
        // LHS lower half
        for (n = 0; n < cornerIndexBottomLeft(); n++) {
            idx = verticalPixelIndex(x, y, ny) + n + cornerIndexBottomLeft();
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-center.x >= std::abs(py-center.y) && px < ibxmax) ibxmax = px;
            if (px-center.x <= -std::abs(py-center.y) && px > ibxmin) ibxmin = px;
            if (py-center.y >= std::abs(px-center.x) && py < ibymax) ibymax = py;
            if (py-center.y <= -std::abs(px-center.x) && py > ibymin) ibymin = py;
        }
        // bottom row including corners
        for (; n <= cornerIndexBottomRight(); n++) {
            idx = horizontalPixelIndex(x, y, nx) + (n - cornerIndexBottomLeft());
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y;
            if (px-center.x >= std::abs(py-center.y) && px < ibxmax) ibxmax = px;
            if (px-center.x <= -std::abs(py-center.y) && px > ibxmin) ibxmin = px;
            if (py-center.y >= std::abs(px-center.x) && py < ibymax) ibymax = py;
            if (py-center.y <= -std::abs(px-center.x) && py > ibymin) ibymin = py;
        }
        // RHS
        for (; n < cornerIndexTopRight(); n++) {
            idx = verticalPixelIndex(x + 1, y, ny) + (cornerIndexTopRight() - n - 1);
            double px = verticalBoundaryPointsData[idx].x + 1.0;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-center.x >= std::abs(py-center.y) && px < ibxmax) ibxmax = px;
            if (px-center.x <= -std::abs(py-center.y) && px > ibxmin) ibxmin = px;
            if (py-center.y >= std::abs(px-center.x) && py < ibymax) ibymax = py;
            if (py-center.y <= -std::abs(px-center.x) && py > ibymin) ibymin = py;
        }
        // top row including corners
        for (; n <= cornerIndexTopLeft(); n++) {
            idx = horizontalPixelIndex(x, y + 1, nx) + (cornerIndexTopLeft() - n);
            double px = horizontalBoundaryPointsData[idx].x;
            double py = horizontalBoundaryPointsData[idx].y + 1.0;
            if (px-center.x >= std::abs(py-center.y) && px < ibxmax) ibxmax = px;
            if (px-center.x <= -std::abs(py-center.y) && px > ibxmin) ibxmin = px;
            if (py-center.y >= std::abs(px-center.x) && py < ibymax) ibymax = py;
            if (py-center.y <= -std::abs(px-center.x) && py > ibymin) ibymin = py;
        }
        // LHS upper half
        for (; n < _nv; n++) {
            idx = verticalPixelIndex(x, y, ny) + (n - cornerIndexTopLeft() - 1);
            double px = verticalBoundaryPointsData[idx].x;
            double py = verticalBoundaryPointsData[idx].y;
            if (px-center.x >= std::abs(py-center.y) && px < ibxmax) ibxmax = px;
            if (px-center.x <= -std::abs(py-center.y) && px > ibxmin) ibxmin = px;
            if (py-center.y >= std::abs(px-center.x) && py < ibymax) ibymax = py;
            if (py-center.y <= -std::abs(px-center.x) && py > ibymin) ibymin = py;
        }

        // store results in actual bound structure
        pixelInnerBoundsData[k].setXMin(ibxmin);
        pixelInnerBoundsData[k].setXMax(ibxmax);
        pixelInnerBoundsData[k].setYMin(ibymin);
        pixelInnerBoundsData[k].setYMax(ibymax);
    }

    template <typename T>
    void Silicon::updatePixelDistortions(ImageView<T> target)
    {
        dbg<<"updatePixelDistortions\n";
        // This updates the pixel distortions in the linear boundary arrays
        // based on the amount of additional charge in each pixel
        // This distortion assumes the electron is created at the
        // top of the silicon.  It mus be scaled based on the conversion depth
        // This is handled in insidePixel.
        int nxCenter = (_nx - 1) / 2;
        int nyCenter = (_ny - 1) / 2;

        // Now add in the displacements
        const int nx = target.getNCol();
        const int ny = target.getNRow();
        const int step = target.getStep();
        const int stride = target.getStride();

        T* targetData = target.getData();
        const int npix = nx * ny;

        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

        bool* changedData = _changed.get();

        // Loop through the boundary arrays and update any points affected by nearby pixels
        // Horizontal array first
        // map image data and changed array throughout all GPU loops

#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for
#else
#pragma omp target teams distribute parallel for
#endif
#endif
        for (int p=0; p < npix; p++) {
            // Calculate which pixel we are currently below
            int x = p % nx;
            int y = p / nx;

            // Loop over rectangle of pixels that could affect this row of points
            int polyi1 = imax(x - _qDist, 0);
            int polyi2 = imin(x + _qDist, nx - 1);
            // NB. We are working between rows y and y-1, so need polyj1 = y-1 - _qDist.
            int polyj1 = imax(y - (_qDist + 1), 0);
            int polyj2 = imin(y + _qDist, ny - 1);

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
#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for
#else
#pragma omp target teams distribute parallel for
#endif
#endif
        for (int p=0; p < (nx * ny); p++) {
            // Calculate which pixel we are currently on
            int x = p / ny;
            int y = (ny - 1) - (p % ny); // remember vertical points run top-to-bottom

            // Loop over rectangle of pixels that could affect this column of points
            int polyi1 = imax(x - (_qDist + 1), 0);
            int polyi2 = imin(x + _qDist, nx - 1);
            int polyj1 = imax(y - _qDist, 0);
            int polyj2 = imin(y + _qDist, ny - 1);

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
#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for
#else
#pragma omp target teams distribute parallel for
#endif
#endif
        for (int k=0; k<npix; ++k) {
            if (changedData[k]) {
                updatePixelBounds(nx, ny, k, pixelInnerBoundsData,
                                  pixelOuterBoundsData,
                                  horizontalBoundaryPointsData,
                                  verticalBoundaryPointsData);
                changedData[k] = false;
            }
        }
    }

    // This version of calculateTreeRingDistortion only distorts a polygon.
    // Used in the no-flux pixel area calculation.
    void Silicon::calculateTreeRingDistortion(int i, int j, Position<int> orig_center,
                                              Polygon& poly) const
    {
        for (int n=0; n<_nv; n++) {
            xdbg<<"i,j,n = "<<i<<','<<j<<','<<n<<": x,y = "<<
                poly[n].x <<"  "<< poly[n].y<<std::endl;
            double tx = (double)i + poly[n].x - _treeRingCenter.x + (double)orig_center.x;
            double ty = (double)j + poly[n].y - _treeRingCenter.y + (double)orig_center.y;
            xdbg<<"tx,ty = "<<tx<<','<<ty<<std::endl;
            double r = sqrt(tx * tx + ty * ty);
            if (r > 0 && r < _tr_radial_table.argMax()) {
                double shift = _tr_radial_table.lookup(r);
                xdbg<<"r = "<<r<<", shift = "<<shift<<std::endl;
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
        iteratePixelBoundary(
            i-i1, j-j1, nx, ny, [&](int n, Position<float>& pt, bool rhs, bool top) {
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
                if (r > 0 && r < _tr_radial_table.argMax()) {
                    double shift = _tr_radial_table.lookup(r);
                    //xdbg<<"r = "<<r<<", shift = "<<shift<<std::endl;
                    // Shifts are along the radial vector in direction of the doping gradient
                    double dx = shift * tx / r;
                    double dy = shift * ty / r;
                    //xdbg<<"dx,dy = "<<dx<<','<<dy<<std::endl;
                    pt.x += dx;
                    pt.y += dy;
                }
            }
        );
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
        const int nx = target.getNCol();
        const int ny = target.getNRow();
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
                updatePixelBounds(nx, ny, k, _pixelInnerBounds.data(),
                                  _pixelOuterBounds.data(),
                                  _horizontalBoundaryPoints.data(),
                                  _verticalBoundaryPoints.data());
            }
        }
    }

    // Scales a linear pixel boundary into a polygon object.
    void Silicon::scaleBoundsToPoly(int i, int j, int nx, int ny,
                                    const Polygon& emptypoly, Polygon& result,
                                    double factor) const
    {
        result = emptypoly;

        iteratePixelBoundary(
            i, j, nx, ny,
            [&](int n, const Position<float>& pt, bool rhs, bool top) {
                Position<double> p = pt;
                if (rhs) p.x += 1.0;
                if (top) p.y += 1.0;
                result[n].x += (p.x - emptypoly[n].x) * factor;
                result[n].y += (p.y - emptypoly[n].y) * factor;
            }
        );

        result.updateBounds();
    }

    bool Silicon::insidePixel(int ix, int iy, double x, double y, double zconv,
                              Bounds<int>& targetBounds, bool* off_edge,
                              int emptypolySize,
                              Bounds<double>* pixelInnerBoundsData,
                              Bounds<double>* pixelOuterBoundsData,
                              Position<float>* horizontalBoundaryPointsData,
                              Position<float>* verticalBoundaryPointsData,
                              Position<double>* emptypolyData) const
    {
        // This scales the pixel distortion based on the zconv, which is the depth
        // at which the electron is created, and then tests to see if the delivered
        // point is inside the pixel.
        // (ix,iy) is the pixel being tested, and (x,y) is the coordinate of the
        // photon within the pixel, with (0,0) in the lower left

        // If test pixel is off the image, return false.  (Avoids seg faults!)
        if (!targetBounds.includes(ix, iy)) {
            if (off_edge) *off_edge = true;
            return false;
        }

        const int i1 = targetBounds.getXMin();
        const int i2 = targetBounds.getXMax();
        const int j1 = targetBounds.getYMin();
        const int j2 = targetBounds.getYMax();
        const int nx = i2-i1+1;
        const int ny = j2-j1+1;

        int index = (ix - i1) * ny + (iy - j1);

        // First do some easy checks if the point isn't terribly close to the boundary.

        bool inside;
        if (pixelInnerBoundsData[index].includes(x, y)) {
            inside = true;
        } else if (!pixelOuterBoundsData[index].includes(x, y)) {
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

#if 0
            // Old version that used a temporary polygon per thread (_testpoly)
#ifdef _OPENMP
            int t = omp_get_thread_num();
#else
            int t  = 0;
#endif
            // Scale the testpoly vertices by zfactor
            scaleBoundsToPoly(ix - i1, iy - j1, nx, ny, _emptypoly, _testpoly[t],
                              zfactor);

            // Now test to see if the point is inside
            Position<double> p(x, y);
            inside = _testpoly[t].contains(p);
#else
            // New version that doesn't use a temporary polygon object
            // This is required for GPU as due to the high number of threads,
            // having a temporary polygon per thread is not practical

            // compute first point of polygon (index 0)
            double x1=0, y1=0, xfirst, yfirst, xinters = 0.0;
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
                    xfirst = xp;
                    yfirst = yp;
                }
                else {
                    // shoelace algorithm
                    double x2 = xp;
                    double y2 = yp;
                    if (n == _nv) {
                        x2 = xfirst;
                        y2 = yfirst;
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
#endif
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

    // Helper function to calculate how far down into the silicon the photon converts into
    // an electron.

    double Silicon::calculateConversionDepth(bool photonsHasAllocatedWavelengths,
                                             const double* photonsWavelength,
                                             const double* abs_length_table_data,
                                             bool photonsHasAllocatedAngles,
                                             const double* photonsDXDZ,
                                             const double* photonsDYDZ, int i,
                                             double randomNumber) const
    {
        // Determine the distance the photon travels into the silicon
        double si_length;
        if (photonsHasAllocatedWavelengths) {
            double lambda = photonsWavelength[i]; // in nm
            // Lookup the absorption length in the imported table

            // perform abs_length_table lookup with linear interpolation
            int tableIdx = int((lambda - _abs_length_arg_min) / _abs_length_increment);
            double alpha = (lambda - ((float(tableIdx) * _abs_length_increment) +
                                      _abs_length_arg_min)) / _abs_length_increment;
            int tableIdx1 = tableIdx + 1;
            if (tableIdx <= 0) tableIdx = tableIdx1 = 0;
            if (tableIdx >= _abs_length_size-1) tableIdx = tableIdx1 = _abs_length_size - 1;
            double abs_length = (abs_length_table_data[tableIdx] * (1.0 - alpha)) +
                (abs_length_table_data[tableIdx1] * alpha); // in microns

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
        if (photonsHasAllocatedAngles) {
            double dxdz = photonsDXDZ[i];
            double dydz = photonsDYDZ[i];
            double dz = si_length / std::sqrt(1.0 + dxdz*dxdz + dydz*dydz); // in microns
#ifdef DEBUGLOGGING
            if (i % 1000 == 0) {
                xdbg<<"dxdz = "<<dxdz<<std::endl;
                xdbg<<"dydz = "<<dydz<<std::endl;
                xdbg<<"dz = "<<dz<<std::endl;
            }
#endif
            return std::fmin(_sensorThickness - 1.0, dz);  // max 1 micron from bottom
        } else {
            return si_length;
        }
    }

    bool searchNeighbors(const Silicon& silicon, int& ix, int& iy, double x, double y, double zconv,
                         Bounds<int>& targetBounds, int& step, int emptypolysize,
                         Bounds<double>* pixelInnerBoundsData,
                         Bounds<double>* pixelOuterBoundsData,
                         Position<float>* horizontalBoundaryPointsData,
                         Position<float>* verticalBoundaryPointsData,
                         Position<double>* emptypolyData)
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
            if (silicon.insidePixel(ix_off, iy_off, x_off, y_off, zconv,
                                    targetBounds, nullptr, emptypolysize,
                                    pixelInnerBoundsData, pixelOuterBoundsData,
                                    horizontalBoundaryPointsData,
                                    verticalBoundaryPointsData,
                                    emptypolyData)) {
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
        const int nx = target.getNCol();
        const int ny = target.getNRow();
        const int npix = nx * ny;

        if (use_flux) {
            dbg<<"Start full pixel area calculation\n";
            dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;
            dbg<<"total memory = "<<npix*_nv*sizeof(Position<float>)/(1024.*1024.)<<" MBytes"<<std::endl;

            // This will add distortions according to the current flux in the image, on the
            // GPU where appropriate.
            initialize(target, orig_center);

            // Copy the distorted pixel boundaries from GPU back to CPU if necessary.
#ifdef _OPENMP
#ifdef GALSIM_USE_GPU
            Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
            Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
            int hbpSize = _horizontalBoundaryPoints.size();
            int vbpSize = _verticalBoundaryPoints.size();
#pragma omp target update from(horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize])
#endif
#endif

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
        _horizontalBoundaryPoints.shrink_to_fit();
        _verticalBoundaryPoints.shrink_to_fit();

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
        _pixelInnerBounds.shrink_to_fit();
        _pixelOuterBounds.shrink_to_fit();
        for (int k = 0; k < (nx * ny); k++) {
            updatePixelBounds(nx, ny, k, _pixelInnerBounds.data(),
                              _pixelOuterBounds.data(),
                              _horizontalBoundaryPoints.data(),
                              _verticalBoundaryPoints.data());
        }
    }

    template <typename T>
    void Silicon::initialize(ImageView<T> target, Position<int> orig_center)
    {
        // release old GPU storage if allocated
        if (_targetData != nullptr) {
            finalize();
        }

        // work out minimum and maximum addresses of image data in memory
        T *targetDataStart, *targetDataEnd;
        int step = target.getStep();
        int stride = target.getStride();
        T *data = target.getData();
        int ncol = target.getNCol();
        int nrow = target.getNRow();

        if (stride >= 0) {
            if (step >= 0) {
                // positive step and stride
                targetDataStart = data;
                targetDataEnd = data + (ncol - 1) * step + (nrow - 1) * stride + 1;
            }
            else {
                // negative step, positive stride
                targetDataStart = data + (ncol - 1) * step;
                targetDataEnd = data + (nrow - 1) * stride + 1;
            }
        }
        else {
            if (step >= 0) {
                // positive step, negative stride
                targetDataStart = data + (nrow - 1) * stride;
                targetDataEnd = data + (ncol - 1) * step + 1;
            }
            else {
                // negative step and stride
                targetDataStart = data + (ncol - 1) * step + (nrow - 1) * stride;
                targetDataEnd = data + 1;
            }
        }
        _targetDataLength = targetDataEnd - targetDataStart;

        // and store target image pointer and type for later
        _targetData = static_cast<void*>(targetDataStart);
        _targetIsDouble = (sizeof(T) == 8);

        Bounds<int> b = target.getBounds();
        if (!b.isDefined())
            throw std::runtime_error("Attempting to PhotonArray::addTo an Image with"
                                     " undefined Bounds");

        const int nx = target.getNCol();
        const int ny = target.getNRow();
        dbg<<"nx,ny = "<<nx<<','<<ny<<std::endl;

        initializeBoundaryPoints(nx, ny);

        dbg<<"Built poly list\n";
        // Now we add in the tree ring distortions
        addTreeRingDistortions(target, orig_center);

        // Keep track of the charge we are accumulating on a separate image for efficiency
        // of the distortion updates.
        _delta.resize(b);
        _delta.setZero();

        int npix = nx * ny;
        _changed.reset(new bool[npix]);
        bool* changedData = _changed.get();
        for (int i=0; i<npix; i++) changedData[i] = false;

#ifdef GALSIM_USE_GPU
        // Map data to GPU
        double* deltaData = _delta.getData();

        int pixelBoundsSize = _pixelInnerBounds.size();

        int hbpSize = _horizontalBoundaryPoints.size();
        int vbpSize = _verticalBoundaryPoints.size();

        int hdSize = _horizontalDistortions.size();
        int vdSize = _verticalDistortions.size();

        double* abs_length_table_data = _abs_length_table_GPU.data();

        int emptypolySize = _emptypoly.size();
        Position<double>* emptypolyData = _emptypolyGPU.data();

        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();

        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

#pragma omp target enter data map(to: this[:1], deltaData[0:npix], targetDataStart[0:_targetDataLength], pixelInnerBoundsData[0:pixelBoundsSize], pixelOuterBoundsData[0:pixelBoundsSize], horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize], abs_length_table_data[0:_abs_length_size], emptypolyData[0:emptypolySize], horizontalDistortionsData[0:hdSize], verticalDistortionsData[0:vdSize], changedData[0:npix])
#endif

        // Start with the correct distortions for the initial image as it is already
        dbg<<"Initial updatePixelDistortions\n";
        updatePixelDistortions(target);
    }

    void Silicon::finalize()
    {
#ifdef GALSIM_USE_GPU
        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();

        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();
        Position<float>* horizontalDistortionsData = _horizontalDistortions.data();
        Position<float>* verticalDistortionsData = _verticalDistortions.data();

        double* abs_length_table_data = _abs_length_table_GPU.data();

        const int nx = _delta.getNCol();
        const int ny = _delta.getNRow();
        int npix = nx * ny;

        int pixelBoundsSize = _pixelInnerBounds.size();

        int hbpSize = _horizontalBoundaryPoints.size();
        int vbpSize = _verticalBoundaryPoints.size();

        int hdSize = _horizontalDistortions.size();
        int vdSize = _verticalDistortions.size();

        int emptypolySize = _emptypoly.size();
        Position<double>* emptypolyData = _emptypolyGPU.data();

        bool* changedData = _changed.get();

        double* deltaData = _delta.getData();

        if (_targetIsDouble) {
            double* targetData = static_cast<double*>(_targetData);
#pragma omp target exit data map(release: this[:1], deltaData[0:npix], targetData[0:_targetDataLength], pixelInnerBoundsData[0:pixelBoundsSize], pixelOuterBoundsData[0:pixelBoundsSize], horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize], abs_length_table_data[0:_abs_length_size], emptypolyData[0:emptypolySize], horizontalDistortionsData[0:hdSize], verticalDistortionsData[0:vdSize], changedData[0:npix])
        }
        else {
            float* targetData = static_cast<float*>(_targetData);
#pragma omp target exit data map(release: this[:1], deltaData[0:npix], targetData[0:_targetDataLength], pixelInnerBoundsData[0:pixelBoundsSize], pixelOuterBoundsData[0:pixelBoundsSize], horizontalBoundaryPointsData[0:hbpSize], verticalBoundaryPointsData[0:vbpSize], abs_length_table_data[0:_abs_length_size], emptypolyData[0:emptypolySize], horizontalDistortionsData[0:hdSize], verticalDistortionsData[0:vdSize], changedData[0:npix])
        }
#endif
    }

    // The adDelta and subtractDelta functions are slightly complicated, so rather than
    // repeat the code twice, we do all the looping logic once, and use templates to
    // let the compiler turn this into 3 different functions with slight differences.
    // I.e. plus, zero_delta = (true, true), (true, false), and (false, false).
    template <bool plus, bool zero_delta, typename T>
    void _addDelta(ImageView<T> target, ImageAlloc<double>& _delta)
    {
        assert(_delta.isContiguous());

        double* deltaData = _delta.getData();
        T* targetData = target.getData();
        const int skip = target.getNSkip();
        const int step = target.getStep();
        const int nrow = target.getNRow();
        const int ncol = target.getNCol();
        const int npix = ncol*nrow;

        assert(targetData + (nrow-1)*skip + ncol*step <= target.getMaxPtr() || step<0 || skip<0);
        assert(deltaData + nrow*ncol <= _delta.getMaxPtr());
        if (step == 1) {
#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for
#else
#pragma omp target teams distribute parallel for
#endif
#endif
            for (int p=0; p<npix; p++) {
                // If step == 1, then for the first row, k = p.
                // After the first row, there may be additional skips of j*skip.
                int k = p + (p/ncol)*skip;
                // NB. The compiler will optimize these branches away,
                //     since plus and zero_delta are compile-time constants.
                if (plus)
                    targetData[k] += deltaData[p];
                else
                    targetData[k] -= deltaData[p];
                if (zero_delta) deltaData[p] = 0.0;
            }
        } else {
#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for
#else
#pragma omp target teams distribute parallel for
#endif
#endif
            for (int p=0; p<npix; p++) {
                // If step != 1, then for the first row, k = p * step.
                // After the first row, there may be additional skips of j * skip.
                int k = p*step + (p/ncol)*skip;
                if (plus)
                    targetData[k] += deltaData[p];
                else
                    targetData[k] -= deltaData[p];
                if (zero_delta) deltaData[p] = 0.0;
            }
        }
    }

    template <typename T>
    void Silicon::addDelta(ImageView<T> target)
    {
        // This would be (and was before v2.5) simply:
        //   target += _delta;
        // But this doesn't port to the GPU.  To make this work on the GPU,
        // we need to unroll that function and work with the arrays directly.

        // When calling this from python, we don't want to zero the current delta values.
        // Hence subtract_delta=false.
        // (The first true means add, not subtract.)
        _addDelta<true, false>(target, _delta);

        // And if doing this on the GPU, we need to copy back to the CPU now.
#ifdef GALSIM_USE_GPU
        T* targetData = static_cast<T*>(_targetData);
#pragma omp target update from(targetData[0:_targetDataLength])
#endif
    }

    template <typename T>
    void Silicon::subtractDelta(ImageView<T> target)
    {
        // add=false
        // subtract_delta=false
        _addDelta<false, false>(target, _delta);

        // And if doing this on the GPU, we need to copy back to the CPU now.
#ifdef GALSIM_USE_GPU
        T* targetData = static_cast<T*>(_targetData);
#pragma omp target update from(targetData[0:_targetDataLength])
#endif
    }

    template <typename T>
    double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                               BaseDeviate rng, ImageView<T> target)
    {
        const int nphotons = i2 - i1;

        // Generate random numbers in advance for reproducibility across CPU, GPU,
        // different numbers of threads
        // we store four random numbers for each photon in a single array.
        // using separate arrays would require too many arguments for the OpenMP
        // kernel on GPU (at least with the Clang runtime)
        std::vector<double> randomValues(nphotons * 4);

        UniformDeviate ud(rng);
        GaussianDeviate gd(ud, 0, 1);

        for (int i=0; i<nphotons; i++) {
            randomValues[i*4] = gd();    // diffStep x
            randomValues[i*4+1] = gd();  // diffstep y
            randomValues[i*4+2] = ud();  // pixel not found
            randomValues[i*4+3] = ud();  // conversion depth
        }

        const double invPixelSize = 1./_pixelSize; // pixels/micron
        const double diffStep_pixel_z = _diffStep / (_sensorThickness * _pixelSize);
        Bounds<int> b = target.getBounds();
        double addedFlux = 0.;

        // Get everything out of C++ classes and into arrays/structures suitable for GPU.
        // Mapping to GPU requires raw pointers - std::vector and similar objects cannot
        // presently be mapped correctly.
        // photons
        const double* photonsX = photons.getXArray();
        const double* photonsY = photons.getYArray();
        const double* photonsDXDZ = photons.getDXDZArray();
        const double* photonsDYDZ = photons.getDYDZArray();
        const double* photonsFlux = photons.getFluxArray();
        const double* photonsWavelength = photons.getWavelengthArray();
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

        // random array
        double* randomArray = randomValues.data();

        // delta image
        int deltaXMin = _delta.getXMin();
        int deltaYMin = _delta.getYMin();
        int deltaStep = _delta.getStep();
        int deltaStride = _delta.getStride();

        int emptypolySize = _emptypoly.size();

        double* deltaData = _delta.getData();
        Bounds<double>* pixelInnerBoundsData = _pixelInnerBounds.data();
        Bounds<double>* pixelOuterBoundsData = _pixelOuterBounds.data();
        Position<float>* horizontalBoundaryPointsData = _horizontalBoundaryPoints.data();
        Position<float>* verticalBoundaryPointsData = _verticalBoundaryPoints.data();

        double* abs_length_table_data = _abs_length_table_GPU.data();

        Position<double>* emptypolyData = _emptypolyGPU.data();

#ifdef _OPENMP
#ifndef GALSIM_USE_GPU
#pragma omp parallel for reduction(+:addedFlux)
#else
#pragma omp target teams distribute parallel for map(to: photonsX[i1:i2-i1], photonsY[i1:i2-i1], photonsDXDZ[i1:i2-i1], photonsDYDZ[i1:i2-i1], photonsFlux[i1:i2-i1], photonsWavelength[i1:i2-i1], randomArray[0:(i2-i1)*4]) reduction(+:addedFlux)
#endif
#endif
        for (int i = i1; i < i2; i++) {
            // Get the location where the photon strikes the silicon:
            double x0 = photonsX[i]; // in pixels
            double y0 = photonsY[i]; // in pixels
            xdbg<<"x0,y0 = "<<x0<<','<<y0;

            // get uniform random number for conversion depth from randomArray
            // (4th of 4 numbers for this photon)
            double dz = calculateConversionDepth(photonsHasAllocatedWavelengths,
                                                 photonsWavelength,
                                                 abs_length_table_data,
                                                 photonsHasAllocatedAngles,
                                                 photonsDXDZ,
                                                 photonsDYDZ, i,
                                                 randomArray[(i - i1) * 4 + 3]);
            if (photonsHasAllocatedAngles) {
                double dxdz = photonsDXDZ[i];
                double dydz = photonsDYDZ[i];
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
                double diffStep = std::fmax(0.0, diffStep_pixel_z * std::sqrt(zconv * _sensorThickness));
                // use gaussian random numbers for diffStep from randomArray
                // (1st and 2nd of 4 numbers for this photon)
                x0 += diffStep * randomArray[(i-i1)*4];
                y0 += diffStep * randomArray[(i-i1)*4+1];
            }
            xdbg<<" => "<<x0<<','<<y0<<std::endl;
            double flux = photonsFlux[i];

#ifdef DEBUGLOGGING
            if (i % 1000 == 0) {
                xdbg<<"diffStep = "<<_diffStep<<std::endl;
                xdbg<<"zconv = "<<zconv<<std::endl;
                xdbg<<"x0 = "<<x0<<std::endl;
                xdbg<<"y0 = "<<y0<<std::endl;
            }
#endif

            // Now we find the undistorted pixel
            int ix = int(std::floor(x0 + 0.5));
            int iy = int(std::floor(y0 + 0.5));

            double x = x0 - ix + 0.5;
            double y = y0 - iy + 0.5;
            // (ix,iy) are the undistorted pixel coordinates.
            // (x,y) are the coordinates within the pixel, centered at the lower left

            // First check the obvious choice, since this will usually work.
            bool off_edge;
            bool foundPixel;

            foundPixel = insidePixel(ix, iy, x, y, zconv, b, &off_edge,
                                     emptypolySize, pixelInnerBoundsData,
                                     pixelOuterBoundsData,
                                     horizontalBoundaryPointsData,
                                     verticalBoundaryPointsData,
                                     emptypolyData);

            // If the nominal position is on the edge of the image, off_edge reports whether
            // the photon has fallen off the edge of the image. In this case, we won't find it in
            // any of the neighbors either.  Just let the photon fall off the edge in this case.
            if (!foundPixel && off_edge) continue;

            // Then check neighbors
            int step;  // We might need this below, so let searchNeighbors return it.
            if (!foundPixel) {
                foundPixel = searchNeighbors(*this, ix, iy, x, y, zconv,
                                             b, step, emptypolySize,
                                             pixelInnerBoundsData,
                                             pixelOuterBoundsData,
                                             horizontalBoundaryPointsData,
                                             verticalBoundaryPointsData,
                                             emptypolyData);
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
                insidePixel(ix, iy, x, y, zconv, b, &off_edge,
                            emptypolySize, pixelInnerBoundsData,
                            pixelOuterBoundsData,
                            horizontalBoundaryPointsData,
                            verticalBoundaryPointsData,
                            emptypolyData);
                searchNeighbors(*this, ix, iy, x, y, zconv, b, step, emptypolySize,
                                pixelInnerBoundsData, pixelOuterBoundsData,
                                horizontalBoundaryPointsData,
                                verticalBoundaryPointsData, emptypolyData);
                set_verbose(1);
#endif
                const int xoff[9] = {0,1,1,0,-1,-1,-1,0,1}; // Displacements to neighboring pixels
                const int yoff[9] = {0,0,1,1,1,0,-1,-1,-1}; // Displacements to neighboring pixels
                // use uniform random numbers for pixel not found from randomArray
                // (3rd of 4 numbers for this photon)
                int n = (randomArray[(i-i1) * 4 + 2] > 0.5) ? 0 : step;
                ix = ix + xoff[n];
                iy = iy + yoff[n];
            }

            if (b.includes(ix, iy)) {
                int deltaIdx = (ix - deltaXMin) * deltaStep + (iy - deltaYMin) * deltaStride;
#ifdef _OPENMP
#pragma omp atomic
#endif
                deltaData[deltaIdx] += flux;

                // This isn't atomic -- openmp is handling the reduction for us.
                addedFlux += flux;
            }
        }

        return addedFlux;
    }

    template <typename T>
    void Silicon::update(ImageView<T> target)
    {
        updatePixelDistortions(_delta.view());

        // The second true here indicates that we want to zero out the current _delta values
        // for the next round of photons (if any)
        // (The first true means add, not subtract.)
        _addDelta<true, true>(target, _delta);
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

    template double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                                        BaseDeviate rng, ImageView<double> target);
    template double Silicon::accumulate(const PhotonArray& photons, int i1, int i2,
                                        BaseDeviate rng, ImageView<float> target);

    template void Silicon::update(ImageView<double> target);
    template void Silicon::update(ImageView<float> target);

    template void Silicon::fillWithPixelAreas(ImageView<double> target, Position<int> orig_center,
                                              bool);
    template void Silicon::fillWithPixelAreas(ImageView<float> target, Position<int> orig_center,
                                              bool);

} // namespace galsim
