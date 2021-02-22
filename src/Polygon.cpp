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
 * Date: Jan 13, 2016
 * Polygon utilities
 */

//#define DEBUGLOGGING

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include "Std.h"
#include "Polygon.h"

namespace galsim {

    void Polygon::add(const Point& point)
    {
        xdbg<<"Current size = "<<_points.size()<<" = "<<_npoints<<std::endl;
        xdbg<<"add point "<<point.x<<','<<point.y<<std::endl;
        _points.push_back(point);
        ++_npoints;
        _sorted = false;
        _area = 0.0;
        xdbg<<"Done add."<<std::endl;
    }

    void Polygon::sort()
    {
        if (!_sorted && size() >= 3) {
            xdbg<<"Start Poly sort"<<std::endl;
            // calculate centroid of the polygon
            double cx = 0.0;
            double cy = 0.0;
            for (int i=0; i<_npoints; i++) {
                cx += _points[i].x;
                cy += _points[i].y;
            }
            cx /= _npoints;
            cy /= _npoints;

            // Calculate angle between centroid and point
            std::vector<double> angle(_npoints);
            for (int i=0; i<_npoints; i++) {
                angle[i] = std::atan2(_points[i].y - cy, _points[i].x - cx);
            }

            //sort _points in place using the angles
            for (int j=0; j<_npoints-1; j++) {
                int imin = std::min_element(angle.begin(),angle.end()) - angle.begin();
                // min_element returns a pointer to the minimum, given pointers to the start and end
                std::swap(_points[j], _points[imin]);
                angle[imin] = angle[j];
                angle[j] = 20.0;
            }
            // Also update the bounds now.
            updateBounds();
        }
        _sorted = true;
    }

    double Polygon::area() const
    {
        if (_area == 0.) {
            xdbg<<"Start Poly area"<<std::endl;
            assert(_sorted);
            // Calculates the area of a polygon using the shoelace algorithm
            for (int i=0; i<_npoints; i++) {
                int j = (i + 1) % _npoints;
                _area += _points[i].x * _points[j].y;
                _area -= _points[j].x * _points[i].y;
            }
            _area = std::abs(_area) / 2.0;
        }
        return _area;
    }

    bool Polygon::contains(const Point& point) const
    {
        //Determines if a given point is inside the polygon
        assert(_sorted);
        if (triviallyContains(point)) return true;
        if (!mightContain(point)) return false;
        double x1 = _points[0].x;
        double y1 = _points[0].y;
        double xinters = 0.0;
        bool inside = false;
        for (int i=1; i<=_npoints; i++) {
            double x2 = _points[i % _npoints].x;
            double y2 = _points[i % _npoints].y;
            if (point.y > std::min(y1,y2)) {
                if (point.y <= std::max(y1,y2)) {
                    if (point.x <= std::max(x1,x2)) {
                        if (y1 != y2) {
                            xinters = (point.y-y1)*(x2-x1)/(y2-y1)+x1;
                        }
                        if (x1 == x2 or point.x <= xinters) {
                            inside = !inside;
                        }
                    }
                }
            }
            x1 = x2;
            y1 = y2;
        }
        return inside;
    }

    void Polygon::scale(const Polygon& refpoly, const Polygon& emptypoly, double factor)
    {
        for (int i=0; i<_npoints; ++i) {
            _points[i].x = emptypoly[i].x + (refpoly[i].x - emptypoly[i].x) * factor;
            _points[i].y = emptypoly[i].y + (refpoly[i].y - emptypoly[i].y) * factor;
        }
        updateBounds();
    }

    void Polygon::distort(const Polygon& refpoly, double factor)
    {
        std::vector<Point>::iterator it = _points.begin();
        std::vector<Point>::const_iterator ref = refpoly._points.begin();
        for (int n=_npoints; n; --n) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            (*it).x += (*ref).x * factor;
#ifdef _OPENMP
#pragma omp atomic
#endif
            (*it).y += (*ref).y * factor;
            it++;
            ref++;
        }
    }

    void Polygon::updateBounds()
    {
#ifdef DEBUGGING
        xdbg<<"Start updateBounds:\n";
        for (int i=0; i<_npoints; ++i) {
            xdbg<<"   "<<_points[i]<<std::endl;
        }
#endif

        // The outer bounds are easy.  Just use the regular Bounds += operator.
        _outer = Bounds<double>();
        for (int i=0; i<_npoints; ++i) _outer += _points[i];
        xdbg<<"outer = "<<_outer<<std::endl;
        Position<double> center = _outer.center();

        // The inner bounds need to be done manually. Then the right side of the inner bounds is
        // the smallest x value from points with x-cenx >= |y-ceny|.  Likewise the other 3 sides.
        _inner = _outer;
        for (int i=0; i<_npoints; ++i) {
            double x = _points[i].x;
            double y = _points[i].y;
            if (x-center.x >= std::abs(y-center.y) && x < _inner.getXMax()) _inner.setXMax(x);
            if (x-center.x <= -std::abs(y-center.y) && x > _inner.getXMin()) _inner.setXMin(x);
            if (y-center.y >= std::abs(x-center.x) && y < _inner.getYMax()) _inner.setYMax(y);
            if (y-center.y <= -std::abs(x-center.x) && y > _inner.getYMin()) _inner.setYMin(y);
        }
        xdbg<<"inner = "<<_inner<<std::endl;

        // Mark the area as wrong if it was saved.
        _area = 0.;
    }
}
