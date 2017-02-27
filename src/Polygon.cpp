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
        dbg<<"Current size = "<<_points.size()<<std::endl;
        dbg<<"add point "<<point.x<<','<<point.y<<std::endl;
        _points.push_back(point);
        _sorted = false;
        _area = 0.0;
        dbg<<"Done add."<<std::endl;
    }

    void Polygon::sort()
    {
        if (!_sorted && size() >= 3) {
            dbg<<"Start Poly sort"<<std::endl;
            // calculate centroid of the polygon
            double cx = 0.0;
            double cy = 0.0;
            const int npoints = size();
            for (int i=0; i<npoints; i++) {
                cx += _points[i].x;
                cy += _points[i].y;
            }
            cx /= npoints;
            cy /= npoints;

            // Calculate angle between centroid and point
            std::vector<double> angle(npoints);
            for (int i=0; i<npoints; i++) {
                angle[i] = atan2(_points[i].y - cy, _points[i].x - cx);
            }

            //sort _points in place using the angles
            for (int j=0; j<npoints-1; j++) {
                int imin = std::min_element(angle.begin(),angle.end()) - angle.begin();
                // min_element returns a pointer to the minimum, given pointers to the start and end
                std::swap(_points[j], _points[imin]);
                angle[imin] = angle[j];
                angle[j] = 20.0;
            }
            _sorted = true;
        }
    }

    double Polygon::area()
    {
        if (_area == 0.) {
            dbg<<"Start Poly area"<<std::endl;
            // Calculates the area of a polygon using the shoelace algorithm
            sort(); //Polygon points must be in CCW order
            const int npoints = size();
            for (int i=0; i<npoints; i++) {
                int j = (i + 1) % npoints;
                _area += _points[i].x * _points[j].y;
                _area -= _points[j].x * _points[i].y;
            }
            _area = std::abs(_area) / 2.0;
        }
        return _area;
    }

    bool Polygon::contains(const Point& point)
    {
        //Determines if a given point is inside the polygon
        sort(); //Polygon points must be in CCW order
        double x1 = _points[0].x;
        double y1 = _points[0].y;
        double xinters = 0.0;
        bool inside = false;
        const int npoints = size();
        for (int i=1; i<=npoints; i++) {
            double x2 = _points[i % npoints].x;
            double y2 = _points[i % npoints].y;
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

    Polygon Polygon::scale(const Polygon& emptypoly, double zfactor) const
    {
        Polygon ret(*this);
        const int npoints = size();
        for (int n=0; n<npoints; n++) {
            ret._points[n].x = emptypoly[n].x + (_points[n].x - emptypoly[n].x) * zfactor;
            ret._points[n].y = emptypoly[n].y + (_points[n].y - emptypoly[n].y) * zfactor;
        }
        return ret;
    }
}
