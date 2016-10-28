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

//****************** polygon.h **************************

#ifndef POLYGON_H
#define POLYGON_H

class Point
{
public:
    double x,y;
    void* owner;
    Point() {};
    Point(double, double);
};

class Polygon
{
public:
    int npoints;
    bool sorted;
    double area;
    Point** pointlist;
    Polygon() {};
    Polygon(int);// Constructor
    ~Polygon();  //Destructor
    void AddPoint(Point*);
    void Sort();
    double Area();
    bool PointInside(Point*);
};
#endif
