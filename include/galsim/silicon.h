/*
  ------------------------------------------------------------------------------
  Author: Craig Lage, UC Davis
  Date: Mar 14, 2016
  Routines for integrating the CCD simulations into GalSim
*/

//****************** silicon.h **************************

#ifndef SILICON_H
#define SILICON_H

#include "polygon.h"
#include "Image.h"
#include "PhotonArray.h"

namespace galsim
{

    class Silicon
    {
    public:
        Polygon** polylist;
        Point* testpoint;
        double DiffStep, collXmin, collXwidth, collYmin, collYwidth;
        int Nx, Ny, Nv, NumVertices, NumElec;
        Silicon() {};
        Silicon(std::string); // Constructor
        ~Silicon();  // Destructor
        double random_gaussian(void);

        template <typename T>
        bool InsidePixel(int ix, int iy, double x, double y, double zconv,
                         ImageView<T> target) const;

        template <typename T>
        double accumulate(const PhotonArray& photons, UniformDeviate ud,
                          ImageView<T> target) const;
    };
}

#endif
