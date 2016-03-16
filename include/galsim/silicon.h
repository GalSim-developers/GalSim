/*
  ------------------------------------------------------------------------------
  Author: Craig Lage, UC Davis
  Date: Mar 14, 2016
  Routines for integrating the CCD simulations into GalSim
*/

//****************** silicon.h **************************

#include "polygon.h"
#include "Image.h"

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
     bool InsidePixel(int, int, double, double, double, ImageView<float>&);
     double random_gaussian(void);
 };
}  
