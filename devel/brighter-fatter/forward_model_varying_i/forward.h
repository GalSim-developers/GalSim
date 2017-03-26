/*
Craig Lage - 15-Jul-15

C++ code to calculate forward model fit
for Gaussian spots

file: forward.h

*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>            // for min_element
#include <string.h>
#include <math.h>
#define pi 4.0 * atan(1.0)     // Pi
#define max(x,y) (x>y?x:y)      // max macro definition
#define min(x,y) (x>y?y:x)      // min macro definition

using namespace std;

//  DATA STRUCTURES:  


class Array //This packages the 2D data sets which are brought from Python
{
 public:
  long nx, ny, nstamps;
  double xmin, xmax, ymin, ymax, dx, dy, *x, *y, *xoffset, *yoffset, *imax, *data;
  Array() {};
  ~Array(){};
};

// FUNCTION PROTOTYPES


double FOM(Array*, double, double);
double Area(double, double, double, double, double, double);




