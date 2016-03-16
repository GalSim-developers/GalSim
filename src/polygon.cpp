/*
  ------------------------------------------------------------------------------
  Author: Craig Lage, UC Davis
  Date: Jan 13, 2016
  Polygon utilities
*/

//****************** polygon.cpp **************************

#include <stdio.h>       
#include <stdlib.h>      
#include <algorithm>            // for min_element
#include <math.h>        
#include "polygon.h"

Point::Point(double xin, double yin)  //Constructor
{
  x = xin;
  y = yin;
  owner = this; 
}

Polygon::Polygon(int nmax)// Constructor - nmax is the max number of vertices
{
  npoints = 0;
  sorted = true;
  area = 0.0;
  pointlist = new Point*[nmax];
}

Polygon::~Polygon()// Destructor
{
  for (int i=0; i<npoints; i++) 
    {
      if (pointlist[i]->owner == this)
	{
	  delete pointlist[i];// Delete it only if we're the owner
	}
    }
  delete[] pointlist;
}

void Polygon::AddPoint(Point* point)
{
  pointlist[npoints] = point;
  npoints += 1;
  sorted = false;
  if (point->owner == point) point->owner = this;
  // Take ownership if it has no owner.
}

void Polygon::Sort()
{
  double cx, cy, angle[npoints];
  Point* oldpoint;
  int i, j, imin = 0;
  // calculate centroid of the polygon
  if (npoints < 3)
    {
      sorted = true;
      return;
    }
  else
    {
      cx = 0.0;
      cy = 0.0;
      for (i=0; i<npoints; i++)
	{
	  cx = cx + pointlist[i]->x;
	  cy = cy + pointlist[i]->y;
	}
      cx = cx / (double) npoints;
      cy = cy / (double) npoints;
      // Calculate angle between centroid and point
      for (i=0; i<npoints; i++)
	{
	  angle[i] = atan2(pointlist[i]->y - cy, pointlist[i]->x - cx);
	}
      //sort pointlist in place using the angles
      for (j=0; j<npoints-1; j++)
	{
	  imin = std::min_element(angle,angle+npoints) - angle;
	  // min_element returns a pointer to the minimum, given pointers to the start and end
	  oldpoint = pointlist[j];
	  pointlist[j] = pointlist[imin];
	  pointlist[imin] = oldpoint;
	  angle[imin] = angle[j];
	  angle[j] = 20.0;
	}
      sorted = true;
      return;
    }
}

double Polygon::Area()
{
  // Calculates the area of a polygon using the shoelace algorithm
  int i, j;
  if (! sorted) Sort(); //Polygon points must be in CCW order
  area = 0.0;
  for (i=0; i<npoints; i++)
    {
      j = (i + 1) % npoints;
      area += pointlist[i]->x * pointlist[j]->y;
      area -= pointlist[j]->x * pointlist[i]->y;
    }
  area = fabs(area) / 2.0;
  return area;
}

bool Polygon::PointInside(Point* point)
{
  //Determines if a given point is inside the polygon
  int i;
  bool inside = false;
  double x1, y1, x2, y2, xinters = 0.0;
  if (! sorted) Sort(); //Polygon points must be in CCW order
  x1 = pointlist[0]->x;
  y1 = pointlist[0]->y;
  for (i=1; i<npoints+1; i++)
    {
      x2 = pointlist[i % npoints]->x;
      y2 = pointlist[i % npoints]->y;
      if (point->y > fmin(y1,y2))
	{
	  if (point->y <= fmax(y1,y2))
	    {
	      if (point->x <= fmax(x1,x2))
		{
		  if (y1 != y2)
		    {
		      xinters = (point->y-y1)*(x2-x1)/(y2-y1)+x1;
		    }
		  if (x1 == x2 or point->x <= xinters)
		    {
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
