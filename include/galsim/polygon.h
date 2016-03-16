/*
  ------------------------------------------------------------------------------
  Author: Craig Lage, UC Davis
  Date: Jan 13, 2016
  Polygon utilities
*/

//****************** polygon.h **************************


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

