// Classes to represent lookup tables.
// A is the argument class, which must have ordering
// operations, and +-*/ for interpolation.
// D is the value class, which must have + and * operations
// to permit interpolation.
// 	$Id: Table.h,v 1.3 2012/01/01 18:31:33 garyb Exp $
#ifndef TABLE_H
#define TABLE_H

//** Omit - not needed generically: #include "Function1d.h"
#include "Std.h"

#include <vector>
using std::vector;
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>

// Exception classes:
class TableError: public std::runtime_error {
 public:
  TableError(const string &m=""): std::runtime_error("Table Error: " +m) {}
};
class TableOutOfRange: public TableError {
public:
  TableOutOfRange(): TableError("Argument out of range") {}
};
class TableReadError: public TableError {
public:
  TableReadError(const string &c): 
    TableError("Data read error for line ->"+c) {}
};

// Table element:
template<class V=double, class A=double>
class TableEntry {
public:
  TableEntry(A a, V v): arg(a), val(v) {}
  A arg;
  V val;
  bool operator==(const TableEntry rhs) const {return arg==rhs.arg;}
  bool operator==(const A rhs) const {return arg==rhs;}
  bool operator!=(const TableEntry rhs) const {return arg!=rhs.arg;}
  bool operator!=(const A rhs) const {return arg!=rhs;}
  bool operator>(const TableEntry rhs) const {return arg>rhs.arg;}
  bool operator>(const A rhs) const {return arg>rhs;}
  bool operator<(const TableEntry rhs) const {return arg<rhs.arg;}
  bool operator<(const A rhs) const {return arg<rhs;}
  bool operator<=(const TableEntry rhs) const {return arg<=rhs.arg;}
  bool operator<=(const A rhs) const {return arg>=rhs;}
  bool operator>=(const TableEntry rhs) const {return arg>=rhs.arg;}
  bool operator>=(const A rhs) const {return arg>=rhs;}
};

// The Table itself:
// Derive from Function1d if that has been defined, otherwise not needed.
template<class V=double, class A=double>
#ifdef FUNCTION1D_H
class Table: public Function1d<V,A> {
#else
class Table {
#endif
public:
  enum interpolant {linear, spline, floor, ceil};
  //Construct empty table
  Table(interpolant i=linear): v(), iType(i), isReady(false), y2() {} 
  //Table from two arrays:
  Table(const A* argvec, const V* valvec, int N, interpolant in=linear) ;
  Table(const vector<A> &a, const vector<V> &v, interpolant in=linear) ;
  Table(istream &is, interpolant in=linear): v(), iType(in), isReady(),
					     y2() {read(is);}
  void clear() {v.clear(); isReady=false;}
  void read(istream &is);
  void addEntry(const A a, const V v) ; //new element for table.
  V operator() (const A a) const ;	//lookup & interp. function value.
  V lookup(const A a) const ;	//interp, but exception if beyond bounds
  int size() const {return v.size();}	//size of table
  A argMin() const { setup(); if (v.size()>0) return v.front().arg;
   else throw TableError("argMin for null Table");}	//Smallest argument
  A argMax() const { setup(); if (v.size()>0) return v.back().arg;
   else throw TableError("argMax for null Table");} 	//Largest argument

  template <class T>
  void TransformVal(T &xfrm) {
    for (iter p=v.begin(); p!=v.end(); ++p)
      p->val = xfrm(p->arg, p->val);
    isReady=false; setup();
  }
  template <class T>
  void TransformArg(T &xfrm) {
    for (iter p=v.begin(); p!=v.end(); ++p)
      p->arg = xfrm(p->arg, p->val);
    isReady=false; setup();
  }

/**/void dump() const {setup(); for (citer p=v.begin(); p!=v.end(); ++p) 
		 cout << p->arg << " " << p->val << endl; }
private:
  typedef TableEntry<V,A> Entry;
  typedef typename vector<Entry>::const_iterator citer;
  typedef typename vector<Entry>::iterator iter;
  interpolant iType;

  mutable vector<Entry> v;
  mutable int	lastIndex;	//Index for last lookup into table.
  mutable bool  isReady;	//Flag if table has been prepped.
  mutable bool  equalSpaced;	//Flag set if arguments are nearly equally spaced.
  mutable A     dx;		// ...in which case this is argument interval
  mutable vector<V> y2;		//vector of 2nd derivs for spline
  //get index to 1st element >= argument.  Can throw the exception here.
  iter upperIndex(const A a) const;
  void sortIt() const {std::sort(v.begin(), v.end());}
  void setup() const;	//Do any necessary preparation;
//Interpolate value btwn p & --p:
  V interpolate(const A a, const citer p) const; 
};
#endif
