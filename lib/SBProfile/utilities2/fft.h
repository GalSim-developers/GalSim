// 	$Id: fft.h,v 1.13 2012/01/09 16:28:41 garyb Exp $	 
// Objects that make use of 2d FFT's in VERSION 3 of the FFTW package.
// I am ASSUMING that the FFTW is set up to do double-precision.

// Notes:
//	*All tables have even dimensions (enforced on construction).
//	*The complex arrays (KTables) must be Hermitian, so transforms are real.
//	*All arrays are 0-indexed.
//      *FFTW arrays are stored in ROW-MAJOR order, meaning that in matrix notation, the most 
// rapidly varying index is the last one. However in an "image" view of the array, we would
// label this as the x value that is increasing along rows.  When doing real-to-complex
// transforms, the complex output contains (N/2+1) elements along this latter index, 0<=j<=N/2,
// and the j<0 elements must be obtained via conjugation. Imaginary parts of 0 and N/2 are zero.
//      *This interface will assume an "image" based convention in which all access to real
// or complex elements is in (ix, iy) format, and ix will be the rapidly varying index, with
// the k-space arrays being half-sized in the x direction.  
// **** so: when filling arrays, make ix your inner loop.  And provide only kx>=0 to fill KTable.
//	*xTable arrays have indices -N/2 <= ix,iy < N/2.  To store in FFTW arrays, which assume
// 0<=j < N, we add N/2 to indices before accessing FFTW arrays. This means that k
// values need to be multiplied by -1^(ix+iy) before/after transforms. 
//	*kTable arrays can be accessed by -N/2 <= jx, jy <= N/2.  FFTW puts DC at [0,0] element, 
// so the code in this class changes negative input indices to wrap them properly, also
// considering that jx<0 must be conjugated.
//	*"forward" transform, x->k, has -1 in exponent.
//	*value in the table must be multiplied by "scaleby" double to get
// the correctly dimensioned/scaled value.  Done automatically when getting/setting.  You'll get
// NaN's on some operations if scaleby becomes zero.


#ifndef FFT_H
#define FFT_H

#include <stdexcept>
#include <deque>
#include "Std.h"
#include "fftw3.h"
#include "Interpolant.h"

namespace fft {
// Class for errors
class FFTError: public std::runtime_error {
 public:
 FFTError(const string &m=""): std::runtime_error("FFT error: " + m) {}
};
class FFTOutofRange: public FFTError {
 public:
  FFTOutofRange(const string &m="value out of range"): FFTError(m) {}
};
class FFTMalloc: public FFTError {
 public:
  FFTMalloc(const string &m="malloc failure"): FFTError(m) {}
};
class FFTInvalid: public FFTError {
 public:
  FFTInvalid(const string &m="invalid plan or data"): FFTError(m) {}
};

  // A helper function that will return the smallest 2^n or 3x2^n value that is
  // even and >= the input integer.
  int goodFFTSize(int input);

// FFTW3 now states that C++ complex<double> will be bit-compatible with 
// the fftw_complex type.  So all interfaces will be through our DComplex.
// And the fftw real type is now just double.

class XTable;

// KTable is class holding k-space representation of real function.
// It will be based on an assumed Hermitian 2d square array.
// Table will be forced to be of even size.

class KTable {
  friend class XTable; 
 public:
  KTable(int _N, double _dk, DComplex _value=DComplex(0.,0.));
  KTable(const KTable& rhs): array(0), N(rhs.N), dk(rhs.dk), scaleby(rhs.scaleby) {
    copy_array(rhs);};
  KTable(): array(0), N(0), dk(0), scaleby(0) { }; // dummy constructor
  virtual KTable& operator=(const KTable& rhs) {
    if (&rhs==this) return *this;
    copy_array(rhs);
    N=rhs.N; dk=rhs.dk; scaleby=rhs.scaleby;
   return *this;};
  ~KTable() { kill_array(); }; 

  //  Fourier transform methods:
  // FFT to give pointer to a new XTable
  XTable* transform() const; 
  void transform(XTable& xt) const;
  // Have FFTW develop "wisdom" on doing this kind of transform
  void fftwMeasure() const;
  // This one does a "dumb" Fourier transform for a single (x,y) point:
  double xval(double x, double y) const;

  // Data access methods:
  // return value at grid point ix,iy (k = (ix*dk, iy*dk))
  DComplex kval(int ix, int iy) const; 
  // interpolate to k=(kx, ky) - WILL wrap k values to fill interpolant kernel
  DComplex interpolate(double kx, double ky, const Interpolant2d& interp) const;

  void kSet(int ix, int iy, DComplex value);

  void clear();  // Set all values to zero
  void accumulate(const KTable& rhs, double scalar=1.); // this += scalar*rhs
  void operator*=(double scalar) {scaleby *= scalar; cache.clear();}
  void operator*=(const KTable& rhs);

  // Produce a new KTable which wraps this one onto range +-Nout/2.  Nout will
  // be raised to even value.  In other words, aliases the data.
  KTable* wrap(int Nout) const;

  // Info about the table:  
  double getN() const {return N;}
  double getDk() const {return dk;}
  
  // Translate to move origin at (x0,y0)
  void translate(double x0, double y0);

  // Fill table from a function or function object:
  void fill( DComplex func(const double kx, const double ky)) ;
  template <class T> void fill( const T &f) ;
  // New table is function of this one:
  KTable* function( DComplex func(const double kx, 
				  const double ky, 
				  const DComplex val)) const ;
  // Integrate a function over d^2k:
  DComplex  integrate( DComplex func(const double kx, 
				     const double ky, 
				     const DComplex val)) const ;
  // Integrate KTable over d^2k (sum of all pixels * dk * dk)
  DComplex integratePixels() const;

 private:
  DComplex *array;
  double  dk;			//k-space increment
  int     N;			//Size in each dimension.
  double  scaleby;	//multiply table by this to get values
  size_t  index(int ix, int iy) const;	//Return index into data array.
  // this is also responsible for bounds checking.

  void copy_array(const KTable &rhs);	//copy an array
  void get_array(const DComplex value=DComplex(0.,0.));	//allocate an array  
  void kill_array();			//deallocate array
  void check_array() const {if (!array) throw FFTError("KTable operation on null array");}

  // Objects used to accelerate interpolation with seperable interpolants:
  mutable deque<DComplex> cache;
  mutable int cacheStartY;
  mutable double cacheX;
  mutable const InterpolantXY* cacheInterp;
};

// The x-space lookup table is a simple real matrix.  Force N even again,
// put origin at (N/2, N/2).
class XTable {
  friend class KTable;
 public:
  XTable(int _N, double _dx, double _value=0.);
  XTable(const XTable& rhs) : array(0), N(rhs.N), dx(rhs.dx), scaleby(rhs.scaleby) {
    copy_array(rhs);
  };
  XTable& operator=(const XTable& rhs) {
    if (&rhs==this) return *this;
    copy_array(rhs);
    N=rhs.N; dx=rhs.dx; scaleby=rhs.scaleby;
    return *this;};
  ~XTable() {kill_array();};

  ///// Fourier transforms:
  // FFT to give pointer to a new KTable
  KTable* transform() const;
  // Have FFTW develop "wisdom" on doing this kind of transform
  void fftwMeasure() const;
  // Do a "dumb" FT at a single frequency:
  DComplex kval(double kx, double ky) const; 

  ////// Data access:
  // get value at grid point x=(ix*dx, iy*dx)
  double xval(int ix, int iy) const; 
  // interpolate to (x,y) - will NOT wrap the x data around +-N/2
  double interpolate(double x, double y, const Interpolant2d& interp) const;
  void xSet(int ix, int iy, double value);

  void clear();  // Set all values to zero
  void accumulate(const XTable& rhs, double scalar=1.); // this += scalar*rhs
  void operator*=(double scalar) {scaleby *= scalar; cache.clear();}

  // Produce a new XTable which wraps this one onto range +-Nout/2.  Nout will
  // be raised to even value.  
  XTable* wrap(int Nout) const;

  ////// Info on the table:
  double getN() const {return N;}
  double getDx() const {return dx;}

  ///// Other operations:
  // Fill table from a function:
  void fill( double func(const double x, const double y)) ;
  // Integrate a (real) function over d^2x; set flag for sum:
  double  integrate( double func(const double kx, 
				 const double ky, 
				 const double val),
		     bool sumonly=false) const ;
  // Integrate XTable over d^2x (sum of all pixels * dx * dx)
  double integratePixels() const;

 private:
  double  scaleby;	//multiply table by this to get values
  double  dx;			//k-space increment
  int     N;			//Size in each dimension.
  double  *array;		//hold the values.

  size_t  index(int ix, int iy) const;	//Return index into data array.
  // this is also responsible for bounds checking.

  void get_array(const double value);	//allocate an array
  void copy_array(const XTable &rhs);	//copy an array
  void kill_array();			//deallocate array
  void check_array() const {if (!array) throw FFTError("KTable operation on null array");}

  // Objects used to accelerate interpolation with seperable interpolants:
  mutable deque<double> cache;
  mutable double cacheX;
  mutable int cacheStartY;
  mutable const InterpolantXY* cacheInterp;
};

// Fill table from a function class:
template <class T>
void
KTable::fill( const T& f) {
  cache.clear();	// invalidate any stored interpolations
  DComplex *zptr=array;
  double kx, ky;
  for (int iy=0; iy< N/2; iy++) {
   ky = iy*dk;
    for (int ix=0; ix< N/2+1 ; ix++) {
      kx = ix*dk;
      *(zptr++) = f(kx,ky);
    }
  }
  // wrap to the negative ky's
  for (int iy=-N/2; iy< 0; iy++) {
   ky = iy*dk;
    for (int ix=0; ix< N/2+1 ; ix++) {
      kx = ix*dk;
      *(zptr++) = f(kx,ky);
    }
  }
  return;
}

} // namespace fft

#endif
